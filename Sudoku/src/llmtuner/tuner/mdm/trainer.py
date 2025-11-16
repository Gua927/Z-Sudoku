import os
import json
import torch
import numpy as np
import torch.nn as nn
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers import Trainer
from torch import nn
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from llmtuner.extras.constants import IGNORE_INDEX
from llmtuner.extras.logging import get_logger
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

logger = get_logger(__name__)


def _log_prefixed_metrics(trainer, metric_prefix: str, metrics: Dict[str, float]):
    """Log metrics again with slash-separated keys so wandb charts treat each variant as independent series."""
    if wandb is None or wandb.run is None:
        return

    def _to_scalar(value):
        return value.item() if hasattr(value, "item") else value

    loggable: Dict[str, float] = {}
    slash_prefix = metric_prefix.replace("_", "/")
    for key, value in metrics.items():
        if not key.startswith(metric_prefix):
            continue
        if not isinstance(value, (int, float)) and not hasattr(value, "item"):
            continue
        suffix = key[len(metric_prefix):].lstrip("_")
        log_key = slash_prefix if not suffix else f"{slash_prefix}/{suffix}"
        loggable[log_key] = _to_scalar(value)

    if loggable:
        trainer.log(loggable)

class CustomDiffusionTrainer(Trainer):
    def __init__(
        self,
        diff_args,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Optional[Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]] = None,
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        sampling_temperature: Optional[float] = None,
        eval_temperature: Optional[float] = None,
        eval_denoise_method: Optional[str] = None,
        eval_temperatures: Optional[List[float]] = None,
    ):
        optimizers = optimizers or (None, None)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.diff_args = diff_args
        fallback = getattr(diff_args, "temperature", None)
        temperature = sampling_temperature if sampling_temperature is not None else fallback
        temperature = 1.0 if temperature is None else float(temperature)
        if temperature < 0:
            raise ValueError("sampling_temperature must be non-negative.")
        self.temperature = temperature
        self.eval_temperature = eval_temperature
        self.eval_denoise_method = eval_denoise_method
        self.eval_temperature_list: List[Optional[float]] = list(eval_temperatures or [])
        self._manual_eval_temperature: Optional[float] = None
        self._manual_eval_denoise_method: Optional[str] = None
        print(self.diff_args)
        
        # 确保tokenizer配置文件被保存到输出目录
        if self.tokenizer is not None and hasattr(self.args, 'output_dir'):
            self.tokenizer.save_pretrained(self.args.output_dir)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: Optional[bool] = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        r"""
        Computes diffusion loss.
        """
        # 兼容新版transformers的compute_loss方法签名
        # num_items_in_batch参数在新版本中被添加，但我们不使用它
        final_loss = self.inner_forward(model, inputs)
        return final_loss

    @contextmanager
    def _apply_eval_overrides(self, temperature: Optional[float], method: Optional[str]):
        original_temperature = self.temperature
        original_method = getattr(self.diff_args, "denoise_method", None)
        try:
            if temperature is not None:
                self.temperature = float(temperature)
            if method is not None:
                self.diff_args.denoise_method = method
            yield
        finally:
            self.temperature = original_temperature
            if original_method is not None:
                self.diff_args.denoise_method = original_method

    def set_temporary_eval_overrides(self, temperature: Optional[float], denoise_method: Optional[str]):
        self._manual_eval_temperature = temperature
        self._manual_eval_denoise_method = denoise_method

    def clear_temporary_eval_overrides(self):
        self._manual_eval_temperature = None
        self._manual_eval_denoise_method = None

    @staticmethod
    def _unique_sequence(values: List[Optional[Union[str, float]]]) -> List[Optional[Union[str, float]]]:
        ordered: List[Optional[Union[str, float]]] = []
        for value in values:
            if value not in ordered:
                ordered.append(value)
        return ordered

    def _build_eval_values(self, single_value, multi_values: Optional[List[Optional[Union[str, float]]]]):
        collected: List[Optional[Union[str, float]]] = []
        if multi_values:
            collected.extend(multi_values)
        if single_value is not None:
            collected.append(single_value)
        if not collected:
            collected.append(None)
        return self._unique_sequence(collected)

    def _build_eval_combos(self, manual_override: bool) -> List[Tuple[Optional[str], Optional[float]]]:
        if manual_override:
            return [(self._manual_eval_denoise_method, self._manual_eval_temperature)]

        method_candidates = self._build_eval_values(
            self.eval_denoise_method,
            getattr(self.diff_args, "eval_denoise_methods", None)
        )
        temperature_candidates = self._build_eval_values(
            self.eval_temperature,
            self.eval_temperature_list,
        )
        return [
            (method, temperature)
            for method in method_candidates
            for temperature in temperature_candidates
        ]

    def _format_metric_prefix(
        self,
        base_prefix: str,
        index: int,
        method: Optional[str],
        temperature: Optional[float],
        multi_eval: bool
    ) -> str:
        if not multi_eval:
            return base_prefix

        parts: List[str] = [base_prefix or "eval"]
        method_label = method or getattr(self.diff_args, "denoise_method", None)
        if method_label:
            parts.append(f"method-{method_label}")

        temperature_label = temperature if temperature is not None else self.temperature
        if temperature_label is not None:
            parts.append(f"temp-{temperature_label:g}")

        if len(parts) == 1:
            parts.append(f"variant{index+1}")

        return "_".join(parts)

    def _is_default_combo(self, method: Optional[str], temperature: Optional[float]) -> bool:
        default_method = getattr(self.diff_args, "denoise_method", None)
        default_temp = self.temperature
        method_match = (method is None) or (default_method is None) or (method == default_method)
        temp_match = (temperature is None) or (default_temp is None) or (float(temperature) == float(default_temp))
        return method_match and temp_match

    @staticmethod
    def _alias_metrics(
        metrics: Dict[str, float],
        source_prefix: str,
        target_prefix: str
    ) -> Dict[str, float]:
        alias: Dict[str, float] = {}
        for key, value in metrics.items():
            if not key.startswith(source_prefix):
                continue
            alias_key = f"{target_prefix}{key[len(source_prefix):]}"
            alias[alias_key] = value
        return alias

    def _evaluate_single(
        self,
        eval_dataset: Optional[Dataset],
        ignore_keys: Optional[List[str]],
        metric_key_prefix: str,
        temperature: Optional[float],
        method: Optional[str]
    ) -> Dict[str, float]:
        with self._apply_eval_overrides(temperature, method):
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        manual_override = (self._manual_eval_temperature is not None) or (self._manual_eval_denoise_method is not None)
        eval_combos = self._build_eval_combos(manual_override)
        multi_eval = len(eval_combos) > 1
        combined_metrics: Dict[str, float] = {}
        alias_written = False

        for idx, (method, temperature) in enumerate(eval_combos):
            metric_prefix = self._format_metric_prefix(metric_key_prefix, idx, method, temperature, multi_eval)
            logger.info(
                "Running eval combo %d/%d: method=%s, temperature=%s, prefix=%s",
                idx + 1,
                len(eval_combos),
                method if method is not None else getattr(self.diff_args, "denoise_method", None),
                temperature if temperature is not None else self.temperature,
                metric_prefix,
            )
            metrics = self._evaluate_single(eval_dataset, ignore_keys, metric_prefix, temperature, method)
            combo_metrics = {k: v for k, v in metrics.items() if k.startswith(metric_prefix)}
            _log_prefixed_metrics(self, metric_prefix, combo_metrics)
            combined_metrics.update(combo_metrics)

            alias_needed = (
                multi_eval
                and not manual_override
                and not alias_written
                and metric_key_prefix
                and self._is_default_combo(method, temperature)
            )
            if alias_needed:
                alias_metrics = self._alias_metrics(combo_metrics, metric_prefix, metric_key_prefix)
                if alias_metrics:
                    self.log(alias_metrics)
                    combined_metrics.update(alias_metrics)
                    alias_written = True

        if manual_override:
            self.clear_temporary_eval_overrides()

        return combined_metrics
    
    def q_sample(self, inputs, t, maskable_mask):
        x_0 = inputs["input_ids"]
        u = torch.rand_like(x_0, dtype=torch.float) # t/T prob to mask
        t_mask = (u < ((t+1) / self.diff_args.diffusion_steps)[:, None]) & maskable_mask
        x_t = x_0.masked_fill(t_mask, self.tokenizer.mask_token_id)
        return x_t, t, t_mask  #  True means it's "MASK" token and should have loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        r"""
        rewrite prediction_step for eval loss
        """
        model.eval()
        labels = inputs['input_ids'].masked_fill(inputs['src_mask'].bool(), self.tokenizer.pad_token_id)
        with torch.no_grad():
            # import pdb; pdb.set_trace();
            final_loss = self.inner_forward(model, inputs)
            if prediction_loss_only:
                preds = None
            else:
                preds = self.generate_samples(inputs)
        # ignore the source part when calculating metric and saving
        preds = preds.masked_fill(inputs['src_mask'].bool(), self.tokenizer.pad_token_id)
        return final_loss, preds, labels
    
    def save_predictions(
        self,
        predict_results: "PredictionOutput"
    ) -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        preds = np.where(predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id)
        labels = np.where(predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for pred, label in zip(decoded_preds, decoded_labels):
                res.append(json.dumps({"label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    def inner_forward(
        self,
        model,
        inputs
    ):
        x = inputs["input_ids"]
        src_mask = inputs["src_mask"].bool()
        batch_size = x.size(0)

        if isinstance(model, DDP):
            vocab_size = model.module.vocab_size
        else:
            vocab_size = model.vocab_size
        num_timesteps = self.diff_args.diffusion_steps

        t = torch.randint(0, num_timesteps, (batch_size, ), device=x.device)
        x_t, t, loss_mask = self.q_sample(inputs, t, maskable_mask=~src_mask)

        attention_mask = torch.ones_like(x_t) 

        logits = model(x_t, t, attention_mask=attention_mask)
        logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)

        loss = F.cross_entropy(logits.reshape(-1, vocab_size), x.reshape(-1), reduction="none").float()   # num_masked samples
        
        loss = loss.masked_fill(~loss_mask.reshape(-1), 0)
        
        if self.diff_args.token_reweighting:
            loss = self.diff_args.alpha * (1 - torch.exp(-loss)) ** self.diff_args.gamma * loss

        if self.diff_args.time_reweighting == 'original':
            weight = 1 / (t+1)[:, None].float()
        elif self.diff_args.time_reweighting == 'linear':
            weight = (num_timesteps - t)[:, None].float()
        else:
            weight = t.new_ones((batch_size, 1)).float()

        weight = weight.expand(loss_mask.size())  # b,n, only [MASK] position have loss
        loss = (loss * weight.reshape(-1)).sum() / loss_mask.sum()   # avg token loss
        return loss


    def generate_samples(self, inputs):
        """
            select 1/T% tokens to denoise at each step
        """
        self.model.cuda()
        self.model.eval()
        verbose = not self.is_in_train
        # x = torch.transpose(torch.stack(inputs['input_ids']), 0, 1).cuda()
        # src_mask = torch.transpose(torch.stack(inputs['src_mask']), 0, 1).bool().cuda()
        x = inputs['input_ids'].cuda()
        src_mask = inputs['src_mask'].bool().cuda()
        attention_mask = torch.ones_like(x) 
        batch_size = x.size(0)

        init_maskable_mask = maskable_mask = ~src_mask
        denoise_method = getattr(self.diff_args, "denoise_method", "origin")
        decoding_schedule = self._extract_schedule(getattr(self.diff_args, "decoding_strategy", "linear"))
        
        for t in range(self.diff_args.diffusion_steps-1, -1, -1): # t from T-1 to 0
            with torch.no_grad():
                if t == self.diff_args.diffusion_steps-1:
                    # first forward, all position except src is [M]
                    xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)

                if verbose:
                    print(f"t={t+1}(in):", self.tokenizer.decode(xt.tolist()[0]))

                t_tensor = torch.full((batch_size, ), t, device=x.device)
                logits = self.model(xt, t_tensor, attention_mask=attention_mask)
                logits = torch.cat([logits[:,0:1], logits[:,:-1]], dim=1)

                scores = torch.log_softmax(logits, dim=-1)
                scores[:, :, self.tokenizer.vocab_size:] = -1000
                probs = scores.exp()
                x0_scores, _ = scores.max(-1)
                sampled_x0 = self._sample_tokens(probs)

                #### keep non-[MASK] positions as still
                x0 = xt.masked_scatter(maskable_mask, sampled_x0[maskable_mask])
                if verbose:
                    print(f"t={t+1}(out):", self.tokenizer.decode(x0.tolist()[0]))
                
                if t > 0:
                    selection_scores = None
                    if denoise_method == "topk":
                        selection_scores = x0_scores
                    elif denoise_method == "topk-margin":
                        selection_scores = self._compute_margin_scores(scores)

                    xt, maskable_mask = self._apply_denoise_strategy(
                        xt=xt,
                        x0=x0,
                        selection_scores=selection_scores,
                        maskable_mask=maskable_mask,
                        init_maskable_mask=init_maskable_mask,
                        t=t,
                        schedule=decoding_schedule,
                    )
                else:
                    xt = x0
        return xt

    def _apply_denoise_strategy(
        self,
        xt,
        x0,
        selection_scores,
        maskable_mask,
        init_maskable_mask,
        t,
        schedule,
    ):
        method = getattr(self.diff_args, "denoise_method", "origin")
        if method == "origin":
            return self._origin_denoise_step(xt, x0, maskable_mask, t)

        if method in {"topk", "topk-margin"}:
            if selection_scores is None:
                raise ValueError("Selection scores are required for topk-based denoising.")
            xt = topk_decoding(
                x0,
                selection_scores,
                schedule,
                init_maskable_mask,
                t,
                self.diff_args.diffusion_steps,
                self.tokenizer.mask_token_id,
            )
            return xt, maskable_mask

        raise ValueError(f"Unsupported denoise_method: {method}")

    @staticmethod
    def _compute_margin_scores(log_scores: torch.Tensor) -> torch.Tensor:
        top2_scores, _ = torch.topk(log_scores, k=2, dim=-1)
        return top2_scores[..., 0] - top2_scores[..., 1]

    @staticmethod
    def _origin_denoise_step(xt, x0, maskable_mask, t):
        unmask_prob = 1 / (t+1)
        mask_to_x0 = torch.rand(xt.shape, device=xt.device) < unmask_prob
        mask_to_x0 = torch.bitwise_and(mask_to_x0, maskable_mask)
        xt[mask_to_x0] = x0[mask_to_x0]
        maskable_mask = maskable_mask.clone()
        maskable_mask.masked_fill_(mask_to_x0, False)
        return xt, maskable_mask

    def _sample_tokens(self, probs: torch.Tensor) -> torch.Tensor:
        r"""Sample tokens under the configured temperature."""
        temperature = self.temperature
        if temperature <= 0:
            return probs.argmax(dim=-1)

        scaled = torch.clamp(probs, min=0)
        exponent = 1.0 / max(temperature, torch.finfo(probs.dtype).tiny)
        scaled = torch.pow(scaled, exponent)
        scaled_sum = scaled.sum(dim=-1, keepdim=True)
        zero_mask = scaled_sum <= 0
        scaled_sum = scaled_sum.clamp_min(1e-12)
        scaled = scaled / scaled_sum
        if zero_mask.any():
            uniform = torch.full_like(scaled, 1.0 / scaled.size(-1))
            scaled = torch.where(zero_mask, uniform, scaled)
        flat = scaled.view(-1, scaled.size(-1)).to(dtype=torch.float32)
        samples = torch.multinomial(flat, num_samples=1).squeeze(-1)
        return samples.view(*probs.shape[:-1])

    @staticmethod
    def _extract_schedule(decoding_strategy: str) -> str:
        if "-" in decoding_strategy:
            decoding_strategy = decoding_strategy.split("-")[-1]
        return decoding_strategy

def topk_masking(scores, cutoff_len):
    """Selects the lowest-scored positions deterministically."""
    sorted_scores = scores.sort(-1)[0]
    cutoff_len = cutoff_len.clamp(min=0, max=scores.size(-1) - 1)
    cutoff = sorted_scores.gather(dim=-1, index=cutoff_len)
    return scores < cutoff


def topk_decoding(
        x0,
        selection_scores,
        schedule,
        init_maskable_mask,
        t,
        max_step,
        noise
    ):
        if schedule == "linear":
            rate = t / max_step
        elif schedule == "cosine":
            rate = np.cos((max_step - t) / max_step * np.pi * 0.5)
        else:
            raise NotImplementedError

        cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
        _scores_for_topk = selection_scores.masked_fill(~init_maskable_mask, 1000.0)
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len)

        if isinstance(noise, torch.Tensor):
            xt = x0.masked_scatter(lowest_k_mask, noise[lowest_k_mask])
        elif isinstance(noise, (int, float)):
            xt = x0.masked_fill(lowest_k_mask, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")

        return xt