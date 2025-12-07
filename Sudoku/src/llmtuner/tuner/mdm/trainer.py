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
from transformers.trainer_utils import EvalPrediction, PredictionOutput
from transformers.training_args import TrainingArguments
import math
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
        
        suffix = key[len(metric_prefix):].lstrip("_")
        # 仅记录loss和accuracy
        if "loss" not in suffix and "acc" not in suffix:
            continue

        if not isinstance(value, (int, float)) and not hasattr(value, "item"):
            continue
        
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
    def _apply_eval_overrides(self, temperature: Optional[float], method: Optional[str], reunmask: Optional[bool] = None):
        original_temperature = self.temperature
        original_method = getattr(self.diff_args, "denoise_method", None)
        original_reunmask = getattr(self.diff_args, "reunmask", False)
        try:
            if temperature is not None:
                self.temperature = float(temperature)
            if method is not None:
                self.diff_args.denoise_method = method
            if reunmask is not None:
                self.diff_args.reunmask = reunmask
            yield
        finally:
            self.temperature = original_temperature
            if original_method is not None:
                self.diff_args.denoise_method = original_method
            self.diff_args.reunmask = original_reunmask

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

    def _build_eval_combos(self, manual_override: bool) -> List[Tuple[Optional[str], Optional[float], Optional[bool]]]:
        if manual_override:
            return [(self._manual_eval_denoise_method, self._manual_eval_temperature, getattr(self.diff_args, "reunmask", False))]

        method_candidates = self._build_eval_values(
            self.eval_denoise_method,
            getattr(self.diff_args, "eval_denoise_methods", None)
        )
        temperature_candidates = self._build_eval_values(
            self.eval_temperature,
            self.eval_temperature_list,
        )
        
        reunmask_candidates = [False]
        if getattr(self.diff_args, "reunmask", False):
            reunmask_candidates = [True, False]

        return [
            (method, temperature, reunmask)
            for method in method_candidates
            for temperature in temperature_candidates
            for reunmask in reunmask_candidates
        ]

    def _format_metric_prefix(
        self,
        base_prefix: str,
        index: int,
        method: Optional[str],
        temperature: Optional[float],
        reunmask: Optional[bool],
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
            
        if reunmask is not None:
             parts.append(f"reunmask-{str(reunmask).lower()}")

        if len(parts) == 1:
            parts.append(f"variant{index+1}")

        return "_".join(parts)

    def _is_default_combo(self, method: Optional[str], temperature: Optional[float], reunmask: Optional[bool]) -> bool:
        default_method = getattr(self.diff_args, "denoise_method", None)
        default_temp = self.temperature
        default_reunmask = getattr(self.diff_args, "reunmask", False)
        
        method_match = (method is None) or (default_method is None) or (method == default_method)
        temp_match = (temperature is None) or (default_temp is None) or (float(temperature) == float(default_temp))
        reunmask_match = (reunmask is None) or (reunmask == default_reunmask)
        
        return method_match and temp_match and reunmask_match

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
        method: Optional[str],
        reunmask: Optional[bool]
    ) -> Dict[str, float]:
        with self._apply_eval_overrides(temperature, method, reunmask):
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

        for idx, (method, temperature, reunmask) in enumerate(eval_combos):
            metric_prefix = self._format_metric_prefix(metric_key_prefix, idx, method, temperature, reunmask, multi_eval)
            logger.info(
                "Running eval combo %d/%d: method=%s, temperature=%s, reunmask=%s, prefix=%s",
                idx + 1,
                len(eval_combos),
                method if method is not None else getattr(self.diff_args, "denoise_method", None),
                temperature if temperature is not None else self.temperature,
                reunmask if reunmask is not None else getattr(self.diff_args, "reunmask", False),
                metric_prefix,
            )
            metrics = self._evaluate_single(eval_dataset, ignore_keys, metric_prefix, temperature, method, reunmask)
            combo_metrics = {k: v for k, v in metrics.items() if k.startswith(metric_prefix)}
            _log_prefixed_metrics(self, metric_prefix, combo_metrics)
            combined_metrics.update(combo_metrics)

            alias_needed = (
                multi_eval
                and not manual_override
                and not alias_written
                and metric_key_prefix
                and self._is_default_combo(method, temperature, reunmask)
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
    
    def q_sample(self, model, inputs, t, maskable_mask, attention_mask):
        use_z_mask = bool(getattr(self.diff_args, "z_mask", False)) and model.training
        if use_z_mask:
            return self._z_mask_q_sample(model, inputs, t, maskable_mask, attention_mask)
        x_t, t, mask = self._standard_q_sample(inputs, t, maskable_mask)
        return x_t, t, mask, None

    def _standard_q_sample(self, inputs, t, maskable_mask):
        x_0 = inputs["input_ids"]
        uniform_noise = torch.rand_like(x_0, dtype=torch.float)
        ratios = (t + 1).float() / float(self.diff_args.diffusion_steps)
        mask = self._mask_from_uniform(uniform_noise, ratios, maskable_mask)
        x_t = x_0.masked_fill(mask, self.tokenizer.mask_token_id)
        return x_t, t, mask

    def _z_mask_q_sample(self, model, inputs, t, maskable_mask, attention_mask):
        x_0 = inputs["input_ids"]
        steps = int(self.diff_args.diffusion_steps)

        z_mask_noise_step = getattr(self.diff_args, "z_mask_noise_step", 1)
        if z_mask_noise_step == -1:
            low = t + 1
            high = torch.full_like(t, steps - 1)
            low = torch.min(low, high)
            dist = high - low + 1
            rand_offset = torch.rand(t.shape, device=t.device) * dist.float()
            t_plus = low + rand_offset.long()
        else:
            t_plus = t + z_mask_noise_step
            t_plus = torch.clamp(t_plus, max=steps - 1)

        uniform_noise = torch.rand_like(x_0, dtype=torch.float)
        ratios_t = (t + 1).float() / float(steps)
        mask_t = self._mask_from_uniform(uniform_noise, ratios_t, maskable_mask)

        ratios_t_plus = (t_plus + 1).float() / float(steps)
        mask_t_plus = self._mask_from_uniform(uniform_noise, ratios_t_plus, maskable_mask)

        x_plain = x_0.masked_fill(mask_t, self.tokenizer.mask_token_id)
        final_mask, x_t = self._z_mask_denoise(
            model=model,
            xt_plus=x_0.masked_fill(mask_t_plus, self.tokenizer.mask_token_id),
            mask=mask_t_plus,
            t_plus=t_plus,
            t_target=t + 1,
            maskable_mask=maskable_mask,
            attention_mask=attention_mask,
            x0_gt=x_0,
        )
        
        loss_mask = final_mask
        if getattr(self.diff_args, "reunmask", False):
            loss_mask = mask_t_plus

        extras = {
            "x_plain": x_plain,
            "mask_plain": mask_t,
            "mask_t_plus": mask_t_plus,
            "final_mask": final_mask,
        }
        return x_t, t, loss_mask, extras

    @staticmethod
    def _mask_from_uniform(uniform_noise: torch.Tensor, ratios: torch.Tensor, maskable_mask: torch.Tensor) -> torch.Tensor:
        ratios = ratios.clamp_(0.0, 1.0).unsqueeze(-1).to(uniform_noise.dtype)
        return (uniform_noise < ratios) & maskable_mask

    def _z_mask_denoise(self, model, xt_plus, mask, t_plus, t_target, maskable_mask, attention_mask, x0_gt=None):
        method = getattr(self.diff_args, "z_mask_method", getattr(self.diff_args, "denoise_method", "origin"))
        z_temp = getattr(self.diff_args, "z_mask_temperature", self.temperature)

        raw_logits = self._run_model_for_z_mask(model, xt_plus, t_plus, attention_mask)
        scores = torch.log_softmax(raw_logits, dim=-1)
        probs = scores.exp()

        with self._apply_eval_overrides(temperature=z_temp, method=method):
            sampled_x0 = self._sample_tokens(probs)
            x0 = xt_plus.masked_scatter(mask, sampled_x0[mask])

            selection_scores = None
            if method in ["topk", "downk"]:
                selection_scores, _ = scores.max(-1)
            elif method in ["topk-margin", "downk-margin"]:
                selection_scores = self._compute_margin_scores(scores)
            elif method == "loss_driven":
                if x0_gt is None:
                    raise ValueError("x0_gt is required for loss_driven method")
                
                bsz, seq_len, vocab_size = raw_logits.shape
                flat_logits = raw_logits.view(-1, vocab_size)
                flat_target = x0_gt.view(-1)
                
                token_loss = F.cross_entropy(flat_logits, flat_target, reduction='none')
                token_loss = token_loss.view(bsz, seq_len)
                
                # We want to unmask tokens with smallest loss.
                # The unmasking strategy (topk) selects tokens with highest scores.
                # So score = -loss.
                selection_scores = -token_loss

            # For loss_driven, we use topk strategy
            strategy_method = "loss_driven" if method == "loss_driven" else None

            pos_only = getattr(self.diff_args, "pos_only", False)
            if pos_only:
                if x0_gt is None:
                    raise ValueError("x0_gt is required for pos_only method")
                x0_for_strategy = x0_gt
            else:
                x0_for_strategy = x0

            decoding_schedule = self._extract_schedule(getattr(self.diff_args, "decoding_strategy", "linear"))
            xt, final_mask = self._apply_denoise_strategy(
                xt=xt_plus,
                x0=x0_for_strategy,
                selection_scores=selection_scores,
                maskable_mask=mask,
                init_maskable_mask=maskable_mask,
                t=t_target,
                schedule=decoding_schedule,
                method=strategy_method,
                is_batch=True,
            )
        return final_mask, xt

    def _run_model_for_z_mask(self, model, xt_plus, t_plus, attention_mask):
        was_training = model.training if hasattr(model, "training") else False
        if was_training:
            model.eval()
        with torch.no_grad():
            logits = model(xt_plus, t_plus, attention_mask=attention_mask)
            logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
        if was_training:
            model.train()
        return logits

    def _compute_weighted_loss(self, logits, target, loss_mask, t, vocab_size, sample_weights=None):
        if loss_mask is None:
            loss_mask = torch.ones_like(target, dtype=torch.bool)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_target = target.reshape(-1)
        token_loss = F.cross_entropy(flat_logits, flat_target, reduction="none").float()
        mask_flat = loss_mask.reshape(-1).bool()
        valid_count = mask_flat.float().sum()
        if valid_count.item() == 0:
            return token_loss.new_zeros(())
        token_loss = token_loss.masked_fill(~mask_flat, 0)

        if sample_weights is not None:
            flat_weights = sample_weights.reshape(-1)
            token_loss = token_loss * flat_weights

        if self.diff_args.token_reweighting:
            token_loss = self.diff_args.alpha * (1 - torch.exp(-token_loss)) ** self.diff_args.gamma * token_loss

        time_weight = self._build_time_weight(t, loss_mask).reshape(-1)
        weighted_loss = (token_loss * time_weight).sum()
        return weighted_loss / valid_count

    def _build_time_weight(self, t, template_mask):
        num_timesteps = self.diff_args.diffusion_steps
        if self.diff_args.time_reweighting == 'original':
            base = 1 / (t + 1).float()
        elif self.diff_args.time_reweighting == 'linear':
            base = (num_timesteps - t).float()
        else:
            base = torch.ones_like(t).float()
        return base[:, None].expand_as(template_mask).float()

    def _z_mask_weight_schedule(self):
        """Return lambda_z in [0,1] according to z_schedule.

        progress 定义为已训练步数比例 in [0,1]，默认使用余弦调度。
        额外支持：
        - all-z: 始终为 1，只用 loss_z
        - sigmoid: 使用 Sigmoid(a*(p-b))，a=5，b in [0,1]
        - piecewise: 前50%为0，后50%余弦增加到1
        - oscillate: 3步loss_o，1步loss_z
        """
        # 计算当前训练进度（步数比例）
        max_steps = max(1, getattr(self.state, "max_steps", 1) or 1)
        global_step = getattr(self.state, "global_step", 0)
        progress = global_step / max_steps
        progress = float(min(max(progress, 0.0), 1.0))

        # 读取调度方式，默认 cos
        z_schedule = getattr(self.diff_args, "z_schedule", "cos")
        z_schedule = str(z_schedule).lower()

        # all-z: 始终 1
        if z_schedule == "all-z":
            return 1.0

        # sigmoid: lambda(p) = sigmoid(a*(p-b)), a=5, b in [0,1]
        if z_schedule == "sigmoid":
            # a 固定为 5，b 可由 diff_args.z_sigmoid_center 控制，范围 [0,1]
            a = 5.0
            b = float(getattr(self.diff_args, "z_sigmoid_center", 0.5))
            b = min(max(b, 0.0), 1.0)
            x = a * (progress - b)
            # 数值稳定的 Sigmoid
            if x >= 0:
                exp_neg_x = math.exp(-x)
                weight = 1.0 / (1.0 + exp_neg_x)
            else:
                exp_x = math.exp(x)
                weight = exp_x / (1.0 + exp_x)
            return float(min(max(weight, 0.0), 1.0))

        # piecewise: 前 piecewise_ratio 为0，后 (1-piecewise_ratio) 线性增加到1
        if z_schedule == "piecewise":
            ratio = float(getattr(self.diff_args, "piecewise_ratio", 0.5))
            ratio = min(max(ratio, 0.0), 1.0)

            if progress < ratio:
                return 0.0
            else:
                # 映射 [ratio, 1.0] -> [0, 1]
                if ratio >= 1.0 - 1e-6: # Avoid division by zero
                    return 0.0
                weight = (progress - ratio) / (1.0 - ratio)
                return float(min(max(weight, 0.0), 1.0))

        # oscillate: 3步loss_o，1步loss_z
        if z_schedule == "oscillate":
            # 每4步为一个周期，前3步为0 (loss_o)，第4步为1 (loss_z)
            if global_step % 4 == 3:
                return 1.0
            else:
                return 0.0

        # 默认：余弦调度，从 0 平滑增加到 1
        weight = 0.5 * (1.0 - math.cos(math.pi * progress))
        return float(min(max(weight, 0.0), 1.0))

    def _log_dual_losses(self, loss_o, loss_z, loss_total, lambda_z, loss_re=None, loss_mask=None):
        log_payload = {
            "loss_o": loss_o.detach().item(),
            "loss_z": loss_z.detach().item(),
            "loss_t": loss_total.detach().item(),
            "lambda_z": lambda_z.detach().item() if torch.is_tensor(lambda_z) else float(lambda_z),
        }
        if loss_re is not None:
            log_payload["loss_re"] = loss_re.detach().item()
        if loss_mask is not None:
            log_payload["loss_mask"] = loss_mask.detach().item()
        self.log(log_payload)

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
        initial_attention_mask = torch.ones_like(x)
        x_t, t, loss_mask, extras = self.q_sample(
            model=model,
            inputs=inputs,
            t=t,
            maskable_mask=~src_mask,
            attention_mask=initial_attention_mask,
        )

        attention_mask = torch.ones_like(x_t)

        logits_z = model(x_t, t, attention_mask=attention_mask)
        logits_z = torch.cat([logits_z[:,0:1], logits_z[:,:-1]], dim=1)

        sample_weights = None
        if getattr(self.diff_args, "z_mask", False) and getattr(self.diff_args, "reunmask", False) and extras is not None and "mask_t_plus" in extras and "final_mask" in extras:
             mask_t_plus = extras["mask_t_plus"]
             final_mask = extras["final_mask"]
             mask_re = mask_t_plus & (~final_mask)
             
             correction_weight = getattr(self.diff_args, "reunmask_correction_weight", 1.0)
             if correction_weight != 1.0:
                 incorrect_mask = (x_t != x) & mask_re
                 
                 sample_has_error = incorrect_mask.any(dim=1)
                 batch_weights = torch.ones(x.size(0), device=x.device, dtype=torch.float)
                 batch_weights = batch_weights.masked_fill(sample_has_error, correction_weight)
                 sample_weights = batch_weights.unsqueeze(1).expand_as(x)

        loss_z = self._compute_weighted_loss(logits_z, x, loss_mask, t, vocab_size, sample_weights=sample_weights)

        loss_re = None
        loss_mask_val = None
        if getattr(self.diff_args, "reunmask", False) and extras is not None and "mask_t_plus" in extras and "final_mask" in extras:
             mask_t_plus = extras["mask_t_plus"]
             final_mask = extras["final_mask"]
             mask_re = mask_t_plus & (~final_mask)
             loss_re = self._compute_weighted_loss(logits_z, x, mask_re, t, vocab_size, sample_weights=sample_weights)
             loss_mask_val = self._compute_weighted_loss(logits_z, x, final_mask, t, vocab_size)

        loss = loss_z
        if extras is not None and "x_plain" in extras and "mask_plain" in extras:
            x_plain = extras["x_plain"]
            mask_plain = extras["mask_plain"]
            logits_o = model(x_plain, t, attention_mask=torch.ones_like(x_plain))
            logits_o = torch.cat([logits_o[:,0:1], logits_o[:,:-1]], dim=1)
            loss_o = self._compute_weighted_loss(logits_o, x, mask_plain, t, vocab_size)
            lambda_z = self._z_mask_weight_schedule()
            loss = (1 - lambda_z) * loss_o + lambda_z * loss_z
            self._log_dual_losses(loss_o, loss_z, loss, lambda_z, loss_re, loss_mask_val)

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
        
        # Track tokens that can be modified (re-unmasked)
        modifiable_mask = torch.zeros_like(maskable_mask)
        do_reunmask = getattr(self.diff_args, "reunmask", False)

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
                    if denoise_method in ["topk", "downk"]:
                        selection_scores = x0_scores
                    elif denoise_method in ["topk-margin", "downk-margin"]:
                        selection_scores = self._compute_margin_scores(scores)

                    current_mask = maskable_mask.clone()
                    xt, maskable_mask = self._apply_denoise_strategy(
                        xt=xt,
                        x0=x0,
                        selection_scores=selection_scores,
                        maskable_mask=maskable_mask,
                        init_maskable_mask=init_maskable_mask,
                        t=t,
                        schedule=decoding_schedule,
                        method=denoise_method,
                    )
                    
                    if do_reunmask:
                        # Correct modifiable tokens using argmax
                        correction_tokens = logits.argmax(dim=-1)
                        xt = xt.masked_scatter(modifiable_mask, correction_tokens[modifiable_mask])
                        # Update modifiable mask: tokens unmasked in this step
                        modifiable_mask = current_mask & (~maskable_mask)

                else:
                    xt = x0
                    if do_reunmask:
                        correction_tokens = logits.argmax(dim=-1)
                        xt = xt.masked_scatter(modifiable_mask, correction_tokens[modifiable_mask])
                        
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
        method: Optional[str] = None,
        is_batch: bool = False,
    ):
        denoise_method = method if method is not None else getattr(self.diff_args, "denoise_method", "origin")
        if denoise_method == "origin":
            if is_batch:
                return self._origin_denoise_step_batch(xt, x0, maskable_mask, t)
            return self._origin_denoise_step(xt, x0, maskable_mask, t)

        if denoise_method in {"topk", "topk-margin", "downk", "downk-margin", "loss_driven"}:
            if selection_scores is None:
                raise ValueError(f"Selection scores are required for {denoise_method} denoising.")
            
            if denoise_method in ["downk", "downk-margin"]:
                return downk_decoding(
                    x0,
                    selection_scores,
                    schedule,
                    init_maskable_mask,
                    maskable_mask,
                    t,
                    self.diff_args.diffusion_steps,
                    self.tokenizer.mask_token_id,
                    getattr(self.diff_args, "downk_ratio", 0.8)
                )

            if denoise_method == "loss_driven":
                return downk_decoding(
                    x0,
                    selection_scores,
                    schedule,
                    init_maskable_mask,
                    maskable_mask,
                    t,
                    self.diff_args.diffusion_steps,
                    self.tokenizer.mask_token_id,
                    getattr(self.diff_args, "loss_driven_ratio", 1.0)
                )

            new_xt, final_mask = topk_decoding(
                x0,
                selection_scores,
                schedule,
                init_maskable_mask,
                t,
                self.diff_args.diffusion_steps,
                self.tokenizer.mask_token_id,
            )
            return new_xt, final_mask

        raise ValueError(f"Unsupported denoise_method: {denoise_method}")

    @staticmethod
    def _compute_margin_scores(log_scores: torch.Tensor) -> torch.Tensor:
        top2_scores, _ = torch.topk(log_scores, k=2, dim=-1)
        return top2_scores[..., 0] - top2_scores[..., 1]

    @staticmethod
    def _origin_denoise_step(xt, x0, maskable_mask, t):
        unmask_prob = 1 / (t + 1)
        mask_to_x0 = torch.rand(xt.shape, device=xt.device) < unmask_prob
        mask_to_x0 = torch.bitwise_and(mask_to_x0, maskable_mask)
        xt[mask_to_x0] = x0[mask_to_x0]
        maskable_mask = maskable_mask.clone()
        maskable_mask.masked_fill_(mask_to_x0, False)
        return xt, maskable_mask

    @staticmethod
    def _origin_denoise_step_batch(xt, x0, maskable_mask, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=xt.device)
        t = t.to(device=xt.device)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.size(0) != xt.size(0):
            t = t.expand(xt.size(0))

        unmask_prob = (1 / (t + 1).float()).unsqueeze(-1)
        mask_to_x0 = torch.rand_like(xt, dtype=torch.float) < unmask_prob
        mask_to_x0 = torch.bitwise_and(mask_to_x0, maskable_mask)
        xt = xt.clone()
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
    # ensure cutoff_len has shape [B, 1] so that each sample has
    # a single scalar cutoff index, matching the semantic of
    # "select k lowest-scored positions per sample"
    if cutoff_len.dim() == 1:
        cutoff_len = cutoff_len.unsqueeze(-1)

    cutoff_len = cutoff_len.clamp(min=0, max=scores.size(-1) - 1)

    # gather the per-sample cutoff score at position cutoff_len
    cutoff = sorted_scores.gather(dim=-1, index=cutoff_len)

    # scores: [B, L], cutoff: [B, 1] -> broadcast to [B, L]
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
        # t can be a tensor of shape [B] (different time step per sample)
        # or a scalar. We want a per-sample rate in both cases.
        if isinstance(t, torch.Tensor):
            # make sure t is float tensor on the same device
            t_tensor = t.to(dtype=torch.float32, device=init_maskable_mask.device)
            if t_tensor.dim() == 0:
                # expand scalar to per-sample tensor
                t_tensor = t_tensor.expand(init_maskable_mask.size(0))
        else:
            # scalar -> expand to [B]
            t_tensor = torch.full(
                (init_maskable_mask.size(0),),
                float(t),
                dtype=torch.float32,
                device=init_maskable_mask.device,
            )

        if schedule == "linear":
            # rate: [B]
            rate = t_tensor / float(max_step)
        elif schedule == "cosine":
            rate = torch.cos((float(max_step) - t_tensor) / float(max_step) * math.pi * 0.5)
        else:
            raise NotImplementedError

        # init_maskable_mask.sum(1, keepdim=True): [B, 1]
        # rate.unsqueeze(1): [B, 1]
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True).float() * rate.unsqueeze(1)).long()
        _scores_for_topk = selection_scores.masked_fill(~init_maskable_mask, 1000.0)
        lowest_k_mask = topk_masking(_scores_for_topk, cutoff_len)

        if isinstance(noise, torch.Tensor):
            xt = x0.masked_scatter(lowest_k_mask, noise[lowest_k_mask])
        elif isinstance(noise, (int, float)):
            xt = x0.masked_fill(lowest_k_mask, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")

        return xt, lowest_k_mask

def select_k_lowest_with_mask(scores, k, mask):
    """Selects k positions with lowest scores from masked positions."""
    scores_filled = scores.masked_fill(~mask, float('inf'))
    ranks = scores_filled.argsort(dim=-1).argsort(dim=-1)
    if k.dim() == 1:
        k = k.unsqueeze(-1)
    return (ranks < k) & mask

def downk_decoding(
        x0,
        selection_scores,
        schedule,
        init_maskable_mask,
        maskable_mask,
        t,
        max_step,
        noise,
        downk_ratio
    ):
        # t can be a tensor of shape [B] (different time step per sample)
        # or a scalar. We want a per-sample rate in both cases.
        if isinstance(t, torch.Tensor):
            t_tensor = t.to(dtype=torch.float32, device=init_maskable_mask.device)
            if t_tensor.dim() == 0:
                t_tensor = t_tensor.expand(init_maskable_mask.size(0))
        else:
            t_tensor = torch.full(
                (init_maskable_mask.size(0),),
                float(t),
                dtype=torch.float32,
                device=init_maskable_mask.device,
            )

        if schedule == "linear":
            rate = t_tensor / float(max_step)
        elif schedule == "cosine":
            rate = torch.cos((float(max_step) - t_tensor) / float(max_step) * math.pi * 0.5)
        else:
            raise NotImplementedError

        # Calculate target mask count
        cutoff_len = (init_maskable_mask.sum(1, keepdim=True).float() * rate.unsqueeze(1)).long()
        
        # Ensure No Re-masking
        current_mask_count = maskable_mask.sum(1, keepdim=True)
        target_mask_count = torch.min(cutoff_len, current_mask_count)

        # Determine K_hard and K_soft
        k_hard = (target_mask_count.float() * downk_ratio).long()
        k_soft = target_mask_count - k_hard

        # Select Hard Mask
        hard_mask = select_k_lowest_with_mask(selection_scores, k_hard, maskable_mask)

        # Select Soft Mask
        soft_candidates = maskable_mask & (~hard_mask)
        rand_scores = torch.rand_like(selection_scores)
        soft_mask = select_k_lowest_with_mask(rand_scores, k_soft, soft_candidates)

        final_mask = hard_mask | soft_mask

        if isinstance(noise, torch.Tensor):
            xt = x0.masked_scatter(final_mask, noise[final_mask])
        elif isinstance(noise, (int, float)):
            xt = x0.masked_fill(final_mask, noise)
        else:
            raise NotImplementedError("noise should be either a tensor or a scalar")

        return xt, final_mask