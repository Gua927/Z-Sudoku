from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

import torch
from torch import Tensor

# Ensure the repository's src directory is on sys.path when running the demo directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llmtuner.hparams import DiffusionArguments, FinetuningArguments, ModelArguments
from llmtuner.tuner.core.loader import load_model_and_tokenizer
from llmtuner.tuner.mdm.trainer import downk_decoding, topk_decoding

DIGIT_TOKENS = {str(i) for i in range(1, 10)}


def _default_diffusion_args() -> DiffusionArguments:
    """Return defaults matching the provided Sudoku checkpoint."""
    return DiffusionArguments(
        diffusion_steps=20,
        decoding_strategy="linear",
        denoise_method="downk-margin",
        z_mask=True,
        z_mask_method="downk-margin",
        downk_ratio=0.3,
        z_mask_temperature=1.0,
        reunmask=True,
        time_reweighting="linear",
        token_reweighting=False,
        alpha=0.25,
        gamma=1.0,
    )


@dataclass
class SamplerConfig:
    checkpoint_dir: Path
    base_model_dir: Path
    device: Optional[str] = None


class SudokuDiffusionSampler:
    """Wrapper around the diffusion evaluator for interactive use."""

    def __init__(self, config: SamplerConfig):
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.base_diff_args = _default_diffusion_args()
        self.model, self.tokenizer = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self.sequence_len = 164  # 81 digits + [SEP] + 81 digits + [EOS]
        self.solution_len = 81
        self.src_len = 81
        self.solution_slice = slice(self.src_len + 1, self.src_len + 1 + self.solution_len)

    def _load_model(self):
        model_args = ModelArguments(
            model_name_or_path=str(self.config.base_model_dir),
            checkpoint_dir=str(self.config.checkpoint_dir),
            cache_dir=None,
        )
        finetune_args = FinetuningArguments(stage="mdm", finetuning_type="full", lora_rank=0)
        model, tokenizer = load_model_and_tokenizer(
            model_args=model_args,
            finetuning_args=finetune_args,
            diffusion_args=self.base_diff_args,
            is_trainable=False,
        )
        return model, tokenizer

    def _prepare_inputs(self, puzzle: str) -> Dict[str, Tensor]:
        src_ids = self.tokenizer.encode(puzzle)
        if len(src_ids) != self.src_len:
            raise ValueError("Puzzle must contain exactly 81 characters.")
        tgt_placeholder = [self.tokenizer.mask_token_id] * self.solution_len
        input_ids = src_ids + [self.tokenizer.sep_token_id] + tgt_placeholder + [self.tokenizer.eos_token_id]
        attention_mask = [1] * len(input_ids)
        src_mask = [1] * (len(src_ids) + 1) + [0] * (len(input_ids) - len(src_ids) - 1)
        tensor_inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=self.device),
            "src_mask": torch.tensor([src_mask], dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long, device=self.device),
        }
        return tensor_inputs

    def solve(
        self,
        puzzle: str,
        diffusion_steps: int,
        temperature: float,
        denoise_method: str,
        decoding_strategy: str,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, object]:
        runtime_args = copy.deepcopy(self.base_diff_args)
        runtime_args.diffusion_steps = diffusion_steps
        runtime_args.denoise_method = denoise_method
        runtime_args.decoding_strategy = decoding_strategy
        mask_schedule = []

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        tensors = self._prepare_inputs(puzzle)
        start = time.perf_counter()
        with torch.no_grad():
            xt, probs, mask_schedule, timeline = self._denoise(
                tensors,
                runtime_args,
                temperature,
                verbose,
                mask_schedule,
            )
        elapsed = time.perf_counter() - start

        target_tokens = xt[0, self.solution_slice]
        solution = self._digits_from_tokens(target_tokens)

        confidence = None
        if probs is not None:
            target_probs = probs[0, self.solution_slice]
            gathered = target_probs.gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)
            confidence = float(gathered.mean().item())

        timeline_sorted = sorted(timeline, key=lambda item: item["step"])

        return {
            "solution": solution,
            "elapsed": elapsed,
            "mask_history": mask_schedule,
            "confidence": confidence,
            "timeline": timeline_sorted,
        }

    def _denoise(
        self,
        tensors: Dict[str, Tensor],
        diff_args: DiffusionArguments,
        temperature: float,
        verbose: bool,
        history: List[Tuple[int, int]],
    ) -> Tuple[Tensor, Optional[Tensor], List[Tuple[int, int]], List[Dict[str, object]]]:
        x = tensors["input_ids"]
        src_mask = tensors["src_mask"].bool()
        attention_mask = tensors["attention_mask"]
        maskable_mask = (~src_mask).bool()
        init_maskable_mask = maskable_mask.clone()
        denoise_method = getattr(diff_args, "denoise_method", "origin")
        decoding_schedule = self._extract_schedule(diff_args.decoding_strategy)
        modifiable_mask = torch.zeros_like(maskable_mask)
        do_reunmask = getattr(diff_args, "reunmask", False)
        xt = None
        final_probs = None
        timeline: List[Dict[str, object]] = []

        for t in range(diff_args.diffusion_steps - 1, -1, -1):
            if xt is None:
                xt = x.masked_fill(maskable_mask, self.tokenizer.mask_token_id)
            if verbose:
                print(f"t={t + 1}(in):", self.tokenizer.decode(xt.tolist()[0]))

            t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
            logits = self.model(xt, t_tensor, attention_mask=attention_mask)
            logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
            scores = torch.log_softmax(logits, dim=-1)
            scores[:, :, self.tokenizer.vocab_size :] = -1000
            probs = scores.exp()
            final_probs = probs
            x0_scores, _ = scores.max(-1)
            sampled_x0 = self._sample_tokens(probs, temperature)
            x0 = xt.masked_scatter(maskable_mask, sampled_x0[maskable_mask])

            if verbose:
                print(f"t={t + 1}(out):", self.tokenizer.decode(x0.tolist()[0]))

            current_mask = maskable_mask.clone()
            mask_remaining = int(current_mask.sum().item())
            history.append((t, mask_remaining))
            timeline.append(
                {
                    "step": int(t),
                    "remaining_masks": mask_remaining,
                    "candidate": self._format_solution_block(x0),
                }
            )

            if t > 0:
                selection_scores = None
                if denoise_method in {"topk", "downk"}:
                    selection_scores = x0_scores
                elif denoise_method in {"topk-margin", "downk-margin"}:
                    selection_scores = self._compute_margin_scores(scores)

                current_mask = maskable_mask.clone()
                xt, maskable_mask = self._apply_denoise_strategy(
                    xt=xt,
                    x0=x0,
                    selection_scores=selection_scores,
                    maskable_mask=maskable_mask,
                    init_maskable_mask=init_maskable_mask,
                    t=t,
                    diff_args=diff_args,
                )

                if do_reunmask:
                    correction_tokens = logits.argmax(dim=-1)
                    xt = xt.masked_scatter(modifiable_mask, correction_tokens[modifiable_mask])
                    modifiable_mask = current_mask & (~maskable_mask)
            else:
                xt = x0
                if do_reunmask:
                    correction_tokens = logits.argmax(dim=-1)
                    xt = xt.masked_scatter(modifiable_mask, correction_tokens[modifiable_mask])

        return xt, final_probs, history, timeline

    def _apply_denoise_strategy(
        self,
        xt: Tensor,
        x0: Tensor,
        selection_scores: Optional[Tensor],
        maskable_mask: Tensor,
        init_maskable_mask: Tensor,
        t: int,
        diff_args: DiffusionArguments,
    ) -> Tuple[Tensor, Tensor]:
        method = getattr(diff_args, "denoise_method", "origin")
        schedule = self._extract_schedule(diff_args.decoding_strategy)
        if method == "origin":
            return self._origin_denoise_step(xt, x0, maskable_mask, t)
        if method in {"topk", "topk-margin"}:
            if selection_scores is None:
                raise ValueError("Selection scores required for top-k decoding.")
            return topk_decoding(
                x0,
                selection_scores,
                schedule,
                init_maskable_mask,
                t,
                diff_args.diffusion_steps,
                self.tokenizer.mask_token_id,
            )
        if method in {"downk", "downk-margin"}:
            if selection_scores is None:
                raise ValueError("Selection scores required for down-k decoding.")
            return downk_decoding(
                x0,
                selection_scores,
                schedule,
                init_maskable_mask,
                maskable_mask,
                t,
                diff_args.diffusion_steps,
                self.tokenizer.mask_token_id,
                getattr(diff_args, "downk_ratio", 0.8),
            )
        raise ValueError(f"Unsupported denoise method: {method}")

    @staticmethod
    def _compute_margin_scores(log_scores: Tensor) -> Tensor:
        top2_scores, _ = torch.topk(log_scores, k=2, dim=-1)
        return top2_scores[..., 0] - top2_scores[..., 1]

    @staticmethod
    def _origin_denoise_step(xt: Tensor, x0: Tensor, maskable_mask: Tensor, t: int) -> Tuple[Tensor, Tensor]:
        unmask_prob = 1.0 / float(t + 1)
        mask_to_x0 = (torch.rand_like(xt, dtype=torch.float, device=xt.device) < unmask_prob) & maskable_mask
        xt = xt.clone()
        xt[mask_to_x0] = x0[mask_to_x0]
        next_mask = maskable_mask.clone()
        next_mask.masked_fill_(mask_to_x0, False)
        return xt, next_mask

    @staticmethod
    def _sample_tokens(probs: Tensor, temperature: float) -> Tensor:
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
        return samples.view(*probs.shape[:-1]).to(dtype=torch.long, device=probs.device)

    @staticmethod
    def _extract_schedule(decoding_strategy: str) -> str:
        if "-" in decoding_strategy:
            return decoding_strategy.split("-")[-1]
        return decoding_strategy

    def _digits_from_tokens(self, token_tensor: Tensor) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(token_tensor.tolist())
        digits: List[str] = []
        for tok in tokens:
            if tok in DIGIT_TOKENS:
                digits.append(tok)
            elif tok == "[MASK]":
                digits.append("0")
            elif tok.isdigit():
                digits.append(tok[-1])
            else:
                digits.append("0")
        return "".join(digits[: self.solution_len])

    def _format_solution_block(self, tensor: Tensor) -> str:
        token_ids = tensor[0, self.solution_slice]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
        flat: List[str] = []
        for tok in tokens:
            if tok in DIGIT_TOKENS:
                flat.append(tok)
            elif tok == "[MASK]":
                flat.append("□")
            elif tok.strip().isdigit():
                flat.append(tok.strip())
            else:
                flat.append("·")
        rows = [" ".join(flat[i * 9 : (i + 1) * 9]) for i in range(9)]
        return "\n".join(rows)


def build_sampler(checkpoint_dir: Path, base_model_dir: Optional[Path] = None, device: Optional[str] = None) -> SudokuDiffusionSampler:
    repo_root = Path(__file__).resolve().parents[1]
    base_model = base_model_dir or (repo_root / "model_config_tiny")
    config = SamplerConfig(
        checkpoint_dir=checkpoint_dir,
        base_model_dir=base_model,
        device=device,
    )
    return SudokuDiffusionSampler(config)
