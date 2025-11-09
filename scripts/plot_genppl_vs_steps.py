#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, List

import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import dataloader  # noqa: E402
from main import _load_from_checkpoint  # noqa: E402
from utils.eval_utils import compute_entropy, compute_generative_perplexity as compute_gen_ppl  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Plot GenPPL and entropy vs sampling steps.")
  parser.add_argument("--checkpoint", type=str, required=True, help="Path to the diffusion checkpoint.")
  parser.add_argument("--steps", type=int, nargs="+", default=[256, 512, 768, 1024], help="Sampling steps to evaluate.")
  parser.add_argument("--total-samples", type=int, default=512, help="Number of decoded samples per setting.")
  parser.add_argument("--batch-size", type=int, default=16, help="Eval batch size per forward pass.")
  parser.add_argument("--perplexity-batch", type=int, default=2, help="Batch size used when computing GenPPL.")
  parser.add_argument("--nucleus-p", type=float, default=1.0, help="Nucleus sampling probability.")
  parser.add_argument("--sampling-dtype", type=str, default="fp32", choices=["fp32", "fp64"], help="Sampling dtype for categorical draws.")
  parser.add_argument("--eval-model", type=str, default="gpt2-large", help="AR model for GenPPL (e.g., gpt2-large or meta-llama/Llama-2-7b-hf).")
  parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF cache directory override.")
  parser.add_argument("--output", type=str, default="plots/genppl_vs_steps.png", help="Output plot path.")
  parser.add_argument("--device", type=str, default="cuda", help="Torch device to run sampling on.")
  parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling.")
  return parser.parse_args()


def init_config(args: argparse.Namespace) -> Any:
  config_dir = Path(__file__).resolve().parent.parent / "configs"
  overrides = [
    "mode=ppl_eval",
    f"eval.checkpoint_path={args.checkpoint}",
    "eval.generate_samples=True",
    "eval.compute_generative_perplexity=True",
    "eval.compute_zero_shot=False",
    f"eval.perplexity_batch_size={args.perplexity_batch}",
    f"loader.global_batch_size={args.batch_size}",
    f"loader.eval_global_batch_size={args.batch_size}",
    f"loader.batch_size={args.batch_size}",
    f"loader.eval_batch_size={args.batch_size}",
    "trainer.devices=1",
    "trainer.num_nodes=1",
    f"sampling.sampling_dtype={args.sampling_dtype}",
    f"sampling.nucleus_p={args.nucleus_p}",
  ]
  if args.cache_dir:
    overrides.append(f"data.cache_dir={args.cache_dir}")
  with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="config", overrides=overrides)
  cfg.eval.gen_ppl_eval_model_name_or_path = args.eval_model
  cfg.sampling.total_samples = args.total_samples
  return cfg


def load_diffusion(cfg: Any) -> torch.nn.Module:
  tokenizer = dataloader.get_tokenizer(cfg)
  cfg.prime.base = math.ceil(cfg.prime.vocab_size ** (1 / cfg.prime.target_length))
  model = _load_from_checkpoint(cfg, tokenizer)
  model.config.eval.gen_ppl_eval_model_name_or_path = cfg.eval.gen_ppl_eval_model_name_or_path
  model.config.eval.perplexity_batch_size = cfg.eval.perplexity_batch_size
  model.config.loader.eval_batch_size = cfg.loader.eval_batch_size
  model.config.sampling.nucleus_p = cfg.sampling.nucleus_p
  model.config.sampling.sampling_dtype = cfg.sampling.sampling_dtype
  return model


def decode_batches(model, num_steps: int, total_samples: int) -> List[str]:
  texts: List[str] = []
  while len(texts) < total_samples:
    samples = model.restore_model_and_sample(num_steps=num_steps)
    decoded = model.tokenizer.batch_decode(samples.detach().cpu().tolist())
    texts.extend(decoded)
  return texts[:total_samples]


def evaluate_setting(model, texts: List[str]) -> tuple:
  model.gen_ppl_metric.reset()
  gen_ppl = compute_gen_ppl(model, texts)
  model.gen_ppl_metric.reset()
  entropy = compute_entropy(model, texts)
  return gen_ppl, entropy


def main() -> None:
  args = parse_args()
  device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
  cfg = init_config(args)
  if cfg.data.cache_dir:
    os.environ.setdefault("TRANSFORMERS_CACHE", cfg.data.cache_dir)
  model = load_diffusion(cfg)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
  model = model.to(device)
  model.eval()

  steps_list: List[int] = []
  ppl_list: List[float] = []
  entropy_list: List[float] = []

  for steps in args.steps:
    if device.type == "cuda":
      torch.cuda.empty_cache()
    texts = decode_batches(model, steps, args.total_samples)
    gen_ppl, entropy = evaluate_setting(model, texts)
    steps_list.append(steps)
    ppl_list.append(gen_ppl)
    entropy_list.append(entropy)
    print(f"steps={steps} gen_ppl={gen_ppl:.4f} entropy={entropy:.4f}")

  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  fig, ax1 = plt.subplots(figsize=(7, 4))
  ax1.plot(steps_list, ppl_list, marker="o", color="#f4a261", label="Gen PPL")
  ax1.set_xlabel("Sampling Steps")
  ax1.set_ylabel("Generative Perplexity", color="#f4a261")
  ax1.tick_params(axis="y", labelcolor="#f4a261")

  ax2 = ax1.twinx()
  ax2.plot(steps_list, entropy_list, marker="s", color="#2a9d8f", label="Entropy")
  ax2.set_ylabel("Entropy", color="#2a9d8f")
  ax2.tick_params(axis="y", labelcolor="#2a9d8f")

  lines = ax1.get_lines() + ax2.get_lines()
  labels = [line.get_label() for line in lines]
  ax1.legend(lines, labels, loc="upper right")
  fig.tight_layout()
  fig.savefig(output_path, dpi=300)
  plt.close(fig)


if __name__ == "__main__":
  main()
