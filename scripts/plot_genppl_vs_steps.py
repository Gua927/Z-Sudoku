#!/usr/bin/env python3
import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
  parser.add_argument("--sample-method", "--sample-methods", dest="sample_methods", type=str, nargs="+", default=["origin"], help="Sampling strategies to use during decoding. Provide one or more values (e.g. origin top-k).")
  parser.add_argument("--sample-top-k", type=int, default=None, help="Optional hard cap on decoded tokens per step when using top-k.")
  parser.add_argument("--eval-model", type=str, default="gpt2-large", help="AR model for GenPPL (e.g., gpt2-large or meta-llama/Llama-2-7b-hf).")
  parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF cache directory override.")
  parser.add_argument("--output", type=str, default="plots/genppl_vs_steps.png", help="Output plot path.")
  parser.add_argument("--device", type=str, default="cuda", help="Torch device to run sampling on.")
  parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling.")
  return parser.parse_args()


def init_config(args: argparse.Namespace, sample_method: str) -> Any:
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
  cfg.sampling.sample_method = sample_method
  cfg.sampling.sample_top_k = args.sample_top_k if sample_method == "top-k" else None
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
  model.config.sampling.sample_method = cfg.sampling.sample_method
  model.config.sampling.sample_top_k = cfg.sampling.sample_top_k
  return model


def configure_sampling_method(model: torch.nn.Module, sample_method: str, sample_top_k: Optional[int]) -> None:
  method = sample_method.lower()
  model.sample_method = method
  model.config.sampling.sample_method = method
  top_k_value = sample_top_k if method == "top-k" else None
  if hasattr(model.config.sampling, "sample_top_k"):
    model.config.sampling.sample_top_k = top_k_value
  model.sample_top_k = top_k_value


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
  method_pairs = [(method, method.lower()) for method in args.sample_methods]
  if not method_pairs:
    raise ValueError("At least one sampling method must be specified.")

  device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
  cfg = init_config(args, sample_method=method_pairs[0][1])
  if cfg.data.cache_dir:
    os.environ.setdefault("TRANSFORMERS_CACHE", cfg.data.cache_dir)
  model = load_diffusion(cfg)
  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
  model = model.to(device)
  model.eval()

  results: List[Dict[str, Any]] = []
  label_counts: Dict[str, int] = {}
  cmap = plt.get_cmap("tab10")
  output_path = Path(args.output)
  output_path.parent.mkdir(parents=True, exist_ok=True)

  for idx, (method_label, method_key) in enumerate(method_pairs):
    configure_sampling_method(model, method_key, args.sample_top_k)
    method_seed = args.seed
    torch.manual_seed(method_seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(method_seed)

    steps_collector: List[int] = []
    ppl_collector: List[float] = []
    entropy_collector: List[float] = []

    for steps in args.steps:
      if device.type == "cuda":
        torch.cuda.empty_cache()
      texts = decode_batches(model, steps, args.total_samples)
      gen_ppl, entropy = evaluate_setting(model, texts)
      steps_collector.append(steps)
      ppl_collector.append(gen_ppl)
      entropy_collector.append(entropy)
      print(f"method={method_label} steps={steps} gen_ppl={gen_ppl:.4f} entropy={entropy:.4f}")

    label_key = method_label.lower()
    label_counts[label_key] = label_counts.get(label_key, 0) + 1
    display_label = method_label if label_counts[label_key] == 1 else f"{method_label} ({label_counts[label_key]})"

    results.append({
      "label": method_label,
      "display": display_label,
      "steps": steps_collector,
      "ppl": ppl_collector,
      "entropy": entropy_collector,
      "color": cmap(idx % cmap.N),
    })

  fig, ax1 = plt.subplots(figsize=(7, 4))
  ax1.set_xlabel("Sampling Steps")
  ax1.set_ylabel("Generative Perplexity")

  ax2 = ax1.twinx()
  ax2.set_ylabel("Entropy")

  metric_markers = {"ppl": "o", "entropy": "*"}
  handles: List[Any] = []
  labels: List[str] = []

  for data in results:
    color = data["color"]
    ppl_line = ax1.plot(
      data["steps"],
      data["ppl"],
      linestyle="-",
      marker=metric_markers["ppl"],
      color=color,
      markersize=6,
      label=f"{data['display']} - Gen PPL",
    )[0]
    entropy_line = ax2.plot(
      data["steps"],
      data["entropy"],
      marker=metric_markers["entropy"],
      linestyle="--",
      color=color,
      markersize=9,
      label=f"{data['display']} - Entropy",
    )[0]
    handles.extend([ppl_line, entropy_line])
    labels.extend([ppl_line.get_label(), entropy_line.get_label()])

  ax1.legend(handles, labels, loc="upper right")
  fig.tight_layout()
  fig.savefig(output_path, dpi=300)
  plt.close(fig)


if __name__ == "__main__":
  main()
