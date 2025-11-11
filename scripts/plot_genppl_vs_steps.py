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
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import dataloader  # noqa: E402
from main import _load_from_checkpoint  # noqa: E402
from utils.eval_utils import compute_entropy, compute_generative_perplexity as compute_gen_ppl  # noqa: E402


def normalize_sample_method(name: str) -> str:
  return name.lower().replace('_', '-').replace(' ', '-')


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Plot GenPPL and entropy vs sampling steps.")
  parser.add_argument("--checkpoint", type=str, required=True, help="Path to the diffusion checkpoint.")
  parser.add_argument("--steps", type=int, nargs="+", default=[256, 512, 768, 1024], help="Sampling steps to evaluate.")
  parser.add_argument("--total-samples", type=int, default=512, help="Number of decoded samples per setting.")
  parser.add_argument("--batch-size", type=int, default=16, help="Eval batch size per forward pass.")
  parser.add_argument("--perplexity-batch", type=int, default=2, help="Batch size used when computing GenPPL.")
  parser.add_argument("--nucleus-p", type=float, default=1.0, help="Nucleus sampling probability.")
  parser.add_argument("--sampling-dtype", type=str, default="fp32", choices=["fp32", "fp64"], help="Sampling dtype for categorical draws.")
  parser.add_argument("--sample-method", "--sample-methods", dest="sample_methods", type=str, nargs="+", default=["origin"], help="Sampling strategies to use during decoding. Provide one or more values (e.g. origin top-k top-k-margin).")
  parser.add_argument("--sample-top-k", type=int, default=None, help="Optional hard cap on decoded tokens per step when using top-k or top-k-margin.")
  parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. 0.0 for greedy decoding, 1.0 for standard sampling, >1.0 for more randomness, <1.0 for less randomness.")
  parser.add_argument(
    "--eval-model",
    type=str,
    nargs="+",
    default=None,
    help="GenPPL AR model(s). Provide one or more names/paths in priority order, e.g. meta-llama/Llama-2-7b-hf gpt2-large.")
  parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF cache directory override.")
  parser.add_argument("--output", type=str, default="plots/genppl_vs_steps.png", help="Output plot path.")
  parser.add_argument("--device", type=str, default="cuda", help="Torch device to run sampling on.")
  parser.add_argument("--seed", type=int, default=1, help="Random seed for sampling.")
  parser.add_argument("--model-config", type=str, default="small", choices=["tiny", "small", "medium"], help="Diffusion model configuration size (tiny, small, medium).")
  return parser.parse_args()


def init_config(args: argparse.Namespace, sample_method: str) -> Any:
  sample_method = normalize_sample_method(sample_method)
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
    f"sampling.temperature={args.temperature}",
    f"model={args.model_config}",
  ]
  if args.cache_dir:
    overrides.append(f"data.cache_dir={args.cache_dir}")
  with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
    cfg = compose(config_name="config", overrides=overrides)
  if args.eval_model:
    cfg.eval.gen_ppl_eval_model_name_or_path = args.eval_model if len(args.eval_model) > 1 else args.eval_model[0]
  cfg.sampling.total_samples = args.total_samples
  cfg.sampling.sample_method = sample_method
  cfg.sampling.sample_top_k = args.sample_top_k if sample_method in {"top-k", "top-k-margin"} else None
  cfg.sampling.temperature = args.temperature
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
  model.config.sampling.temperature = cfg.sampling.temperature
  return model


def configure_sampling_method(model: torch.nn.Module, sample_method: str, sample_top_k: Optional[int], temperature: float = 1.0) -> None:
  method = normalize_sample_method(sample_method)
  model.sample_method = method
  model.config.sampling.sample_method = method
  top_k_value = sample_top_k if method in {"top-k", "top-k-margin"} else None
  if hasattr(model.config.sampling, "sample_top_k"):
    model.config.sampling.sample_top_k = top_k_value
  model.sample_top_k = top_k_value
  
  # 设置温度参数
  if hasattr(model.config.sampling, "temperature"):
    model.config.sampling.temperature = temperature
  # 如果模型有temperature属性，也设置它
  if hasattr(model, 'temperature'):
    model.temperature = temperature


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
  method_pairs = [(method, normalize_sample_method(method)) for method in args.sample_methods]
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
    configure_sampling_method(model, method_key, args.sample_top_k, args.temperature)
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
  ax2.set_ylim(4.0, 8.5)

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
