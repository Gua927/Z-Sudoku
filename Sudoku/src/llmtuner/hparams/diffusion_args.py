from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class DiffusionArguments:
    r"""
    Arguments of Diffusion Models.
    """
    diffusion_steps: int = field(
        default=64,
        metadata={"help": "timesteps of diffusion models."}
    )
    decoding_strategy: str = field(
        default="linear",
        metadata={"help": "Mask scheduling strategy, e.g., linear or cosine."}
    )
    token_reweighting: bool = field(
        default=False,
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    alpha: float = field(
        default=0.25,
        metadata={"help": "for focal loss"}
    )
    gamma: float = field(
        default=2,
        metadata={"help": "for focal loss"}
    )
    time_reweighting: str = field(
        default='original',
        metadata={"help": "use focal loss for token-level reweighting"}
    )
    denoise_method: str = field(
        default="origin",
        metadata={"help": "Denoising method: origin | topk | topk-margin."}
    )
    eval_denoise_method: Optional[str] = field(
        default=None,
        metadata={"help": "Optional override for evaluation denoising method."}
    )
    eval_denoise_methods: List[str] = field(
        default_factory=list,
        metadata={"help": "List of evaluation denoising methods to iterate over.", "nargs": "+"}
    )
    topk_decoding: Optional[bool] = field(
        default=None,
        metadata={"help": "[Deprecated] Backwards compatibility alias for denoise_method."}
    )

    def __post_init__(self):
        supported = {"origin", "topk", "topk-margin"}
        if self.denoise_method not in supported:
            raise ValueError(f"Unsupported denoise_method: {self.denoise_method}. Choose from {supported}.")

        if self.eval_denoise_method is not None and self.eval_denoise_method not in supported:
            raise ValueError(f"Unsupported eval_denoise_method: {self.eval_denoise_method}. Choose from {supported}.")

        if self.eval_denoise_methods:
            invalid = [m for m in self.eval_denoise_methods if m not in supported]
            if invalid:
                raise ValueError(f"Unsupported eval_denoise_methods: {invalid}. Choose from {supported}.")

        if self.topk_decoding is not None:
            if self.topk_decoding and self.denoise_method == "origin":
                self.denoise_method = "topk"
            elif (not self.topk_decoding) and self.denoise_method == "topk":
                self.denoise_method = "origin"
