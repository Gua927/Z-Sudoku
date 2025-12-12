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
    z_mask: bool = field(
        default=False,
        metadata={"help": "Enable z-mask forward strategy that adds two mask steps then denoises once."}
    )
    z_mask_method: str = field(
        default="origin",
        metadata={"help": "Denoising method used during z-mask forward: origin | topk | topk-margin."}
    )
    z_mask_noise_step: int = field(
        default=1,
        metadata={"help": "Number of steps to add noise in z-mask forward. -1 for random."}
    )
    z_mask_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Temperature for z-mask sampling. If None, use main temperature."}
    )
    pos_only: bool = field(
        default=False,
        metadata={"help": "If True, z-mask unmasking only selects positions, but fills with ground truth."}
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
    z_schedule: str = field(
        default="cos",
        metadata={"help": "Weight schedule for combining loss_o and loss_z: cos | all-z | sigmoid | piecewise | oscillate."}
    )
    z_sigmoid_center: float = field(
        default=0.5,
        metadata={"help": "Center b in [0,1] for sigmoid z_schedule; b * total_steps is the center step."}
    )
    piecewise_ratio: float = field(
        default=0.5,
        metadata={"help": "Ratio of training steps to use loss_o before transitioning to loss_z in piecewise schedule."}
    )

    downk_ratio: float = field(
        default=0.8,
        metadata={"help": "Ratio of hard mask (topk) in downk decoding. Remaining part is soft mask (origin)."}
    )
    loss_driven_ratio: float = field(
        default=1.0,
        metadata={"help": "Ratio of loss driven mask in loss_driven decoding. Remaining part is soft mask (origin)."}
    )
    reunmask: bool = field(
        default=False,
        metadata={"help": "If True, predict logits for unmasked positions in z-mask step and include them in loss."}
    )
    reunmask_correction_weight: float = field(
        default=1.0,
        metadata={"help": "Weight multiplier for incorrect tokens in reunmask loss."}
    )
    z_loss: bool = field(
        default=False,
        metadata={"help": "If True, calculate and log z_loss metrics (loss_former/latter) every z_loss_step_interval steps."}
    )
    z_loss_step_interval: int = field(
        default=500,
        metadata={"help": "Step interval for calculating z_loss metrics."}
    )

    def __post_init__(self):
        supported = {"origin", "topk", "topk-margin", "downk", "downk-margin", "loss_driven"}
        if self.denoise_method not in supported:
            raise ValueError(f"Unsupported denoise_method: {self.denoise_method}. Choose from {supported}.")

        if self.eval_denoise_method is not None and self.eval_denoise_method not in supported:
            raise ValueError(f"Unsupported eval_denoise_method: {self.eval_denoise_method}. Choose from {supported}.")
        
        z_mask_supported = supported | {"loss_driven"}
        if self.z_mask_method not in z_mask_supported:
            raise ValueError(f"Unsupported z_mask_method: {self.z_mask_method}. Choose from {z_mask_supported}.")

        if self.eval_denoise_methods:
            invalid = [m for m in self.eval_denoise_methods if m not in supported]
            if invalid:
                raise ValueError(f"Unsupported eval_denoise_methods: {invalid}. Choose from {supported}.")

        if self.topk_decoding is not None:
            if self.topk_decoding and self.denoise_method == "origin":
                self.denoise_method = "topk"
            elif (not self.topk_decoding) and self.denoise_method == "topk":
                self.denoise_method = "origin"

        # normalize z_schedule
        self.z_schedule = (self.z_schedule or "cos").lower()
        if self.z_schedule not in {"cos", "all-z", "sigmoid", "piecewise", "oscillate"}:
            raise ValueError(f"Unsupported z_schedule: {self.z_schedule}. Choose from {{'cos', 'all-z', 'sigmoid', 'piecewise', 'oscillate'}}.")

        # clamp z_sigmoid_center into [0,1]
        if self.z_sigmoid_center < 0.0:
            self.z_sigmoid_center = 0.0
        elif self.z_sigmoid_center > 1.0:
            self.z_sigmoid_center = 1.0

        # clamp piecewise_ratio into [0,1]
        if self.piecewise_ratio < 0.0:
            self.piecewise_ratio = 0.0
        elif self.piecewise_ratio > 1.0:
            self.piecewise_ratio = 1.0
