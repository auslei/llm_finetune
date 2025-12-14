from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning, loadable from YAML."""

    training_data_path: str
    mode: str = "instruct"  # instruct (SFT) or pretrain (CPT)
    base_model: str = "unsloth/mistral-7b-bnb-4bit"
    model_name: Optional[str] = None  # Name for checkpoint/model folders
    checkpoints_dir: str = "checkpoints"
    output_dir: str = "models"
    resume_from_lora: Optional[str] = None  # Optional explicit LoRA path to resume from
    max_seq_length: int = 2048
    seed: int = 3407
    per_device_train_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 2
    learning_rate: Optional[float] = None
    num_train_epochs: int = 3
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    lora_rank: Optional[int] = None
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Optional[list[str]] = None
    use_gradient_checkpointing: Optional[Any] = None
    packing: Optional[bool] = None
    dataset_text_field: Optional[str] = None
    val_split: float = 0.1
    load_in_4bit: bool = True
    save_gguf: bool = False
    quantization_method: str = "q4_k_m"
    streaming: bool = False  # Enable streaming mode to reduce memory usage
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FineTuneConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "FineTuneConfig":
        """Allow legacy keys and partial dictionaries."""
        data = dict(raw)
        if "training_mode" in data and "mode" not in data:
            data["mode"] = data.pop("training_mode")
        if "base_model_path" in data and "base_model" not in data:
            data["base_model"] = data.pop("base_model_path")
        if "model_dir" in data and "output_dir" not in data:
            data["output_dir"] = data.pop("model_dir")
        # If only a model_name is provided and it looks like a model id, use it as base_model too.
        if "base_model" not in data and "model_name" in data and isinstance(data["model_name"], str):
            model_name = data["model_name"]
            if "/" in model_name:
                data.setdefault("base_model", model_name)
                data["model_name"] = model_name.replace("/", "_")
        return cls(**data)

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in {"instruct", "pretrain"}:
            raise ValueError("mode must be 'instruct' or 'pretrain'")
        if self.model_name is None:
            self.model_name = self.base_model.replace("/", "_")
        self.training_data_path = str(self.training_data_path)
        self.checkpoints_dir = str(self.checkpoints_dir)
        self.output_dir = str(self.output_dir)
        self._apply_mode_defaults()

    def _apply_mode_defaults(self) -> None:
        defaults = {
            "instruct": {
                "lora_rank": 64,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                "per_device_train_batch_size": 8,
                "learning_rate": 2e-4,
                "packing": False,
                "dataset_text_field": None,
                "use_gradient_checkpointing": "unsloth",
            },
            "pretrain": {
                "lora_rank": 128,
                "target_modules": [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                    "embed_tokens",
                    "lm_head",
                ],
                "per_device_train_batch_size": 4,
                "learning_rate": 5e-5,
                "packing": True,
                "dataset_text_field": "text",
                "use_gradient_checkpointing": True,
            },
        }[self.mode]

        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoints_dir) / self.model_name

    @property
    def model_output_path(self) -> Path:
        return Path(self.output_dir) / self.model_name


def load_config(config: FineTuneConfig | Mapping[str, Any] | str | None, overrides: Mapping[str, Any]) -> FineTuneConfig:
    """Resolve configuration from YAML, mapping, dataclass, or kwargs."""
    if config is None:
        if not overrides:
            raise ValueError("Either config or keyword overrides must be provided.")
        return FineTuneConfig.from_dict(overrides)
    if isinstance(config, FineTuneConfig):
        base_dict = dataclasses.asdict(config)
        base_dict.update(overrides)
        return FineTuneConfig.from_dict(base_dict)
    if isinstance(config, Mapping):
        base_dict = dict(config)
        base_dict.update(overrides)
        return FineTuneConfig.from_dict(base_dict)
    loaded = FineTuneConfig.from_yaml(config)
    merged = {**dataclasses.asdict(loaded), **overrides}
    return FineTuneConfig.from_dict(merged)
