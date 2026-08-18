"""Typed experiment configuration with validation."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class DateRange:
    start: str
    end: str

    def validate(self, name: str) -> None:
        if len(self.start) != 7 or len(self.end) != 7:
            raise ValueError(f"{name} dates must use YYYY-MM")
        if self.start > self.end:
            raise ValueError(f"{name}.start must not be after {name}.end")


@dataclass
class ExperimentSection:
    name: str = "pfga"
    seed: int = 42
    device: str = "auto"
    deterministic: bool = True


@dataclass
class DataSection:
    directory: str = "data/monthly"
    strict_paper_schema: bool = True
    train: DateRange = field(default_factory=lambda: DateRange("2016-01", "2018-12"))
    validation: DateRange = field(default_factory=lambda: DateRange("2019-01", "2019-12"))
    test: DateRange = field(default_factory=lambda: DateRange("2020-01", "2020-12"))


@dataclass
class ModelSection:
    hidden_dim: int = 64
    attention_heads: int = 4
    dropout: float = 0.2
    antibody_hidden_dim: int = 32
    fairness_gamma: float = 0.9
    use_antibody_injection: bool = True
    use_fairness_memory: bool = True
    use_social_aggregation: bool = True


@dataclass
class CandidateSection:
    enabled: bool = True
    top_k: int = 32
    preference_weight: float = 1.0
    distance_weight: float = 1.0
    fairness_weight: float = 1.0
    slack_weight: float = 0.1


@dataclass
class LossSection:
    lambda_p: float = 0.7
    utility_weight: float = 1.0
    cost_weight: float = 1.0
    fairness_weight: float = 1.0
    normalize_utility: bool = True
    normalize_cost: bool = True
    reduction: str = "mean"


@dataclass
class AssignmentSection:
    threshold_base: float = 0.4
    threshold_scale: float = 0.1
    force_assign: bool = True


@dataclass
class TrainingSection:
    epochs: int = 500
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    gradient_clip_norm: float = 5.0
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    minimum_learning_rate: float = 1e-5
    output_directory: str = "outputs/pfga"


@dataclass
class EvaluationSection:
    cost_unit_km: float = 100.0
    dense_auc_pair_limit: int = 5_000_000


@dataclass
class PFGAConfig:
    experiment: ExperimentSection = field(default_factory=ExperimentSection)
    data: DataSection = field(default_factory=DataSection)
    model: ModelSection = field(default_factory=ModelSection)
    candidates: CandidateSection = field(default_factory=CandidateSection)
    loss: LossSection = field(default_factory=LossSection)
    assignment: AssignmentSection = field(default_factory=AssignmentSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)

    def validate(self) -> None:
        for name, value in (
            ("train", self.data.train),
            ("validation", self.data.validation),
            ("test", self.data.test),
        ):
            value.validate(name)
        if self.model.hidden_dim <= 0:
            raise ValueError("model.hidden_dim must be positive")
        if self.model.attention_heads <= 0:
            raise ValueError("model.attention_heads must be positive")
        if self.model.hidden_dim % self.model.attention_heads:
            raise ValueError("model.hidden_dim must be divisible by model.attention_heads")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if not 0.0 <= self.model.fairness_gamma < 1.0:
            raise ValueError("model.fairness_gamma must be in [0, 1)")
        if self.candidates.top_k <= 0:
            raise ValueError("candidates.top_k must be positive")
        if not 0.0 <= self.loss.lambda_p <= 1.0:
            raise ValueError("loss.lambda_p must be in [0, 1]")
        if self.loss.reduction not in {"mean", "sum"}:
            raise ValueError("loss.reduction must be 'mean' or 'sum'")
        if self.training.epochs <= 0:
            raise ValueError("training.epochs must be positive")
        if self.training.learning_rate <= 0.0:
            raise ValueError("training.learning_rate must be positive")
        if self.evaluation.cost_unit_km <= 0.0:
            raise ValueError("evaluation.cost_unit_km must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _section(raw: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Configuration section '{key}' must be a mapping")
    return value


def load_config(path: str) -> PFGAConfig:
    """Load a YAML configuration and resolve paths relative to the project cwd."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError("The YAML root must be a mapping")

    data_raw = _section(raw, "data")
    data = DataSection(
        directory=data_raw.get("directory", DataSection.directory),
        strict_paper_schema=data_raw.get("strict_paper_schema", True),
        train=DateRange(**data_raw.get("train", {"start": "2016-01", "end": "2018-12"})),
        validation=DateRange(
            **data_raw.get("validation", {"start": "2019-01", "end": "2019-12"})
        ),
        test=DateRange(**data_raw.get("test", {"start": "2020-01", "end": "2020-12"})),
    )
    config = PFGAConfig(
        experiment=ExperimentSection(**_section(raw, "experiment")),
        data=data,
        model=ModelSection(**_section(raw, "model")),
        candidates=CandidateSection(**_section(raw, "candidates")),
        loss=LossSection(**_section(raw, "loss")),
        assignment=AssignmentSection(**_section(raw, "assignment")),
        training=TrainingSection(**_section(raw, "training")),
        evaluation=EvaluationSection(**_section(raw, "evaluation")),
    )
    config.validate()
    return config
