
from __future__ import annotations

import abc
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence
import statistics


# ----------------------------
# Public configuration objects
# ----------------------------

@dataclass(frozen=True)
class PhysicalConstraints:
    """Fixed constraints for one hyperparameter-validation campaign."""
    max_total_length_m: float = 0.60
    max_phase_gradient_rad_per_m: float = math.pi / (50e-6)


@dataclass(frozen=True)
class SearchSpace:
    """Hyperparameter ranges searched during a campaign."""
    D: int
    total_length_min_m: float = 0.30
    total_length_max_m: float = 0.60
    sampling_spacing_m: float = 50e-6
    smoothness_weight_min: float = 1e-3
    smoothness_weight_max: float = 1.0
    learning_rate_min: float = 0.03
    learning_rate_max: float = 0.5
    weight_decay_min: float = 1e-3
    weight_decay_max: float = 1.0
    separation_min: float = 0.5
    separation_max: float = 3.0
    init_scale_min: float = 1e-3
    init_scale_max: float = 1.0

    @property
    def mask_counts(self) -> tuple[int, int, int]:
        if self.D < 2:
            raise ValueError("D must be >= 2.")
        return (self.D - 1, self.D, self.D + 1)


@dataclass(frozen=True)
class HalvingSchedule:
    initial_trials: int = 200
    epoch_budgets: tuple[int, int, int] = (20, 50, 100)
    keep_fractions: tuple[float, float] = (0.33, 0.33)

    promotion_gradient_multipliers: tuple[float, float] = (1.5, 1.2)

    # NEW seeds introduced at each stage.
    # Therefore cumulative seeds are:
    # stage 1: 101, 102
    # stage 2: 101, 102, 103
    # stage 3: 101, 102, 103, 104, 105
    new_training_seeds_by_stage: tuple[tuple[int, ...], ...] = (
        (101, 102),
        (103,),
        (104, 105),
    )

    def __post_init__(self) -> None:
        if sorted(self.epoch_budgets) != list(self.epoch_budgets):
            raise ValueError("epoch_budgets must be increasing.")

        if len(self.epoch_budgets) != 3:
            raise ValueError("This implementation expects exactly 3 stages.")

        if len(self.keep_fractions) != 2:
            raise ValueError("Need 2 keep fractions for 3 stages.")

        if len(self.promotion_gradient_multipliers) != 2:
            raise ValueError(
                "Need 2 promotion_gradient_multipliers for 3 stages."
            )

        if len(self.new_training_seeds_by_stage) != 3:
            raise ValueError(
                "Need one new-training-seed tuple for each stage."
            )

        all_seeds = [
            seed
            for stage_seeds in self.new_training_seeds_by_stage
            for seed in stage_seeds
        ]

        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError("Training seeds must not repeat across stages.")

    def cumulative_seeds(self, stage_index: int) -> tuple[int, ...]:
        """
        stage_index is zero-based:
          0 -> stage 1
          1 -> stage 2
          2 -> stage 3
        """
        return tuple(
            seed
            for seeds in self.new_training_seeds_by_stage[: stage_index + 1]
            for seed in seeds
        )


@dataclass(frozen=True)
class ConvergenceConfig:
    validation_interval_epochs: int = 5
    min_convergence_epoch: int = 50
    patience_epochs: int = 15
    min_gate_loss_improvement: float = 1e-5
    hard_max_epochs: int = 100


@dataclass(frozen=True)
class ValidationDataConfig:
    n_validation_states: int = 1000
    validation_dataset_seed: int = 12345
    n_test_states: int = 1000
    test_dataset_seed: int = 54321


@dataclass(frozen=True)
class FinalistConfig:
    n_finalists: int = 3
    finalist_training_seeds: tuple[int, ...] = (1001, 1002, 1003)
    final_training_seed: int = 2001
    require_all_seeds_feasible: bool = True
    fidelity_tie_tolerance: float = 1e-4


@dataclass(frozen=True)
class ResourceBudget:
    """Optional hard limits for the search campaign."""
    max_search_wall_time_seconds: Optional[float] = 4 * 3600
    max_trial_rss_bytes: Optional[int] = None


@dataclass(frozen=True)
class TrialParams:
    num_masks: int
    total_length_m: float
    propagation_distance_m: float
    sampling_spacing_m: float
    smoothness_weight: float
    learning_rate: float
    weight_decay: float
    separation: float
    init_scale: float


@dataclass
class EpochMetrics:
    epoch: int
    total_loss: float
    gate_loss: float
    smoothness_loss: float
    max_phase_gradient: float
    validation_fidelity: Optional[float] = None
    rss_bytes: Optional[int] = None


@dataclass
class TrialRecord:
    trial_id: str
    params: TrialParams
    training_seed: int
    stage: int = 0
    epochs_completed: int = 0
    status: str = "pending"
    latest_validation_fidelity: Optional[float] = None
    best_validation_fidelity: Optional[float] = None
    final_phase_gradient: Optional[float] = None
    final_gate_loss: Optional[float] = None
    final_total_loss: Optional[float] = None
    runtime_seconds: float = 0.0
    peak_rss_bytes: Optional[int] = None
    parameter_count: Optional[int] = None
    checkpoint_path: Optional[str] = None
    failure_reason: Optional[str] = None
    history: list[EpochMetrics] = field(default_factory=list)

    @property
    def total_length_m(self) -> float:
        return self.params.total_length_m

@dataclass
class ConfigRecord:
    config_id: str
    params: TrialParams

    # One TrialRecord per initialization seed.
    seed_runs: dict[int, TrialRecord] = field(default_factory=dict)

    stage: int = 0

# --------------------------------
# Adapter: connect your MPLC code
# --------------------------------

class MPLCAdapter(abc.ABC):
    """
    Small interface between this search engine and your existing MPLC code.

    `bundle` can be anything you want: e.g. a dict containing model, optimizer,
    scheduler, simulator, etc.
    """

    @abc.abstractmethod
    def build_training_bundle(
        self,
        params: TrialParams,
        training_seed: int,
    ) -> Any:
        """Construct a freshly initialized model + optimizer for one trial."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_one_epoch(
        self,
        bundle: Any,
        train_dataset: Any,
        params: TrialParams,
        epoch: int,
    ) -> tuple[float, float, float]:
        """
        Train one epoch.

        Returns:
            total_loss, gate_loss, smoothness_loss
        """
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_fidelity(self, bundle: Any, dataset: Any) -> float:
        """
        Return mean |<target | output>| over the supplied dataset.
        Must run without gradient tracking.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def max_phase_gradient(self, bundle: Any) -> float:
        """
        Return maximum WRAP-CORRECTED phase gradient in rad/m
        across all trained masks.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parameter_count(self, bundle: Any) -> int:
        """Return number of trainable mask parameters."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_checkpoint(self, bundle: Any, path: Path) -> None:
        """
        Save enough state to continue training exactly:
        model parameters + optimizer state (+ scheduler if used).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_checkpoint(
        self,
        params: TrialParams,
        training_seed: int,
        path: Path,
    ) -> Any:
        """Restore a checkpoint made by save_checkpoint()."""
        raise NotImplementedError

    def release(self, bundle: Any) -> None:
        """
        Optional cleanup hook.
        Override if you want explicit MPS/torch cache cleanup.
        """
        del bundle


# -----------------
# Utility functions
# -----------------

def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    if low <= 0 or high <= 0:
        raise ValueError("Log-uniform bounds must be positive.")
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def _sample_rss_bytes() -> Optional[int]:
    """
    Approximate process memory sampled at epoch boundaries.
    psutil is preferred because ru_maxrss is process-lifetime peak, not per-trial.
    """
    try:
        import psutil  # type: ignore
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return None


def _jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


class MPLCHyperparameterValidator:
    def __init__(
        self,
        *,
        adapter: MPLCAdapter,
        train_dataset: Any,
        random_field_dataset_factory: Callable[[TrialParams, int, int], Any],
        constraints: PhysicalConstraints,
        search_space: SearchSpace,
        schedule: HalvingSchedule = HalvingSchedule(),
        convergence: ConvergenceConfig = ConvergenceConfig(),
        data_config: ValidationDataConfig = ValidationDataConfig(),
        finalist_config: FinalistConfig = FinalistConfig(),
        resource_budget: ResourceBudget = ResourceBudget(),
        output_dir: str | Path = "mplc_validation_runs",
        search_seed: int = 2026,
    ) -> None:
        self.adapter = adapter
        self.train_dataset = train_dataset
        self.random_field_dataset_factory = random_field_dataset_factory
        self.constraints = constraints
        self.search_space = search_space
        self.schedule = schedule
        self.convergence = convergence
        self.data_config = data_config
        self.finalist_config = finalist_config
        self.resource_budget = resource_budget
        self.output_dir = Path(output_dir)
        self.search_seed = search_seed
        self.rng = random.Random(search_seed)
        self._campaign_start_time = time.perf_counter()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self._validate_configuration()

        

    def _validate_configuration(self) -> None:
        if self.search_space.total_length_min_m <= 0:
            raise ValueError("total_length_min_m must be > 0.")
        if self.search_space.total_length_min_m > self.search_space.total_length_max_m:
            raise ValueError("Invalid total-length search interval.")
        if self.constraints.max_total_length_m <= 0:
            raise ValueError("max_total_length_m must be > 0.")
        if self.constraints.max_phase_gradient_rad_per_m <= 0:
            raise ValueError("max_phase_gradient_rad_per_m must be > 0.")
        if self.search_space.sampling_spacing_m <= 0:
            raise ValueError("sampling_spacing_m must be positive.")
        if self.convergence.hard_max_epochs < self.schedule.epoch_budgets[-1]:
            raise ValueError(
                "hard_max_epochs must be >= the largest successive-halving budget."
            )

    def sample_trial_params(self) -> TrialParams:
        num_masks = self.rng.choice(self.search_space.mask_counts)

        length_upper = min(
            self.search_space.total_length_max_m,
            self.constraints.max_total_length_m,
        )
        if self.search_space.total_length_min_m > length_upper:
            raise ValueError(
                "No feasible total-length interval under max_total_length_m."
            )

        total_length_m = self.rng.uniform(
            self.search_space.total_length_min_m,
            length_upper,
        )

        spacing = self.search_space.sampling_spacing_m

        smoothness = _log_uniform(
            self.rng,
            self.search_space.smoothness_weight_min,
            self.search_space.smoothness_weight_max,
        )
        lr = _log_uniform(
            self.rng,
            self.search_space.learning_rate_min,
            self.search_space.learning_rate_max,
        )
        wd = _log_uniform(
            self.rng,
            self.search_space.weight_decay_min,
            self.search_space.weight_decay_max,
        )
        separation = self.rng.uniform(
            self.search_space.separation_min,
            self.search_space.separation_max,
        )

        init_scale = _log_uniform(
            self.rng,
            self.search_space.init_scale_min,
            self.search_space.init_scale_max,
        )

        return TrialParams(
            num_masks=num_masks,
            total_length_m=total_length_m,
            propagation_distance_m=total_length_m / num_masks,
            sampling_spacing_m=spacing,
            smoothness_weight=smoothness,
            learning_rate=lr,
            weight_decay=wd,
            separation=separation,
            init_scale=init_scale,
        )

    def _is_physically_feasible(self, record: TrialRecord) -> bool:
        if record.final_phase_gradient is None:
            return False
        return (
            record.params.total_length_m <= self.constraints.max_total_length_m
            and record.final_phase_gradient
            <= self.constraints.max_phase_gradient_rad_per_m
        )

    def _plateaued(self, history: Sequence[EpochMetrics]) -> bool:
        """
        Gate-loss plateau test:
        no improvement >= min_delta over the last `patience_epochs`.

        We only allow convergence-based stopping once:
          * min_convergence_epoch has been reached, AND
          * the phase-gradient constraint is currently satisfied.

        If the gradient is still too high, the trial continues to its budget
        so smoothness regularization has time to act.
        """
        if not history:
            return False

        latest = history[-1]
        if latest.epoch < self.convergence.min_convergence_epoch:
            return False
        if latest.max_phase_gradient > self.constraints.max_phase_gradient_rad_per_m:
            return False

        p = self.convergence.patience_epochs
        if len(history) < p + 1:
            return False

        if len(history) <= p:
            return False

        # Compare the best value before the patience window with the best
        # value inside the patience window. This asks whether there has been
        # a meaningful improvement at any time during the last p epochs.
        previous_best = min(x.gate_loss for x in history[:-p])
        recent_best = min(x.gate_loss for x in history[-p:])
        improvement = previous_best - recent_best

        return improvement < self.convergence.min_gate_loss_improvement

    def _checkpoint_path(self, trial_id: str) -> Path:
        return self.checkpoint_dir / f"{trial_id}.ckpt"

    def _write_trial_json(self, record: TrialRecord) -> None:
        path = self.output_dir / f"{record.trial_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(record), f, indent=2)

    def _append_summary_csv(self, record: TrialRecord, label: str = "") -> None:
        path = self.output_dir / "trial_summary.csv"
        exists = path.exists()

        row = {
            "label": label,
            "trial_id": record.trial_id,
            "status": record.status,
            "stage": record.stage,
            "training_seed": record.training_seed,
            "epochs_completed": record.epochs_completed,
            "num_masks": record.params.num_masks,
            "total_length_m": record.params.total_length_m,
            "propagation_distance_m": record.params.propagation_distance_m,
            "sampling_spacing_m": record.params.sampling_spacing_m,
            "smoothness_weight": record.params.smoothness_weight,
            "learning_rate": record.params.learning_rate,
            "weight_decay": record.params.weight_decay,
            "separation": record.params.separation,
            "init_scale": record.params.init_scale,
            "latest_validation_fidelity": record.latest_validation_fidelity,
            "best_validation_fidelity": record.best_validation_fidelity,
            "final_phase_gradient": record.final_phase_gradient,
            "phase_gradient_feasible": self._is_physically_feasible(record),
            "final_gate_loss": record.final_gate_loss,
            "final_total_loss": record.final_total_loss,
            "runtime_seconds": record.runtime_seconds,
            "peak_rss_bytes": record.peak_rss_bytes,
            "parameter_count": record.parameter_count,
            "failure_reason": record.failure_reason,
        }

        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _run_to_budget(
        self,
        record: TrialRecord,
        target_epoch: int,
        *,
        stage: int,
        allow_convergence_stop: bool,
    ) -> TrialRecord:
        if record.status == "failed":
            return record

        bundle = None
        checkpoint_path = self._checkpoint_path(record.trial_id)
        start_time = time.perf_counter()

        try:
            if record.epochs_completed == 0:
                bundle = self.adapter.build_training_bundle(
                    record.params,
                    record.training_seed,
                )
                record.parameter_count = self.adapter.parameter_count(bundle)
            else:
                bundle = self.adapter.load_checkpoint(
                    record.params,
                    record.training_seed,
                    checkpoint_path,
                )
            
            validation_dataset = self.random_field_dataset_factory(record.params, self.data_config.n_validation_states, self.data_config.validation_dataset_seed)

            record.stage = stage
            record.status = "running"

            for epoch in range(record.epochs_completed + 1, target_epoch + 1):
                total_loss, gate_loss, smoothness_loss = self.adapter.train_one_epoch(
                    bundle,
                    self.train_dataset,
                    record.params,
                    epoch,
                )

                numeric_values = (total_loss, gate_loss, smoothness_loss)
                if not all(math.isfinite(float(x)) for x in numeric_values):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch {epoch}: {numeric_values}"
                    )

                phase_gradient = float(self.adapter.max_phase_gradient(bundle))
                if not math.isfinite(phase_gradient):
                    raise FloatingPointError(
                        f"Non-finite phase gradient at epoch {epoch}."
                    )

                val_fidelity = None
                must_validate = (
                    epoch % self.convergence.validation_interval_epochs == 0
                    or epoch == target_epoch
                    or epoch in self.schedule.epoch_budgets
                )
                if must_validate:
                    val_fidelity = float(
                        self.adapter.evaluate_fidelity(
                            bundle,
                            validation_dataset,
                        )
                    )
                    if not math.isfinite(val_fidelity):
                        raise FloatingPointError(
                            f"Non-finite validation fidelity at epoch {epoch}."
                        )
                    record.latest_validation_fidelity = val_fidelity
                    record.best_validation_fidelity = (
                        val_fidelity
                        if record.best_validation_fidelity is None
                        else max(record.best_validation_fidelity, val_fidelity)
                    )

                rss = _sample_rss_bytes()
                if rss is not None:
                    record.peak_rss_bytes = (
                        rss if record.peak_rss_bytes is None
                        else max(record.peak_rss_bytes, rss)
                    )
                    if (
                        self.resource_budget.max_trial_rss_bytes is not None
                        and rss > self.resource_budget.max_trial_rss_bytes
                    ):
                        raise MemoryError(
                            f"Trial RSS {rss} exceeded configured limit "
                            f"{self.resource_budget.max_trial_rss_bytes}."
                        )

                metric = EpochMetrics(
                    epoch=epoch,
                    total_loss=float(total_loss),
                    gate_loss=float(gate_loss),
                    smoothness_loss=float(smoothness_loss),
                    max_phase_gradient=phase_gradient,
                    validation_fidelity=val_fidelity,
                    rss_bytes=rss,
                )
                record.history.append(metric)
                record.epochs_completed = epoch
                record.final_phase_gradient = phase_gradient
                record.final_gate_loss = float(gate_loss)
                record.final_total_loss = float(total_loss)

                # Check convergence only on validation checkpoints.
                if (
                    allow_convergence_stop
                    and must_validate
                    and self._plateaued(record.history)
                ):
                    break

            # Mandatory validation at the actual final state.
            if (
                not record.history
                or record.history[-1].validation_fidelity is None
            ):
                val_fidelity = float(
                    self.adapter.evaluate_fidelity(bundle, validation_dataset)
                )
                record.latest_validation_fidelity = val_fidelity
                record.best_validation_fidelity = (
                    val_fidelity
                    if record.best_validation_fidelity is None
                    else max(record.best_validation_fidelity, val_fidelity)
                )

            self.adapter.save_checkpoint(bundle, checkpoint_path)
            record.checkpoint_path = str(checkpoint_path)
            record.status = "completed"

        except Exception as exc:
            record.status = "failed"
            record.failure_reason = f"{type(exc).__name__}: {exc}"

        finally:
            record.runtime_seconds += time.perf_counter() - start_time
            if bundle is not None:
                try:
                    self.adapter.release(bundle)
                except Exception:
                    pass

            self._write_trial_json(record)

        return record

    @staticmethod
    def _ranking_key(record: TrialRecord) -> float:
        if record.status != "completed":
            return float("-inf")
        if record.latest_validation_fidelity is None:
            return float("-inf")
        return record.latest_validation_fidelity

    # def _promote(
    #     self,
    #     records: Sequence[TrialRecord],
    #     keep_fraction: float,
    #     gradient_multiplier: float,
    # ) -> list[TrialRecord]:

    #     eligible = [
    #         r for r in records
    #         if r.status == "completed"
    #         and r.latest_validation_fidelity is not None
    #         and r.final_phase_gradient is not None
    #     ]

    #     if not eligible:
    #         return []

    #     keep_n = max(
    #         1,
    #         math.ceil(len(eligible) * keep_fraction)
    #     )

    #     gradient_limit = (
    #         gradient_multiplier
    #         * self.constraints.max_phase_gradient_rad_per_m
    #     )

    #     # First preference:
    #     # trials already within the relaxed stage constraint.
    #     within_limit = [
    #         r for r in eligible
    #         if r.final_phase_gradient <= gradient_limit
    #     ]

    #     within_limit.sort(
    #         key=self._ranking_key,
    #         reverse=True,
    #     )

    #     promoted = within_limit[:keep_n]

    #     # If too few trials meet the relaxed constraint,
    #     # do NOT kill the search. Fill remaining slots with
    #     # the trials having the smallest gradient violations.
    #     if len(promoted) < keep_n:

    #         promoted_ids = {
    #             r.trial_id for r in promoted
    #         }

    #         outside_limit = [
    #             r for r in eligible
    #             if r.trial_id not in promoted_ids
    #         ]

    #         outside_limit.sort(
    #             key=lambda r: (
    #                 r.final_phase_gradient / gradient_limit,
    #                 -r.latest_validation_fidelity,
    #             )
    #         )

    #         promoted.extend(
    #             outside_limit[
    #                 : keep_n - len(promoted)
    #             ]
    #         )

    #     # Delete continuation checkpoints for trials
    #     # that are no longer needed.
    #     promoted_ids = {
    #         r.trial_id for r in promoted
    #     }

    #     for r in records:
    #         if (
    #             r.trial_id not in promoted_ids
    #             and r.checkpoint_path
    #         ):
    #             try:
    #                 Path(r.checkpoint_path).unlink(
    #                     missing_ok=True
    #                 )
    #                 r.checkpoint_path = None
    #             except Exception:
    #                 pass

    #     return promoted

    def _wall_time_exhausted(self) -> bool:
        limit = self.resource_budget.max_search_wall_time_seconds
        if limit is None:
            return False
        return (time.perf_counter() - self._campaign_start_time) >= limit

    def run_baseline(
        self,
        baseline_params: TrialParams,
        *,
        training_seed: int = 777,
    ) -> TrialRecord:
        """Train/evaluate the agreed reference configuration once."""
        record = TrialRecord(
            trial_id="baseline",
            params=baseline_params,
            training_seed=training_seed,
        )
        record = self._run_to_budget(
            record,
            self.convergence.hard_max_epochs,
            stage=0,
            allow_convergence_stop=True,
        )
        self._append_summary_csv(record, label="baseline")
        return record

    # def run_successive_halving(self) -> list[TrialRecord]:
    #     """
    #     Run 50 -> ~12 -> ~4 (defaults) using 20 -> 50 -> 100 total epochs.
    #     Early stages rank by validation fidelity only.
    #     Physical phase-gradient feasibility becomes a hard criterion after stage 3.
    #     """
    #     # Stage 1
    #     stage1: list[TrialRecord] = []
    #     for i in range(self.schedule.initial_trials):
    #         if self._wall_time_exhausted():
    #             break
    #         params = self.sample_trial_params()
    #         record = TrialRecord(
    #             trial_id=f"trial_{i:04d}",
    #             params=params,
    #             training_seed=self.search_seed + 10_000 + i,
    #         )
    #         stage1.append(
    #             self._run_to_budget(
    #                 record,
    #                 self.schedule.epoch_budgets[0],
    #                 stage=1,
    #                 allow_convergence_stop=False,
    #             )
    #         )

    #     stage2 = self._promote(stage1, self.schedule.keep_fractions[0], self.schedule.promotion_gradient_multipliers[0])

    #     # Stage 2: continue promoted checkpoints to 50 total epochs
    #     for record in stage2:
    #         if self._wall_time_exhausted():
    #             break
    #         self._run_to_budget(
    #             record,
    #             self.schedule.epoch_budgets[1],
    #             stage=2,
    #             allow_convergence_stop=False,
    #         )

    #     stage2_completed = [
    #         r for r in stage2
    #         if r.stage == 2
    #         and r.epochs_completed >= self.schedule.epoch_budgets[1]
    #         and r.status == "completed"
    #     ]
    #     stage3 = self._promote(stage2_completed, self.schedule.keep_fractions[1], self.schedule.promotion_gradient_multipliers[1])

    #     # Stage 3: continue promoted checkpoints to 100 total epochs
    #     for record in stage3:
    #         if self._wall_time_exhausted():
    #             break
    #         self._run_to_budget(
    #             record,
    #             min(
    #                 self.schedule.epoch_budgets[2],
    #                 self.convergence.hard_max_epochs,
    #             ),
    #             stage=3,
    #             allow_convergence_stop=True,
    #         )
    #         self._append_summary_csv(record, label="stage3")

    #     return [
    #         r for r in stage3
    #         if r.stage == 3 and r.status == "completed"
    #     ]

    def _select_finalist_params(
        self,
        stage3: Sequence[TrialRecord],
    ) -> list[TrialParams]:
        feasible = [
            r for r in stage3
            if r.stage == 3
            and r.status == "completed"
            and self._is_physically_feasible(r)
            and r.latest_validation_fidelity is not None
        ]
        feasible.sort(
            key=lambda r: r.latest_validation_fidelity or float("-inf"),
            reverse=True,
        )
        return [
            r.params
            for r in feasible[: self.finalist_config.n_finalists]
        ]

    def _select_finalist_configs(
        self,
        stage3: Sequence[ConfigRecord],
    ) -> list[ConfigRecord]:

        eligible: list[ConfigRecord] = []

        for config in stage3:

            runs = self._config_completed_runs(config)

            if not runs:
                continue

            feasible = [
                self._is_physically_feasible(r)
                for r in runs
            ]

            if self.finalist_config.require_all_seeds_feasible:
                if not all(feasible):
                    continue
            else:
                if not any(feasible):
                    continue

            eligible.append(config)

        eligible.sort(
            key=self._config_score,
            reverse=True,
        )

        return eligible[
            : self.finalist_config.n_finalists
        ]

    def _train_finalist_seed(
        self,
        params: TrialParams,
        seed: int,
        finalist_index: int,
    ) -> TrialRecord:
        record = TrialRecord(
            trial_id=f"finalist_{finalist_index:02d}_seed_{seed}",
            params=params,
            training_seed=seed,
        )
        record = self._run_to_budget(
            record,
            self.convergence.hard_max_epochs,
            stage=4,
            allow_convergence_stop=True,
        )
        self._append_summary_csv(record, label="finalist_seed")
        return record

    # def evaluate_finalists(
    #     self,
    #     stage3: Sequence[TrialRecord],
    # ) -> list[dict[str, Any]]:
    #     finalist_params = self._select_finalist_params(stage3)
    #     results: list[dict[str, Any]] = []

    #     for i, params in enumerate(finalist_params):
    #         runs: list[TrialRecord] = []
    #         for seed in self.finalist_config.finalist_training_seeds:
    #             if self._wall_time_exhausted():
    #                 break
    #             runs.append(self._train_finalist_seed(params, seed, i))

    #         feasible_runs = [
    #             r for r in runs
    #             if r.status == "completed"
    #             and self._is_physically_feasible(r)
    #             and r.latest_validation_fidelity is not None
    #         ]

    #         # Per our SOP, infeasible seeds are NOT averaged into feasibility.
    #         fidelities = [
    #             float(r.latest_validation_fidelity)
    #             for r in feasible_runs
    #             if r.latest_validation_fidelity is not None
    #         ]

    #         mean_fidelity = (
    #             sum(fidelities) / len(fidelities) if fidelities else None
    #         )
    #         std_fidelity = None
    #         if len(fidelities) >= 2:
    #             m = mean_fidelity
    #             assert m is not None
    #             std_fidelity = math.sqrt(
    #                 sum((x - m) ** 2 for x in fidelities)
    #                 / (len(fidelities) - 1)
    #             )

    #         results.append(
    #             {
    #                 "params": params,
    #                 "runs": runs,
    #                 "n_feasible_seeds": len(feasible_runs),
    #                 "all_seeds_feasible": (
    #                     len(runs) == len(self.finalist_config.finalist_training_seeds)
    #                     and len(feasible_runs) == len(runs)
    #                 ),
    #                 "mean_validation_fidelity_feasible_seeds": mean_fidelity,
    #                 "std_validation_fidelity_feasible_seeds": std_fidelity,
    #                 "mean_phase_gradient_feasible_seeds": (
    #                     sum(float(r.final_phase_gradient) for r in feasible_runs)
    #                     / len(feasible_runs)
    #                     if feasible_runs else None
    #                 ),
    #             }
    #         )

    #     return results

    def evaluate_finalists(
        self,
        stage3: Sequence[ConfigRecord],
    ) -> list[dict[str, Any]]:

        finalists = self._select_finalist_configs(stage3)

        results = []

        for config in finalists:

            runs = self._config_completed_runs(config)

            feasible_runs = [
                r
                for r in runs
                if self._is_physically_feasible(r)
            ]

            fidelities = [
                float(r.latest_validation_fidelity)
                for r in feasible_runs
                if r.latest_validation_fidelity is not None
            ]

            results.append(
                {
                    "config_id": config.config_id,
                    "params": config.params,
                    "runs": runs,

                    "n_seeds": len(runs),
                    "n_feasible_seeds": len(feasible_runs),

                    "all_seeds_feasible": (
                        len(feasible_runs) == len(runs)
                    ),

                    "median_validation_fidelity_feasible_seeds": (
                        statistics.median(fidelities)
                        if fidelities
                        else None
                    ),

                    "mean_validation_fidelity_feasible_seeds": (
                        statistics.mean(fidelities)
                        if fidelities
                        else None
                    ),

                    "std_validation_fidelity_feasible_seeds": (
                        statistics.stdev(fidelities)
                        if len(fidelities) >= 2
                        else None
                    ),

                    "best_validation_fidelity_feasible_seeds": (
                        max(fidelities)
                        if fidelities
                        else None
                    ),

                    "worst_validation_fidelity_feasible_seeds": (
                        min(fidelities)
                        if fidelities
                        else None
                    ),
                    "mean_phase_gradient_feasible_seeds": (
                        statistics.mean(
                            float(r.final_phase_gradient)
                            for r in feasible_runs
                            if r.final_phase_gradient is not None
                        )
                        if feasible_runs
                        else None
                    ),
                    
                }
            )

        return results

    @staticmethod
    def _float_or_inf(x: Optional[float]) -> float:
        return float("inf") if x is None else float(x)

    def select_winner(
        self,
        finalist_results: Sequence[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """
        Select by mean validation fidelity, then apply manufacturability/simple-
        design tie-breakers only inside the configured fidelity tolerance.
        """
        eligible = []
        for x in finalist_results:
            if x["median_validation_fidelity_feasible_seeds"] is None:
                continue
            if (
                self.finalist_config.require_all_seeds_feasible
                and not x["all_seeds_feasible"]
            ):
                continue
            eligible.append(x)

        if not eligible:
            return None

        best_fidelity = max(
            float(x["median_validation_fidelity_feasible_seeds"])
            for x in eligible
        )
        tol = self.finalist_config.fidelity_tie_tolerance
        tied = [
            x for x in eligible
            if best_fidelity
            - float(x["median_validation_fidelity_feasible_seeds"]) <= tol
        ]

        def tie_key(x: dict[str, Any]) -> tuple[float, float, float]:
            feasible_runs = [
                r for r in x["runs"]
                if r.status == "completed"
                and self._is_physically_feasible(r)
                and r.latest_validation_fidelity is not None
            ]
            min_params = min(
                (
                    r.parameter_count
                    for r in feasible_runs
                    if r.parameter_count is not None
                ),
                default=10**18,
            )
            return (
                self._float_or_inf(x["mean_phase_gradient_feasible_seeds"]),
                float(x["params"].total_length_m),
                float(min_params),
            )

        return min(tied, key=tie_key)

    def final_train_and_test(
        self,
        winning_params: TrialParams,
    ) -> TrialRecord:
        """
        Fresh training run with a new seed, followed by evaluation on a fresh
        1000-state random test set that was never used for hyperparameter selection.
        """
        record = TrialRecord(
            trial_id="final_model",
            params=winning_params,
            training_seed=self.finalist_config.final_training_seed,
        )
        record = self._run_to_budget(
            record,
            self.convergence.hard_max_epochs,
            stage=5,
            allow_convergence_stop=True,
        )

        if record.status != "completed":
            self._append_summary_csv(record, label="final_model_failed")
            return record

        checkpoint = Path(record.checkpoint_path) if record.checkpoint_path else None
        if checkpoint is None:
            record.status = "failed"
            record.failure_reason = "Final checkpoint missing."
            self._append_summary_csv(record, label="final_model_failed")
            return record

        bundle = None
        try:
            bundle = self.adapter.load_checkpoint(
                record.params,
                record.training_seed,
                checkpoint,
            )
            test_dataset = self.random_field_dataset_factory(
                record.params,
                self.data_config.n_test_states,
                self.data_config.test_dataset_seed,
            )
            test_fidelity = float(
                self.adapter.evaluate_fidelity(bundle, test_dataset)
            )

            # Attach test fidelity to a sidecar JSON rather than overloading
            # validation_fidelity semantics.
            report = {
                "winner_params": _jsonable(winning_params),
                "final_training_seed": record.training_seed,
                "test_dataset_seed": self.data_config.test_dataset_seed,
                "n_test_states": self.data_config.n_test_states,
                "test_fidelity": test_fidelity,
                "max_phase_gradient": record.final_phase_gradient,
                "phase_gradient_feasible": self._is_physically_feasible(record),
                "total_length_m": winning_params.total_length_m,
                "propagation_distance_m": winning_params.propagation_distance_m,
                "parameter_count": record.parameter_count,
                "runtime_seconds": record.runtime_seconds,
            }
            with (self.output_dir / "final_report.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(report, f, indent=2)

            self._append_summary_csv(record, label="final_model")
            return record

        finally:
            if bundle is not None:
                try:
                    self.adapter.release(bundle)
                except Exception:
                    pass

    def run_all(
        self,
        baseline_params: Optional[TrialParams] = None,
    ) -> dict[str, Any]:
        """
        Complete SOP:
          1. successive halving
          2. 3-seed finalist reruns
          3. select winner
          4. fresh training + fresh random-state test
        """
        baseline = (
            self.run_baseline(baseline_params)
            if baseline_params is not None
            else None
        )
        stage3 = self.run_successive_halving()
        finalist_results = self.evaluate_finalists(stage3)
        winner = self.select_winner(finalist_results)

        if winner is None:
            result = {
                "baseline": baseline,
                "stage3": stage3,
                "finalists": finalist_results,
                "winner": None,
                "final_model": None,
                "message": (
                    "No feasible finalist satisfied the phase-gradient and "
                    "length constraints."
                ),
            }
            with (self.output_dir / "search_result.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(_jsonable(result), f, indent=2)
            return result

        final_model = self.final_test_best_seed(winner)

        result = {
            "baseline": baseline,
            "stage3": stage3,
            "finalists": finalist_results,
            "winner": winner,
            "final_model": final_model,
        }
        with (self.output_dir / "search_result.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(_jsonable(result), f, indent=2)

        return result
    
    def _config_completed_runs(
        self,
        config: ConfigRecord,
    ) -> list[TrialRecord]:
        return [
            r
            for r in config.seed_runs.values()
            if (
                r.status == "completed"
                and r.latest_validation_fidelity is not None
            )
        ]


    def _config_validation_fidelities(
        self,
        config: ConfigRecord,
    ) -> list[float]:
        return [
            float(r.latest_validation_fidelity)
            for r in self._config_completed_runs(config)
            if r.latest_validation_fidelity is not None
        ]


    def _config_score(
        self,
        config: ConfigRecord,
    ) -> float:
        """
        Hyperparameter configurations are ranked by their
        median validation fidelity across seeds.
        """
        fidelities = self._config_validation_fidelities(config)

        if not fidelities:
            return float("-inf")

        return float(statistics.median(fidelities))


    def _config_mean_fidelity(
        self,
        config: ConfigRecord,
    ) -> float:
        fidelities = self._config_validation_fidelities(config)

        if not fidelities:
            return float("-inf")

        return float(statistics.mean(fidelities))


    def _config_median_gradient(
        self,
        config: ConfigRecord,
    ) -> float:
        gradients = [
            float(r.final_phase_gradient)
            for r in self._config_completed_runs(config)
            if r.final_phase_gradient is not None
        ]

        if not gradients:
            return float("inf")

        return float(statistics.median(gradients))
    
    def _run_config_to_stage(
        self,
        config: ConfigRecord,
        stage_index: int,
    ) -> ConfigRecord:
        """
        Advance all cumulative seeds for one hyperparameter configuration
        to this stage's epoch budget.

        Existing seeds resume from checkpoint.
        New seeds start from epoch 0.
        """

        stage_number = stage_index + 1
        target_epoch = self.schedule.epoch_budgets[stage_index]

        required_seeds = self.schedule.cumulative_seeds(stage_index)

        for seed in required_seeds:

            if self._wall_time_exhausted():
                break

            # Create this seed only the first time we encounter it.
            if seed not in config.seed_runs:
                config.seed_runs[seed] = TrialRecord(
                    trial_id=f"{config.config_id}_seed_{seed}",
                    params=config.params,
                    training_seed=seed,
                )

            record = config.seed_runs[seed]

            # Existing records automatically resume because _run_to_budget()
            # sees epochs_completed > 0 and loads their checkpoint.
            self._run_to_budget(
                record,
                target_epoch,
                stage=stage_number,
                allow_convergence_stop=(stage_index == 2),
            )

        config.stage = stage_number
        return config

    def _config_completed_stage(
        self,
        config: ConfigRecord,
        stage_index: int,
    ) -> bool:

        required_seeds = self.schedule.cumulative_seeds(stage_index)
        stage_number = stage_index + 1

        for seed in required_seeds:
            if seed not in config.seed_runs:
                return False

            record = config.seed_runs[seed]

            if record.status != "completed":
                return False

            if record.stage != stage_number:
                return False

            if record.latest_validation_fidelity is None:
                return False

        return True
    
    def _promote_configs(
        self,
        configs: Sequence[ConfigRecord],
        keep_fraction: float,
        gradient_multiplier: float,
        stage_index: int,
    ) -> list[ConfigRecord]:

        eligible = [
            c
            for c in configs
            if self._config_completed_stage(c, stage_index)
        ]

        if not eligible:
            return []

        keep_n = max(
            1,
            math.ceil(len(eligible) * keep_fraction),
        )

        gradient_limit = (
            gradient_multiplier
            * self.constraints.max_phase_gradient_rad_per_m
        )

        within_limit = [
            c
            for c in eligible
            if self._config_median_gradient(c) <= gradient_limit
        ]

        outside_limit = [
            c
            for c in eligible
            if self._config_median_gradient(c) > gradient_limit
        ]

        # Primary ranking: median fidelity across seeds.
        within_limit.sort(
            key=self._config_score,
            reverse=True,
        )

        # If we don't have enough physically reasonable configs,
        # use the least-bad gradient violations to fill the stage.
        outside_limit.sort(
            key=lambda c: (
                self._config_median_gradient(c) / gradient_limit,
                -self._config_score(c),
            )
        )

        promoted = (within_limit + outside_limit)[:keep_n]

        promoted_ids = {
            c.config_id
            for c in promoted
        }

        # Delete continuation checkpoints only for entire configurations
        # that were eliminated.
        for config in configs:
            if config.config_id in promoted_ids:
                continue

            for run in config.seed_runs.values():
                if run.checkpoint_path:
                    try:
                        Path(run.checkpoint_path).unlink(
                            missing_ok=True
                        )
                        run.checkpoint_path = None
                    except Exception:
                        pass

        return promoted

    def run_successive_halving(
        self,
    ) -> list[ConfigRecord]:

        # ----------------
        # Stage 1: 2 seeds
        # ----------------

        stage1: list[ConfigRecord] = []

        for i in range(self.schedule.initial_trials):

            if self._wall_time_exhausted():
                break

            params = self.sample_trial_params()

            config = ConfigRecord(
                config_id=f"trial_{i:04d}",
                params=params,
            )

            self._run_config_to_stage(
                config,
                stage_index=0,
            )

            stage1.append(config)

        stage2 = self._promote_configs(
            stage1,
            self.schedule.keep_fractions[0],
            self.schedule.promotion_gradient_multipliers[0],
            stage_index=0,
        )

        # ----------------
        # Stage 2: 3 seeds
        # ----------------

        for config in stage2:

            if self._wall_time_exhausted():
                break

            self._run_config_to_stage(
                config,
                stage_index=1,
            )

        stage2_completed = [
            c
            for c in stage2
            if self._config_completed_stage(c, 1)
        ]

        stage3 = self._promote_configs(
            stage2_completed,
            self.schedule.keep_fractions[1],
            self.schedule.promotion_gradient_multipliers[1],
            stage_index=1,
        )

        # ----------------
        # Stage 3: 5 seeds
        # ----------------

        for config in stage3:

            if self._wall_time_exhausted():
                break

            self._run_config_to_stage(
                config,
                stage_index=2,
            )

            # Keep your individual seed logging.
            for run in config.seed_runs.values():
                if run.stage == 3:
                    self._append_summary_csv(
                        run,
                        label="stage3_seed",
                    )

        return [
            c
            for c in stage3
            if self._config_completed_stage(c, 2)
        ]
    
    def final_test_best_seed(
        self,
        winner: dict[str, Any],
    ) -> TrialRecord:

        feasible_runs = [
            r
            for r in winner["runs"]
            if (
                r.status == "completed"
                and self._is_physically_feasible(r)
                and r.latest_validation_fidelity is not None
            )
        ]

        if not feasible_runs:
            raise RuntimeError(
                "Winning configuration has no feasible seed runs."
            )

        best_run = max(
            feasible_runs,
            key=lambda r: float(
                r.latest_validation_fidelity
            ),
        )

        checkpoint = Path(best_run.checkpoint_path)

        bundle = None

        try:
            bundle = self.adapter.load_checkpoint(
                best_run.params,
                best_run.training_seed,
                checkpoint,
            )

            test_dataset = self.random_field_dataset_factory(
                best_run.params,
                self.data_config.n_test_states,
                self.data_config.test_dataset_seed,
            )

            test_fidelity = float(
                self.adapter.evaluate_fidelity(
                    bundle,
                    test_dataset,
                )
            )

            report = {
                "winner_params": _jsonable(best_run.params),
                "selected_training_seed": best_run.training_seed,
                "selected_validation_fidelity": (
                    best_run.latest_validation_fidelity
                ),
                "test_dataset_seed": (
                    self.data_config.test_dataset_seed
                ),
                "n_test_states": (
                    self.data_config.n_test_states
                ),
                "test_fidelity": test_fidelity,
                "max_phase_gradient": (
                    best_run.final_phase_gradient
                ),
            }

            with (
                self.output_dir / "final_report.json"
            ).open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            return best_run

        finally:
            if bundle is not None:
                self.adapter.release(bundle)