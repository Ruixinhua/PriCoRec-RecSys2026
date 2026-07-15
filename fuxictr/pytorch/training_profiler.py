# Modified by the PriCoRec authors in 2026.
"""
Model-agnostic training efficiency profiler for FuxiCTR models.

Collects the following metrics (per-epoch and aggregate):
  - Wall-clock training time per epoch (seconds)
  - Training throughput (samples/sec)
  - Peak GPU memory (MB)
  - Model parameter count (total and trainable)
  - Convergence speed (epoch at which early-stop fires or best metric is reached)
  - Inference latency (ms per batch)

Usage:
    from fuxictr.pytorch.training_profiler import TrainingProfiler

    profiler = TrainingProfiler(model, enabled=True)
    # ... training loop runs (profiler hooks into BaseModel automatically)
    profiler.report()                     # log summary
    profiler.to_dict()                    # get dict for serialization
    profiler.save_json("profiler.json")   # save to file

To attach to an existing BaseModel *after* construction:
    profiler = TrainingProfiler.attach(model)
"""

import time
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

import torch
import numpy as np

logger = logging.getLogger(__name__)


# =========================================================================
#  Data containers
# =========================================================================
@dataclass
class EpochStats:
    """Metrics collected for a single training epoch."""
    epoch: int = 0
    wall_clock_sec: float = 0.0
    num_samples: int = 0
    num_batches: int = 0
    throughput_samples_per_sec: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    train_loss: float = 0.0


@dataclass
class InferenceStats:
    """Metrics collected during inference latency measurement."""
    batch_size: int = 0
    num_batches: int = 0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0


@dataclass
class ProfilerSummary:
    """Aggregate profiling results."""
    model_id: str = ""
    total_params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
    total_epochs_run: int = 0
    converged_at_epoch: int = 0             # epoch where best metric was achieved
    early_stopped: bool = False
    mean_epoch_time_sec: float = 0.0
    total_training_time_sec: float = 0.0
    mean_throughput_samples_per_sec: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    epoch_stats: List[Dict[str, Any]] = field(default_factory=list)
    inference_stats: Optional[Dict[str, Any]] = None


# =========================================================================
#  Profiler implementation
# =========================================================================
class TrainingProfiler:
    """
    Model-agnostic training profiler that hooks into BaseModel's
    training lifecycle via monkey-patching.

    The profiler wraps:
      - train_epoch()  → to measure per-epoch wall-clock time, sample count, GPU memory
      - train_step()   → to count batches and accumulate loss
      - fit()          → to capture total training time and convergence info
    """

    def __init__(self, model=None, enabled: bool = True):
        self.enabled = enabled
        self._epoch_stats: List[EpochStats] = []
        self._inference_stats: Optional[InferenceStats] = None

        # Per-epoch accumulators (reset each epoch)
        self._epoch_start_time: float = 0.0
        self._epoch_sample_count: int = 0
        self._epoch_batch_count: int = 0
        self._epoch_loss_sum: float = 0.0

        # Overall training timer
        self._fit_start_time: float = 0.0
        self._total_training_time: float = 0.0

        # Convergence tracking (populated from model state after fit)
        self._converged_at_epoch: int = 0
        self._early_stopped: bool = False
        self._model_id: str = ""

        # Reference to the model (weak-style: we don't own it)
        self._model = None

        # Stash original methods so we can restore them
        self._orig_fit = None
        self._orig_train_epoch = None
        self._orig_train_step = None

        if model is not None:
            self.attach_to(model)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    @classmethod
    def attach(cls, model, enabled: bool = True) -> "TrainingProfiler":
        """Factory: create a profiler and attach it to *model*."""
        return cls(model=model, enabled=enabled)

    def attach_to(self, model):
        """Attach profiler hooks to an existing BaseModel instance."""
        self._model = model
        self._model_id = getattr(model, "model_id", model.__class__.__name__)
        if not self.enabled:
            return
        self._patch(model)

    def detach(self):
        """Remove profiler hooks, restoring original methods."""
        if self._model is not None and self._orig_fit is not None:
            self._unpatch(self._model)
        self._model = None

    def report(self):
        """Log a human-readable summary."""
        s = self.summary()
        lines = [
            "",
            "=" * 60,
            "  Training Efficiency Report",
            "=" * 60,
            f"  Model              : {s.model_id}",
            f"  Parameters (total) : {s.total_params:,}",
            f"  Parameters (train) : {s.trainable_params:,}",
            f"  Epochs run         : {s.total_epochs_run}",
            f"  Converged at epoch : {s.converged_at_epoch}",
            f"  Early stopped      : {s.early_stopped}",
            f"  Mean epoch time    : {s.mean_epoch_time_sec:.2f} sec",
            f"  Total train time   : {s.total_training_time_sec:.2f} sec",
            f"  Mean throughput    : {s.mean_throughput_samples_per_sec:,.0f} samples/sec",
            f"  Peak GPU mem       : {s.peak_gpu_memory_mb:.1f} MB",
        ]
        if s.inference_stats is not None:
            inf = s.inference_stats
            lines += [
                "  --- Inference ---",
                f"  Batch size         : {inf['batch_size']}",
                f"  Latency (mean)     : {inf['mean_ms']:.3f} ms",
                f"  Latency (p50)      : {inf['p50_ms']:.3f} ms",
                f"  Latency (p95)      : {inf['p95_ms']:.3f} ms",
                f"  Throughput         : {inf['throughput_samples_per_sec']:,.0f} samples/sec",
            ]
        lines.append("=" * 60)
        logger.info("\n".join(lines))

    def summary(self) -> ProfilerSummary:
        """Build and return an aggregated ProfilerSummary."""
        total_params, trainable_params = self._count_params()
        epoch_times = [e.wall_clock_sec for e in self._epoch_stats]
        throughputs = [e.throughput_samples_per_sec for e in self._epoch_stats]
        peak_mems = [e.peak_gpu_memory_mb for e in self._epoch_stats]

        return ProfilerSummary(
            model_id=self._model_id,
            total_params=total_params,
            trainable_params=trainable_params,
            non_trainable_params=total_params - trainable_params,
            total_epochs_run=len(self._epoch_stats),
            converged_at_epoch=self._converged_at_epoch,
            early_stopped=self._early_stopped,
            mean_epoch_time_sec=float(np.mean(epoch_times)) if epoch_times else 0.0,
            total_training_time_sec=self._total_training_time,
            mean_throughput_samples_per_sec=float(np.mean(throughputs)) if throughputs else 0.0,
            peak_gpu_memory_mb=float(max(peak_mems)) if peak_mems else 0.0,
            epoch_stats=[asdict(e) for e in self._epoch_stats],
            inference_stats=asdict(self._inference_stats) if self._inference_stats else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return summary as a plain dict (JSON-serialisable)."""
        return asdict(self.summary())

    def save_json(self, path: str):
        """Persist profiling results to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Profiler results saved to {path}")

    # ------------------------------------------------------------------
    #  Inference latency measurement (standalone, call after training)
    # ------------------------------------------------------------------
    def measure_inference_latency(
        self,
        data_generator,
        warmup: int = 10,
        repeats: int = 100,
    ) -> InferenceStats:
        """
        Measure inference latency on real data batches.

        Args:
            data_generator: an iterable yielding batch_data dicts (same format as training).
            warmup: number of warmup batches (not timed).
            repeats: number of timed batches.

        Returns:
            InferenceStats with latency statistics.
        """
        model = self._model
        if model is None:
            raise RuntimeError("No model attached. Call attach_to(model) first.")

        device = model.device
        model.eval()

        # Collect enough batches
        batches = []
        for i, batch_data in enumerate(data_generator):
            batches.append(batch_data)
            if len(batches) >= warmup + repeats:
                break

        if len(batches) < warmup + 1:
            logger.warning("Not enough batches for latency measurement.")
            return InferenceStats()

        # Determine batch size from first batch
        first_key = next(iter(batches[0]))
        batch_size = batches[0][first_key].shape[0]

        # Warmup
        with torch.no_grad():
            for i in range(min(warmup, len(batches))):
                _ = model.forward(batches[i])
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Timed runs
        latencies = []
        timed_batches = batches[warmup: warmup + repeats]
        with torch.no_grad():
            for batch_data in timed_batches:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model.forward(batch_data)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

        latencies = np.array(latencies)
        stats = InferenceStats(
            batch_size=batch_size,
            num_batches=len(latencies),
            mean_ms=float(np.mean(latencies)),
            std_ms=float(np.std(latencies)),
            p50_ms=float(np.percentile(latencies, 50)),
            p95_ms=float(np.percentile(latencies, 95)),
            throughput_samples_per_sec=batch_size / (float(np.mean(latencies)) / 1000.0)
            if np.mean(latencies) > 0 else 0.0,
        )
        self._inference_stats = stats
        model.train()
        return stats

    # ------------------------------------------------------------------
    #  Internal: monkey-patching
    # ------------------------------------------------------------------
    def _patch(self, model):
        """Replace model methods with profiled versions."""
        import types

        self._orig_fit = model.fit
        self._orig_train_epoch = model.train_epoch
        self._orig_train_step = model.train_step

        profiler = self  # close over self

        # ---- wrapped fit ----
        def profiled_fit(self_model, data_generator, epochs=1, validation_data=None,
                         max_gradient_norm=10., **kwargs):
            if self_model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self_model.device)

            profiler._fit_start_time = time.perf_counter()
            profiler._orig_fit(data_generator, epochs=epochs,
                               validation_data=validation_data,
                               max_gradient_norm=max_gradient_norm, **kwargs)
            profiler._total_training_time = time.perf_counter() - profiler._fit_start_time

            # Record convergence info from model state
            profiler._early_stopped = getattr(self_model, '_stop_training', False)
            # Find the epoch that achieved the best metric
            if profiler._epoch_stats:
                profiler._converged_at_epoch = profiler._epoch_stats[-1].epoch
                # Walk backwards: the epoch before stopping_steps started increasing
                stopping = getattr(self_model, '_stopping_steps', 0)
                best_epoch_idx = len(profiler._epoch_stats) - 1 - stopping
                if 0 <= best_epoch_idx < len(profiler._epoch_stats):
                    profiler._converged_at_epoch = profiler._epoch_stats[best_epoch_idx].epoch

            profiler.report()

        model.fit = types.MethodType(profiled_fit, model)

        # ---- wrapped train_epoch ----
        def profiled_train_epoch(self_model, data_generator):
            profiler._epoch_sample_count = 0
            profiler._epoch_batch_count = 0
            profiler._epoch_loss_sum = 0.0

            if self_model.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self_model.device)

            profiler._epoch_start_time = time.perf_counter()
            profiler._orig_train_epoch(data_generator)
            elapsed = time.perf_counter() - profiler._epoch_start_time

            peak_mem = 0.0
            if self_model.device.type == "cuda":
                peak_mem = torch.cuda.max_memory_allocated(self_model.device) / (1024 ** 2)

            throughput = (profiler._epoch_sample_count / elapsed) if elapsed > 0 else 0.0
            avg_loss = (profiler._epoch_loss_sum / max(profiler._epoch_batch_count, 1))

            epoch_idx = getattr(self_model, '_epoch_index', len(profiler._epoch_stats))
            stats = EpochStats(
                epoch=epoch_idx + 1,
                wall_clock_sec=elapsed,
                num_samples=profiler._epoch_sample_count,
                num_batches=profiler._epoch_batch_count,
                throughput_samples_per_sec=throughput,
                peak_gpu_memory_mb=peak_mem,
                train_loss=avg_loss,
            )
            profiler._epoch_stats.append(stats)
            logger.info(
                f"[Profiler] Epoch {stats.epoch}: "
                f"{stats.wall_clock_sec:.1f}s, "
                f"{stats.throughput_samples_per_sec:,.0f} samples/s, "
                f"peak GPU mem={stats.peak_gpu_memory_mb:.1f} MB"
            )

        model.train_epoch = types.MethodType(profiled_train_epoch, model)

        # ---- wrapped train_step ----
        def profiled_train_step(self_model, batch_data):
            # Count samples in batch
            first_key = next(iter(batch_data))
            profiler._epoch_sample_count += batch_data[first_key].shape[0]
            profiler._epoch_batch_count += 1

            loss = profiler._orig_train_step(batch_data)
            profiler._epoch_loss_sum += loss.item()
            return loss

        model.train_step = types.MethodType(profiled_train_step, model)

    def _unpatch(self, model):
        """Restore original methods."""
        import types
        if self._orig_fit is not None:
            model.fit = types.MethodType(
                lambda self_model, *a, **kw: self._orig_fit(*a, **kw), model
            ) if callable(self._orig_fit) else self._orig_fit
        if self._orig_train_epoch is not None:
            model.train_epoch = types.MethodType(
                lambda self_model, *a, **kw: self._orig_train_epoch(*a, **kw), model
            ) if callable(self._orig_train_epoch) else self._orig_train_epoch
        if self._orig_train_step is not None:
            model.train_step = types.MethodType(
                lambda self_model, *a, **kw: self._orig_train_step(*a, **kw), model
            ) if callable(self._orig_train_step) else self._orig_train_step
        self._orig_fit = None
        self._orig_train_epoch = None
        self._orig_train_step = None

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    def _count_params(self):
        """Return (total_params, trainable_params) for the attached model."""
        if self._model is None:
            return 0, 0
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        return total, trainable
