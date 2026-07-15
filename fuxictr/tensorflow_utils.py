# Modified by the PriCoRec authors in 2026.
import logging
from typing import Any, Optional


_TF_CPU_ONLY_CONFIGURED = False


def configure_tensorflow_cpu_only(tf: Optional[Any] = None, logger: Optional[logging.Logger] = None) -> Any:
    """Hide GPUs from TensorFlow without changing CUDA visibility for PyTorch."""
    global _TF_CPU_ONLY_CONFIGURED

    if tf is None:
        import tensorflow as tf  # type: ignore

    if _TF_CPU_ONLY_CONFIGURED:
        return tf

    active_logger = logger or logging.getLogger(__name__)
    try:
        tf.config.set_visible_devices([], "GPU")
        visible_gpus = tf.config.get_visible_devices("GPU")
        active_logger.info(
            "Configured TensorFlow CPU-only mode; TensorFlow visible GPUs=%d.",
            len(visible_gpus),
        )
    except RuntimeError as exc:
        active_logger.warning(
            "Could not hide TensorFlow GPUs because the TensorFlow runtime is already initialized: %s",
            exc,
        )
    except Exception as exc:
        active_logger.warning("Could not configure TensorFlow CPU-only mode: %s", exc)

    _TF_CPU_ONLY_CONFIGURED = True
    return tf
