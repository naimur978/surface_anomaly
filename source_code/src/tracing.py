"""Execution tracing utilities for MLflow."""

import time
import mlflow
from functools import wraps
from typing import Callable, Any, Optional, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


class ExecutionTracer:
    """Context manager for tracing execution blocks.

    Logs wall-clock duration to MLflow as a metric, and on failure also
    logs the exception as a param. Prints a one-line status to stdout
    either way. Used to bracket major pipeline stages (feature extraction,
    training, evaluation) so their timing shows up in the MLflow run.
    """

    def __init__(self, step_name: str) -> None:
        """Initialize tracer.

        Args:
            step_name: Name of the step being traced
        """
        self.step_name = step_name
        self.start_time: Optional[float] = None

    def __enter__(self) -> 'ExecutionTracer':
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[Any]) -> bool:
        """Stop timing and log to MLflow."""
        duration = time.time() - self.start_time
        mlflow.log_metric(f"trace_{self.step_name}_duration_seconds", duration)

        if exc_type is None:
            print(f"[OK] {self.step_name}: {duration:.2f}s")
        else:
            mlflow.log_param(f"trace_{self.step_name}_error", str(exc_val))
            print(f"[FAIL] {self.step_name}: {duration:.2f}s (error)")

        return False  # Don't suppress exceptions


def log_trace(step_name: str) -> Callable[[F], F]:
    """Decorator form of ExecutionTracer, for tracing a whole function.

    Args:
        step_name: Name of the step being traced
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with ExecutionTracer(step_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
