"""Execution tracing utilities for MLflow."""

import time
import mlflow
from functools import wraps
from typing import Callable, Any, Optional, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


def log_trace(step_name: str) -> Callable[[F], F]:
    """Decorator to log execution time of a step to MLflow.

    Args:
        step_name: Name of the step being traced
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                mlflow.log_metric(f"trace_{step_name}_duration_seconds", duration)
                print(f"✓ {step_name}: {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                mlflow.log_metric(f"trace_{step_name}_duration_seconds", duration)
                mlflow.log_param(f"trace_{step_name}_error", str(e))
                print(f"✗ {step_name}: {duration:.2f}s (error)")
                raise
        return wrapper
    return decorator


class ExecutionTracer:
    """Context manager for tracing execution blocks."""

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

        if exc_type is None:
            mlflow.log_metric(f"trace_{self.step_name}_duration_seconds", duration)
            print(f"✓ {self.step_name}: {duration:.2f}s")
        else:
            mlflow.log_metric(f"trace_{self.step_name}_duration_seconds", duration)
            mlflow.log_param(f"trace_{self.step_name}_error", str(exc_val))
            print(f"✗ {self.step_name}: {duration:.2f}s (error)")

        return False  # Don't suppress exceptions
