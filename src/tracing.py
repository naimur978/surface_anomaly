"""Execution tracing utilities for MLflow."""

import time
import mlflow
from functools import wraps
from typing import Callable


def log_trace(step_name: str):
    """Decorator to log execution time of a step to MLflow.

    Args:
        step_name: Name of the step being traced
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
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

    def __init__(self, step_name: str):
        """Initialize tracer.

        Args:
            step_name: Name of the step being traced
        """
        self.step_name = step_name
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
