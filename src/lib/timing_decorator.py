import functools
import time


def format_duration_hms(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}小时{minutes}分{secs}秒"


def timing_decorator(log_func):
    def real_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            hms = format_duration_hms(elapsed)
            log_func(
                f"Execution time of {func.__name__}(): {hms} ({elapsed:.1f} seconds)"
            )
            return result
        return wrapper
    return real_decorator
