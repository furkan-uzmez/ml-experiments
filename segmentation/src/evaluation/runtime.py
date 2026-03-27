import time
import functools
import torch
from typing import Callable, Any, TypeVar, cast

F = TypeVar('F', bound=Callable[..., Any])


def track_inference_time(func: F) -> F:
    """Decorator to measure the wall-clock execution time of a function.
    
    The measured time is stored in the function's return dictionary under
    the key 'inference_time_seconds'. If the function does not return a
    dictionary, a ValueError is raised.
    
    Args:
        func: The function to be decorated. Must return a dict.
        
    Returns:
        The decorated function with timing capabilities.
        
    Example:
        >>> @track_inference_time
        ... def my_model_infer(inputs):
        ...     time.sleep(1) # simulate inference
        ...     return {'mask': [0, 1]}
        >>> result = my_model_infer(None)
        >>> 'inference_time_seconds' in result
        True
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        
        if not isinstance(result, dict):
            raise ValueError(
                f"@track_inference_time requires the decorated function to "
                f"return a dictionary. Got {type(result)} instead."
            )
            
        result['inference_time_seconds'] = elapsed_time
        return result
        
    return cast(F, wrapper)


def track_peak_gpu_memory(func: F) -> F:
    """Decorator to measure the peak GPU memory usage during a function call.
    
    Uses PyTorch's CUDA memory tracking. Resets peak memory stats before
    execution, tracks the maximum allocated memory during execution, and
    records it in MB.
    
    If CUDA is not available, records 0.0 MB.
    
    The measured memory is stored in the function's return dictionary under
    the key 'peak_gpu_memory_mb'.
    
    Args:
        func: The function to be decorated. Must return a dict.
        
    Returns:
        The decorated function with GPU memory tracking capabilities.
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Check if CUDA is reachable
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            torch.cuda.reset_peak_memory_stats()
            
        result = func(*args, **kwargs)
        
        if not isinstance(result, dict):
            raise ValueError(
                f"@track_peak_gpu_memory requires the decorated function to "
                f"return a dictionary. Got {type(result)} instead."
            )
            
        if cuda_available:
            peak_memory_bytes = torch.cuda.max_memory_allocated()
            peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        else:
            peak_memory_mb = 0.0
            
        result['peak_gpu_memory_mb'] = peak_memory_mb
        return result
        
    return cast(F, wrapper)
