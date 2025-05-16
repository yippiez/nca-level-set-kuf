import os
import pickle
import pathlib
import sys
import numpy as np
import json
from typing import Any, Dict, List, Union, Optional

CACHE_DIR = pathlib.Path(__file__).parent.parent / "cache"

def cache_save(obj: Any, cache_name: str) -> str:
    """Save a Python object to the cache directory with appropriate format based on type."""
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(exist_ok=True, parents=True)
    
    # Determine file extension and save method based on object type
    if isinstance(obj, np.ndarray):
        file_path = CACHE_DIR / f"{cache_name}.npy"
        np.save(file_path, obj)
    elif 'pandas' in sys.modules and isinstance(obj, sys.modules['pandas'].DataFrame):
        file_path = CACHE_DIR / f"{cache_name}.csv"
        obj.to_csv(file_path, index=False)
    elif 'torch' in sys.modules and isinstance(obj, sys.modules['torch'].Tensor):
        file_path = CACHE_DIR / f"{cache_name}.pt"
        sys.modules['torch'].save(obj, file_path)
    elif isinstance(obj, (dict, list)):
        try:
            # Try to save as JSON if possible
            file_path = CACHE_DIR / f"{cache_name}.json"
            with open(file_path, 'w') as f:
                json.dump(obj, f)
        except TypeError:
            # Fall back to pickle if not JSON serializable
            file_path = CACHE_DIR / f"{cache_name}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(obj, f)
    else:
        # Default to pickle for other types
        file_path = CACHE_DIR / f"{cache_name}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
    
    return str(file_path)



def cache_get_numpy(cache_name: str) -> np.ndarray:
    """Get a cached numpy array."""
    file_path = CACHE_DIR / f"{cache_name}.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"No numpy cache found for {cache_name}")
    
    return np.load(file_path)


def cache_get_pandas(cache_name: str):
    """Get a cached pandas DataFrame."""
    import pandas as pd
    file_path = CACHE_DIR / f"{cache_name}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"No pandas cache found for {cache_name}")
    
    return pd.read_csv(file_path)


def cache_get_torch(cache_name: str):
    """Get a cached PyTorch tensor."""
    import torch
    file_path = CACHE_DIR / f"{cache_name}.pt"
    if not file_path.exists():
        raise FileNotFoundError(f"No PyTorch cache found for {cache_name}")
    
    return torch.load(file_path)


def cache_get_json(cache_name: str) -> Union[Dict, List]:
    """Get a cached JSON object."""
    file_path = CACHE_DIR / f"{cache_name}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"No JSON cache found for {cache_name}")
    
    with open(file_path, 'r') as f:
        return json.load(f)


def cache_get_pickle(cache_name: str) -> Any:
    """Get a cached pickle object."""
    file_path = CACHE_DIR / f"{cache_name}.pkl"
    if not file_path.exists():
        raise FileNotFoundError(f"No pickle cache found for {cache_name}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def cache_exists(cache_name: str, extension: Optional[str] = None) -> bool:
    """Check if a cache file exists."""
    if extension:
        file_path = CACHE_DIR / f"{cache_name}.{extension}"
        return file_path.exists()
    else:
        # Check all supported extensions
        extensions = ["npy", "csv", "pt", "json", "pkl"]
        return any((CACHE_DIR / f"{cache_name}.{ext}").exists() for ext in extensions)

