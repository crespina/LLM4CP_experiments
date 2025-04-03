import os
import pickle
from typing import Any


class ExperimentCache:
    """
    Class to handle caching of experiment results and progress.
    """

    def __init__(self, args):
        """
        Initialize the cache handler.
        """
        self.cache_dir = args.cache_dir
        os.makedirs(args.cache_dir, exist_ok=True)

    def get_cache_path(self, experiment_name: str, cache_type: str) -> str:
        """Get the path to a specific cache file"""
        return os.path.join(self.cache_dir, f"{experiment_name}_{cache_type}.pkl")

    def save_cache(self, experiment_name: str, cache_type: str, data: Any) -> None:
        """Save data to cache"""
        cache_path = self.get_cache_path(experiment_name, cache_type)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

    def load_cache(self, experiment_name: str, cache_type: str, default=None) -> Any:
        """Load data from cache if it exists, otherwise return default"""
        cache_path = self.get_cache_path(experiment_name, cache_type)
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return default

    def cache_exists(self, experiment_name: str, cache_type: str) -> bool:
        """Check if a cache file exists"""
        cache_path = self.get_cache_path(experiment_name, cache_type)
        return os.path.exists(cache_path)

    def clear_cache(self, experiment_name: str = None, cache_type: str = None) -> None:
        """
        Clear specific cache files or all caches
        
        If experiment_name and cache_type are both None, clears all caches
        If only experiment_name is provided, clears all caches for that experiment
        If both are provided, clears only that specific cache file
        """
        if experiment_name is None:
            # Clear all caches
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        elif cache_type is None:
            # Clear all caches for the given experiment
            for filename in os.listdir(self.cache_dir):
                if filename.startswith(f"{experiment_name}_") and filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        else:
            # Clear specific cache
            cache_path = self.get_cache_path(experiment_name, cache_type)
            if os.path.exists(cache_path):
                os.remove(cache_path)
