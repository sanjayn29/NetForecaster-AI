"""
src/utils/seed.py
-----------------
Reproducibility: set all random seeds for Python, NumPy, and PyTorch.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use.  Defaults to 42.

    Notes
    -----
    - Sets Python's built-in ``random`` seed.
    - Sets NumPy's global seed.
    - Sets PyTorch CPU and GPU seeds (if PyTorch is installed).
    - Sets ``PYTHONHASHSEED`` environment variable.
    - For full determinism in PyTorch CUDA operations, also sets
      ``torch.backends.cudnn.deterministic = True``.  This may reduce
      GPU performance but ensures reproducibility.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not yet installed; skip silently
