import numpy as np

def sample_uniform_ints(N: int, K: int, seed: int):
    """Sample N uniform integers from [0, K] using a seeded NumPy RNG.

    Uses NumPy's Generator (PCG64) for reproducible, vectorized sampling.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, K + 1, size=N, dtype=np.int64)