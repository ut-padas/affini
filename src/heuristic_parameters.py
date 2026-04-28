"""
Heuristic parameters for HATANN.
"""


def get_N_TRAIN(dim: int, *, base: float = 1.3, factor: int = 50_000) -> int:
    """
    Heuristic to determine N_TRAIN based on dimension and tolerance.
        factor * (base)^dim
    """
    N_TRAIN = int(factor * (base**dim))
    return N_TRAIN


def get_NUM_INITIAL_CLUSTERS(dim: int, *, base: float = 1.3, factor: int = 25) -> int:
    """
    Heuristic to determine N_INIT_CLUSTERS based on dimension.
        factor * (base)^dim
    """
    N_INIT_CLUSTERS = int(factor * (base**dim))
    return N_INIT_CLUSTERS


def get_MIN_PTS_PER_CLUSTER(dim: int, *, base: float = 1.3, factor: int = 100) -> int:
    """
    Heuristic to determine N_MIN_PER_CLUSTER based on dimension.
        factor * (base)^dim
    """
    N_MIN_PER_CLUSTER = int(factor * (base**dim))
    return N_MIN_PER_CLUSTER
