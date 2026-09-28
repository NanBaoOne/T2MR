# graph/tools.py

import numpy as np


def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0  # undirected
    return A


def k_adjacency(A, k, with_self=False, self_factor=1):
    """
    Compute the k-order adjacency matrix.

    Args:
        A (np.ndarray): Binary adjacency matrix (N, N)
        k (int): Hop order
        with_self (bool): Whether to include self-loop
        self_factor (float): Weight for self-loop (usually 1)

    Returns:
        Ak (np.ndarray): k-hop adjacency matrix
    """
    assert isinstance(A, np.ndarray)
    assert A.ndim == 2 and A.shape[0] == A.shape[1]

    if k == 0:
        if with_self:
            return np.eye(len(A)) * self_factor
        else:
            return np.zeros_like(A)

    Ak = np.minimum(np.linalg.matrix_power(A, k), 1)

    if with_self:
        Ak += np.eye(len(A)) * self_factor

    return Ak


def normalize_adjacency_matrix(A):
    """
    Symmetric normalization: D^(-1/2) A D^(-1/2)
    """
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.power(D, -0.5, where=(D > 0))
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(D_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt