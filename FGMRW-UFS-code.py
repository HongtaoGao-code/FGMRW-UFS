# Author       : Hongtao Gao
# Email        : 12123@163.com
# Title        : Unsupervised Feature Selection Using Fuzzy Graph Momentum Random Walk in Bi-level Granular-Ball Knowledge Space
# Abbreviation : FGMRW-UFS
# Journal      : TKDE (IEEE Transactions on Knowledge and Data Engineering)

import math
import numpy as np
import scipy.io as sio

from GB import getGranularBall2


EPS = 1e-12


# -----------------------------
# Matching kernel function
# -----------------------------
def matching_kernel(a, b):
    """
    Compute the matching-kernel similarity for nominal attributes.
    """
    return 1 if a == b else 0


# -----------------------------
# Gaussian kernel function
# -----------------------------
def gaussian_kernel(a, b, delta):
    """
    Compute the Gaussian-kernel similarity for numerical attributes.
    """
    return np.exp(-np.linalg.norm(a - b) ** 2 / (2 * delta ** 2 + EPS))

def safe_divide(a, b, fill_value=0.0):
    """
    Safe division with broadcasting support to avoid division-by-zero errors.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)

    result = np.where(np.isfinite(result), result, fill_value)
    return result


def l1_projection(x, s=1.0):
    """
    Project vector x onto the L1 ball (||x||_1 <= s).
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.abs(x)

    if u.sum() <= s:
        return x

    theta = np.sort(u)[::-1]
    cumsum = np.cumsum(theta)
    rho = np.max(np.where(theta * (np.arange(len(x)) + 1) > (cumsum - s))[0])
    lambda_ = (cumsum[rho] - s) / (rho + 1)

    return np.sign(x) * np.maximum(u - lambda_, 0.0)


def remove_constant_feature_columns_keep_label(M):
    """
    Remove all-one and all-zero feature columns while keeping the last label column.
    """
    X = M[:, :-1]
    y = M[:, -1:]

    valid_mask = ~(np.all(X == 1, axis=0) | np.all(X == 0, axis=0))
    X_new = X[:, valid_mask]

    return np.hstack([X_new, y])


def gaussian_kernel_matrix(col_data, col_gb, delta):
    """
    Vectorized Gaussian kernel.
    col_data: (n,)
    col_gb:   (n_gb,)
    return:   (n, n_gb)
    """
    diff = col_data[:, None] - col_gb[None, :]
    return np.exp(-(diff ** 2) / (2.0 * delta ** 2 + EPS))


def matching_kernel_matrix(col_data, col_gb):
    """
    Vectorized matching kernel.
    col_data: (n,)
    col_gb:   (n_gb,)
    return:   (n, n_gb)
    """
    return (col_data[:, None] == col_gb[None, :]).astype(np.float64)


def FGMRW_UFS(data, alpha, s, beta=0.2, max_iter=10000, tol=0.0001, decay_rate=0.1):
    """
    Input:
        data  : data matrix without labels (n, m)
        alpha : Gaussian kernel parameter
        s     : L1 projection radius
    Output:
        sorted_indices_FS : feature indices ranked in descending order of importance
    """
    data = np.asarray(data, dtype=np.float64)

    # Granular-ball samples
    data_gb = getGranularBall2(data)[0]
    data_gb = np.asarray(data_gb, dtype=np.float64)

    n = data.shape[0]
    n_gb, m = data_gb.shape

    # -----------------------------
    # Compute the fuzzy relation matrix for each attribute
    # -----------------------------
    Mc = []  # Use a list instead of a dict for faster access

    for attr_idx in range(m):
        col_data = data[:, attr_idx]
        col_gb = data_gb[:, attr_idx]

        # Determine the attribute type according to the specified rule
        if np.min(data[:, attr_idx]) == 0 and np.max(data[:, attr_idx]) == 1:
            # Numerical attribute: Gaussian kernel
            r = gaussian_kernel_matrix(col_data, col_gb, alpha)
        else:
            # Nominal attribute: matching kernel
            r = matching_kernel_matrix(col_data, col_gb)

        Mc.append(r)

    # -----------------------------
    # Construct the feature fuzzy graph (FSG)
    # -----------------------------
    FSG = np.zeros((m, m), dtype=np.float64)

    for i in range(m):
        Dr_i = Mc[i]
        one_minus_Dr_i = 1.0 - Dr_i

        for j in range(m):
            Dr_j = Mc[j]

            LAP_min_x = np.min(np.maximum(one_minus_Dr_i, Dr_j), axis=1)
            LAP_max_x = np.max(np.minimum(Dr_i, Dr_j), axis=1)

            ratio = safe_divide(LAP_min_x, LAP_max_x, fill_value=0.0)
            FSG[i, j] = np.max(ratio)

    # -----------------------------
    # Random walk: inertial random walk
    # -----------------------------
    phi = np.zeros(m, dtype=np.float64)

    degree = FSG.sum(axis=1)
    P = safe_divide(FSG, degree[:, None] + 1e-8, fill_value=0.0)

    d = 1 / (1 + np.log(np.mean(degree) + 1 + EPS))
    pi_0 = np.ones(m, dtype=np.float64) / m
    pi_t = pi_0.copy()
    pi_prev = pi_0.copy()

    visit_count = np.zeros(m, dtype=np.float64)

    for i in range(max_iter):
        # 1. Momentum term
        momentum = (1 - beta) * pi_t + beta * pi_prev

        # 2. Visit penalty
        penalty = 1 / (1 + decay_rate * visit_count)
        momentum *= penalty

        # 3. Update state
        pi_new = d * pi_0 + (1 - d) * (momentum @ P)

        # 4. Update visit count
        pi_new_sum = np.sum(pi_new)
        if pi_new_sum > EPS:
            visit_count += pi_new / pi_new_sum

        # 5. L1 projection for sparsification
        pi_new = l1_projection(pi_new, s)

        # 6. Convergence check
        if np.linalg.norm(pi_new - pi_t, 1) < tol:
            pi_t = pi_new
            break

        pi_prev = pi_t.copy()
        pi_t = pi_new

        if i == max_iter - 1:
            print("Warning: did not converge within the maximum number of iterations")

    # -----------------------------
    # Normalize to [0, 1]
    # -----------------------------
    pi_t_w = pi_t
    min_v = np.min(pi_t_w)
    max_v = np.max(pi_t_w)

    if max_v - min_v < EPS:
        phi[:] = 0.0
    else:
        phi[:] = (pi_t_w - min_v) / (max_v - min_v)

    FS = phi

    # Sort feature indices in descending order of scores
    sorted_indices_FS = np.argsort(-FS)
    return sorted_indices_FS


# -----------------------------
# Test
# -----------------------------
def myTest():
    path = "./Example" + ".mat"
    mat_data = sio.loadmat(path)
    M = mat_data['Example']

    M = remove_constant_feature_columns_keep_label(M)
    fe_list = FGMRW_UFS(M, alpha=0.1, s=0.8) # Parameter Traversal
    print("feature ranking:")
    print(fe_list)


if __name__ == '__main__':
    myTest()