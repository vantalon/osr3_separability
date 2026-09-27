import numpy as np
from scipy.linalg import sqrtm
from utils import *
from sampler import sample_dm_separable_osr3_mixed



def hermitian_schmidt_decomposition(A, m=2, tol=1e-15):
    """
    Compute the Hermitian Schmidt Decomposition of A ∈ M2 ⊗ Mm.

    Returns (lambdas, Es, Fs) with A = sum_i lambda_i * E_i ⊗ F_i,
    {E_i} orthonormal in M2, {F_i} orthonormal in Mm,
    lambda_i real and positive (sign absorbed into F_i).
    """
    basis2 = [ sigma / np.sqrt(2) for sigma in pauli_basis ]  # 4 elements
    basism = [ tau / np.sqrt(m) for tau in [np.eye(m)] + GellMann(m) ]  # m² elements

    n2 = len(basis2)
    nm = len(basism)

    # Build coefficient matrix C[i,j] = Tr(E_i ⊗ F_j, A)
    # = Tr_left(E_i * Tr_right(F_j, A)) 
    C = np.zeros((n2, nm), dtype=float)
    for i, E in enumerate(basis2):
        for j, F in enumerate(basism):
            C[i, j] = np.real(np.trace(np.kron(E, F).conj().T @ A))

    # SVD of coefficient matrix
    U, s, Vt = np.linalg.svd(C, full_matrices=False)

    # Threshold small singular values
    mask = s > tol
    s, U, Vt = s[mask], U[:, mask], Vt[mask, :]

    # Build Schmidt operators
    Es = [sum(U[i, k] * basis2[i] for i in range(n2)) for k in range(len(s))]
    Fs = [sum(Vt[k, j] * basism[j] for j in range(nm)) for k in range(len(s))]
    lambdas = s.copy()

    # Make all lambdas positive (absorb sign into F)
    for k in range(len(lambdas)):
        if lambdas[k] < 0:
            lambdas[k] = -lambdas[k]
            Fs[k] = -Fs[k]

    return lambdas, Es, Fs



def rotate_hsd_positive_first(lambdas, Es, Fs):
    """
    Rotate the HSD  M = sum_i lambdas[i] * Es[i] ⊗ Fs[i]  so that the first
    left operator Es_new[0] is PSD, leaving M invariant.

    Assumes {Es[i]} orthonormal w.r.t. hs_inner. Returns (lambdas_new,
    Es_new, Fs_new) with Es_new, Fs_new unit-norm and lambdas_new[k] >= 0.
    """
    r = len(lambdas)
    if r == 0:
        return lambdas, Es, Fs

    # 1. PSD direction inside span(Es); normalize it. (use positivity of partial trace)
    E0 = np.sum([lambdas[i] * np.trace(Fs[i])  * Es[i] for i in range(len(lambdas)) ], axis=0)
    n0 = np.sqrt(hs_inner(E0, E0))
    if n0 < 1e-12:
        raise ValueError("Pb with partial trace.")
    E0 = E0 / n0

    # 2. Complete E0 to an orthonormal basis of span(Es).
    rest = [Es[i] - hs_inner(E0, Es[i]) * E0 for i in range(r)]
    Es_new = [E0] + gs_orthonormalize(rest)

    # 3. Invariance forces, for coords c_i = <Es[i], E_k>,
    #       G_k = sum_i c_i * lambdas[i] * Fs[i],
    #    then lambdas_new[k] = ||G_k||,  Fs_new[k] = G_k / ||G_k||.
    lambdas_new, Fs_new = [], []
    for k, E_k in enumerate(Es_new):
        G = sum(hs_inner(Es[i], E_k) * lambdas[i] * Fs[i] for i in range(r))
        lam = np.sqrt(hs_inner(G, G))
        lambdas_new.append(lam)
        Fs_new.append(Fs[k] if lam < 1e-12 else G / lam)   # F arbitrary when lam=0

    return lambdas_new, Es_new, Fs_new


def cariello_decomposition(rho, m=2, tol=1e-15, verbose=False):
    """
    Compute the Carriello LOCC filtering of rho ∈ M2 ⊗ Mm, returning (rho_new, Filter) with rho_new = Filter† rho Filter / Tr(Filter† rho Filter)
    """
    lambdas, Es, Fs = hermitian_schmidt_decomposition(rho, m=m, tol=tol)
    r = len(lambdas)
    assert r == 3, "rho must have OSR = 3 for Cariello decomposition."
    lambdas, Es, Fs = rotate_hsd_positive_first(lambdas, Es, Fs)

    B1 = Es[0]
    B2 = Es[1]
    B3 = Es[2]
    # Verify B1 is invertible and well-conditioned
    if verbose:
        print("=== HSD-based initial state ===")
        print(f"Singular values of B1: {np.linalg.svd(B1, compute_uv=False)}")
    R = np.linalg.cholesky(B1)
    assert np.allclose(R @ R.conj().T, B1)
    R_inv = np.linalg.inv(R)
    B2_p = R_inv @ B2 @ R_inv.conj().T
    eig, U = np.linalg.eigh(B2_p)
    D = np.diag(eig)
    assert np.all(np.isclose(U @ D @ U.conj().T, B2_p))
    assert np.abs(D[0, 0] - D[1, 1]) > 1e-5, "D = \lambda Id"  # well-separated eigenvalues
    B3_p = U.conj().T @ R_inv @ B3 @ R_inv.conj().T @ U

    assert (np.abs(B3_p[0, 1]) > 1e-5), "B3_p is diagonal"  
    V = np.diag([1, B3_p[0, 1]] )
    Filter = np.kron( R_inv.conj().T @ U @ V.conj().T, np.eye(m) )
    F = (
        Filter.conj().T
        @ rho
        @ Filter
    )

    rho = F / np.trace(F)
    return rho, Filter

def maximally_mixed_B_marginals(rho, m=2):
    """
    Given rho ∈ M2 ⊗ Mm, return the LOCC filtering F such that F† rho F / Tr(F† rho F) has maximally mixed B_marginals.
    """
    # Partial traces
    rho_B = partial_trace_A(rho, m)

    # Inverse square roots
    sqrt_rho_B_inv = np.linalg.inv(sqrtm(rho_B))

    # Construct the filter
    Filter = np.kron(I2, sqrt_rho_B_inv)

    # Apply the filter
    F = Filter.conj().T @ rho @ Filter
    rho_new = F / np.trace(F)

    return rho_new, Filter



if __name__ == "__main__":
    # Example usage
    m = 4
    rng = np.random.default_rng(0)
    rho = sample_dm_separable_osr3_mixed(m=m, rng=rng)[0]

    lambdas, Es, Fs = hermitian_schmidt_decomposition(rho, m)
    assert np.allclose(rho, sum(lambdas[i] * np.kron(Es[i], Fs[i]) for i in range(len(lambdas)))), "Decomposition failed"

    lambdas, Es, Fs = rotate_hsd_positive_first(lambdas, Es, Fs)
    assert np.all(np.linalg.eigvalsh(Es[0]) >= -1e-12), "First Es is not PSD after rotation"

    rho_new, F = cariello_decomposition(rho, m)
    rho_pt = partial_transpose_A(rho_new,m)
    assert np.allclose(rho_pt, rho_new), "Partial transpose invariance failed"

    rho_mm, F_mm = maximally_mixed_B_marginals(rho_new, m)
    T = build_T_correlation(rho_mm, m)
    print(T)

