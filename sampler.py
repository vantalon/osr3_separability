import numpy as np
from utils import osr,numerical_range

def haar(n, rng=np.random.default_rng()):
    """Sample a Haar-random unitary matrix in C^{n x n}."""
    # complex Ginibre + QR -> Haar-unitary
    M = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2)
    Q, R = np.linalg.qr(M)
    d = np.diag(R)
    return Q * (d / np.abs(d))  


########################## Density Matrices #########################


def sample_dm_separable_osr3_pure(m = 2, rng = np.random.default_rng()):
    """
    Build a random separable state in M2⊗Mm with OSR exactly 3.
    Strategy: sum of 3 product pure states
    """
    
    dim = 2 * m
    state = np.zeros((dim, dim), dtype=complex)
    for _ in range(3):
        v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
        w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        v /= np.linalg.norm(v)
        w /= np.linalg.norm(w)
        psi = np.kron(v, w)
        state += np.outer(psi, psi.conj())
    state = state / np.trace(state)  # normalize
    return state

def sample_dm_separable_osr3_mixed(m = 2, rng = np.random.default_rng(), complex_valued=True):
    """
    Build a random separable state in M2⊗M2 with OSR exactly 3 and rank 4.
    Strategy: sum of 3 product pure states, then check OSR.
    """

    def random_matrix(shape):
        if complex_valued:
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)
        else:
            return rng.normal(size=shape)

    def make_hermitian_psd(X):
        return X @ X.conj().T

    A_list = [make_hermitian_psd(random_matrix((2, 2))) for _ in range(3)]
    B_list = [make_hermitian_psd(random_matrix((m, m))) for _ in range(3)]

    A_list = [A / np.trace(A) for A in A_list]
    B_list = [B / np.trace(B) for B in B_list]
    rho = sum(np.kron(A_list[r], B_list[r]) for r in range(3))
    norm = np.trace(rho).real
    rho /= norm  # normalize to trace 1
    A_list = [A / norm for A in A_list]
    B_list = [B for B in B_list]

    return rho, A_list, B_list


def sample_bipartite_dm(n, d, rng=None, tol=1e-9, max_attempts=100):
    """Random rho on C^2 ⊗ C^n with rho⪰0, tr=1, operator Schmidt rank 3, matrix rank d."""
    if n < 2:
        raise ValueError("Need n>=2 for operator Schmidt rank 3 to exist.")
    if not (3 <= d <= 2 * n):
        raise ValueError("Need 3<=d<=2n  (d=1 forces OSR∈{1,4}; d=2 cannot reach OSR 3).")

    if rng is None:
        rng = np.random.default_rng()
    for _ in range(max_attempts):
        C0 = (rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d))) / np.sqrt(2)
        C = np.vstack([C0, C0 @ haar(d, rng)])          # C = [[C0],[C0 U]] ∈ C^{2n×d}
        rho = C @ C.conj().T                        # ⪰0, equal diagonal blocks ⇒ M3=0 ⇒ OSR≤3
        u = haar(2, rng)                                 # rotate the absent operator-direction off Z
        rho = np.kron(u, np.eye(n)) @ rho @ np.kron(u, np.eye(n)).conj().T
        rho = 0.5 * (rho + rho.conj().T)
        rho /= np.trace(rho).real                   # tr = 1

        ev = np.linalg.eigvalsh(rho)
        if int(np.sum(ev > tol * ev[-1])) == d and osr(rho) == 3:
            return rho
    raise RuntimeError(f"resampling failed for (n={n}, d={d})")


########################## Contraction ##############################


def sample_contraction(n, rng = np.random.default_rng()):
    """Sample A in C^{n x n} with ||A||_2 = 1 (largest singular value = 1)."""
    U, V = haar(n, rng), haar(n, rng)
    s = rng.uniform(0, 1, n)
    s[0] = 1.0  # force ||A||_2 = 1
    return (U * s) @ V.conj().T  

def sample_rank1_defect_contraction(n, sigma=None, complex_matrix=True, rng=None):
    """
    Sample a contraction A in C^{n x n} (or R^{n x n}) whose defect
    I_n - A^* A has rank exactly 1.

    Parameters
    ----------
    n : int
        Matrix dimension.
    sigma : float or None
        The lone contracting singular value in [0, 1). If None, drawn
        uniformly from [0, 1). All other singular values are set to 1.
    complex_matrix : bool
        If True, sample complex A with Haar-unitary U, V.
        If False, sample real A with Haar-orthogonal U, V.
    rng : np.random.Generator or None
        Random generator (defaults to np.random.default_rng()).

    Returns
    -------
    A : ndarray, shape (n, n)
        Contraction with rank-1 defect I_n - A^* A.
    """
    if rng is None:
        rng = np.random.default_rng()

    if sigma is None:
        sigma = rng.uniform(0.0, 1.0)
    elif not (0.0 <= sigma < 1.0):
        raise ValueError("sigma must lie in [0, 1).")

    U, V = haar(n, rng), haar(n, rng)
    Sigma = np.ones(n)
    Sigma[0] = sigma
    A = (U * Sigma) @ V.conj().T  # U @ diag(Sigma) @ V^*
    return A


def sample_hard_contraction(n, rng = np.random.default_rng()):
    """Sample A in C^{n x n} with ||A||_2 = 1 (largest singular value = 1) and eigenvalues > cos(pi/(n+1)).
    Constructed thanks to Schur decomposition to enforce eigenvalues on the complex diagonal, and add small random upper-triangular noise.
    """
    U = haar(n, rng)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n))
    D = np.diag(np.cos(angles) + 1j * np.sin(angles))
    N = np.triu(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)), k=1) / 1e2
    norm = np.linalg.norm(U @ (D + N) @ U.conj().T, 2)
    return U @ (D + N) @ U.conj().T / norm




def sample_round_contraction(n, rng =np.random.default_rng()):
    """Sample A in C^{n x n} with ||A||_2 = 1 and w(A) close to a round disk."""
    while True:
        A = sample_contraction(n, rng)
        W = numerical_range(A)
        if np.max(np.abs(W)) > np.cos(np.pi / (n + 1)):
            return A