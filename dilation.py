import numpy as np
from scipy.linalg import schur
from wu_dilation import dilation
from utils import  *


def psd_sqrt_and_range(A, tol=1e-12):
        """
        Compute sqrt(A) and an orthonormal basis for ran(sqrt(A)),
        assuming A is Hermitian positive semidefinite.

        Parameters
        ----------
        A : ndarray, shape (n, n)
            Hermitian positive semidefinite matrix.
        tol : float
            Numerical tolerance for determining positive eigenvalues.
            Default is 1e-12.

        Returns
        -------
        sqrtA : ndarray, shape (n, n)
            Square root of A.
        V : ndarray, shape (n, k)
            Orthonormal basis for ran(sqrt(A)), where k is the rank of sqrt(A).
        """
        A = (A + A.conj().T) / 2

        eigvals, eigvecs = np.linalg.eigh(A)

        # Remove tiny negative eigenvalues caused by roundoff.
        if np.min(eigvals) < -tol:
            raise ValueError(f"Defect matrix is not positive semidefinite. {np.min(eigvals)}")

        eigvals = np.maximum(eigvals, 0.0)

        sqrtA = (eigvecs * np.sqrt(eigvals)) @ eigvecs.conj().T

        # ran(sqrt(A)) corresponds to the strictly positive eigenvalues.
        mask = eigvals > tol
        V = eigvecs[:, mask]

        return sqrtA, V

def minimal_unitary_dilation(T, theta = 0,  tol=1e-14):
    """
    Construct the (n+d)-dimensional unitary dilation
        U = [[T,                 D_{T*} V_*],
             [V^* D_T,    -V^* T^* V_*]]

    of a contraction T in C^{n x n}, where D_T    = sqrt(I - T^* T), D_{T*} = sqrt(I - T T^*), and d = rank(D_T) = rank(D_{T*}).

    Parameters
    ----------
    T : ndarray, shape (n, n)
        A contraction: ||T||_2 <= 1.
    tol : float
        Numerical tolerance used to determine the defect rank.

    Returns
    -------
    U : ndarray, shape (n+d, n+d)
        Unitary dilation of T.
    J : ndarray, shape (n+d, n)
        Canonical embedding J = [I_n; 0].
    d : int
        Defect rank.
    """

    T = np.asarray(T, dtype=complex)

    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("T must be a square matrix.")

    n = T.shape[0]
    I = np.eye(n, dtype=complex)

    # Check that T is a contraction.
    norm_T = np.linalg.norm(T, ord=2)
    if norm_T > 1 + tol:
        raise ValueError(
            f"T is not a contraction: ||T||_2 = {norm_T} > 1."
        )

    # Defect operators
    D_T, V = psd_sqrt_and_range(I - T.conj().T @ T, tol=tol)
    D_Tstar, V_star = psd_sqrt_and_range(I - T @ T.conj().T, tol=tol)

    d = V.shape[1]
    if V_star.shape[1] != d:
        raise RuntimeError(
            "Numerically detected defect ranks do not agree. "
            "Try adjusting tol."
        )

    # Blocks of U
    A = T
    B = D_Tstar @ V_star
    C = V.conj().T @ D_T
    D = -V.conj().T @ T.conj().T @ V_star
    W = np.exp(1j * theta) * np.eye(d) # could be any unitary of size d x d, but we choose the simplest one.

    U = np.block([
        [A, B],
        [W @ C, W @ D]
    ])

    W = np.exp(1j * theta) * np.eye(d)

    # Canonical embedding J : C^n -> C^(n+d)
    J = np.vstack([
        np.eye(n, dtype=complex),
        np.zeros((d, n), dtype=complex)
    ])
    return U, J, d




def unitary_dilation_with_shift(T, tol=1e-14, verbose=False):
    """
    Construct a unitary dilation of a contraction T in C^{n x n}, using the dilation of the compressed shift.
    Parameters
    ----------
    T : ndarray, shape (n, n)
        A contraction: ||T||_2 <= 1.
    tol : float
        Numerical tolerance used to determine the defect rank. (1e-14 is usually sufficient.)
    Returns
    -------
    U : ndarray, shape (N, N)
        Unitary dilation of T, where N = n + p + m + 1.
    J : ndarray, shape (N, n)
        Canonical embedding J = [I_n; 0]."""
    res = dilation(T,tol=tol,verbose=verbose)
    Dal = res['Dal']
    p = res['p']
    m = res['m']

    U_S, J_S, d = minimal_unitary_dilation(res['S'], tol=tol)
    assert d == 1 
    block = np.zeros((p + m + 1, p + m+ 1), dtype=complex)
    block[:p, :p], block[p:, p:] = Dal, U_S
    U = np.kron(np.eye(res['N']), block)  # Extend to full dimension
    block = np.zeros((p + m + 1, p + m), dtype=complex)
    block[:p, :p], block[p:, p:] = np.eye(p), J_S
    J = np.kron(np.eye(res['N']), block) @ res['V']  # Extend to full dimension
    return U, J



def spectral_projections(U, phase_tol=1e-14):
    """
    Spectral projectors of a unitary matrix using phase clustering.
    U = sum_k lambda_k E_k

    Parameters
    ----------
    U : (n, n) complex ndarray
        Unitary matrix.

    phase_tol : float
        Eigenphases separated by <= phase_tol radians are considered
        to belong to the same eigenspace.

    Returns
    -------
    eigenvalues : list of complex
        Representative eigenvalue lambda_k for each eigenspace.

    projectors : list of ndarray
        Orthogonal spectral projectors E_k.
    """
    U = np.asarray(U, dtype=np.complex128)

    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix.")

    # Complex Schur decomposition: U = Q T Q^*
    # Q is unitary. Since U is normal, T is diagonal up to floating-point errors.
    T, Q = schur(U, output="complex")
    z = np.diag(T)

    # Normalize eigenvalues onto the unit circle.
    if np.any(np.abs(z) == 0):
        raise ValueError("Unexpected zero eigenvalue for a unitary matrix.")
    z = z / np.abs(z)

    clusters = phase_clustering(z, phase_tol=phase_tol)

    eigenvalues = []
    projectors = []

    for idx in clusters:
        V = Q[:, idx]
        # Orthogonal projector onto the eigenspace.
        E = V @ V.conj().T
        # Enforce Hermiticity against roundoff.
        E = (E + E.conj().T) / 2
        # Circular mean of the eigenvalues/phases.
        lam = np.mean(z[idx])
        lam /= abs(lam)
        eigenvalues.append(lam)
        projectors.append(E)

    assert len(eigenvalues) == len(projectors), "Mismatch in number of eigenvalues and projectors."
    assert np.allclose(U, np.sum([v * E for v, E in zip(eigenvalues, projectors)], axis=0)), "Sum of eigenvalues times projectors does not equal U."
    return eigenvalues, projectors



def extract_decomposition(U, J, n, T,verbose=False):
    """
    Given T = U J U^†, extract the decomposition of rho into a sum of density matrices. 

    Parameters:
    - U: The unitary matrix from the Schur decomposition of T.
    - J: The upper triangular matrix from the Schur decomposition of T.
    - n: The dimension of the system.

    Returns:
    - p_k: List of probabilities corresponding to each density matrix.
    - alpha_k: List of density matrices on the A-side.
    - beta_k: List of density matrices on the B-side.
    """

    lambdas, E_k= spectral_projections(U)
    if verbose:
        print("Eigenvalues of U:", np.linalg.eigvals(U))
        print("Eigenvalues of U grouped:", np.real_if_close(lambdas))

    L_k = [J.conj().T @ E @ J for E in E_k]
    p_k = [np.trace(L).real/n for L in L_k] 
    beta_k = [L / np.trace(L).real if p > 1e-15 else L for p, L in zip(p_k, L_k)]
    alpha_k = [1/2*(I2 + v.real *X + v.imag *Z) for v in lambdas]
    return p_k, alpha_k, beta_k



if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    # T = U (+) A  in a rotated basis:  U = diag(1,1,i),  A = 2x2 Jordan block
    n = 6
    Ud = np.diag([np.exp(1j), np.exp(1j), -1j])
    Ab = np.array([[0.41, 0.33, 0.11], [0.08, 0.52, 0.19], [0.21, 0.13, 0.49]], dtype=complex)
    assert len(Ud) + len(Ab) == n
    T0 = np.zeros((n, n), dtype=complex)
    T0[:len(Ud), :len(Ud)] = Ud
    T0[len(Ud):, len(Ud):] = Ab
    rng = np.random.default_rng(0)
    Q, R = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    Q = Q @ np.diag(np.diag(R) / abs(np.diag(R)))
    T = Q @ T0 @ Q.conj().T
    rho = build_rho_from_T_contraction(T)

    U, J = unitary_dilation_with_shift(T, verbose=True)
    assert np.allclose(T, J.conj().T @ U @ J), "Unitary dilation failed"

    p, alpha, beta = extract_decomposition(U, J, n, T,True)
    print( "sum p_k = ", np.sum(p))
    reconstructed_rho = sum(p_k * np.kron(a_k, b_k) for p_k, a_k, b_k in zip(p, alpha, beta))
    print("Length of decomposition: ", len(p))
    print("Check reconstruction: ",np.allclose(rho, reconstructed_rho), np.max(np.abs(rho - reconstructed_rho)))
