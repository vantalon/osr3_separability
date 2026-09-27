import numpy as np


I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], dtype=float)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=float)

# A-side basis: (I, X, Y, Z)
pauli_basis = [I2, X, Y, Z]
# B-side basis: (I, X, Z)  -- Y column vanishes by PT-invariance
tau = [I2, X, Z]

def GellMann(n):
    """
    Return the n^2-1 generalized Gell-Mann matrices.
    Normalization:
        Tr(GM[i] @ GM[j]) = n * delta_ij
    Thus [I_n] + GellMann(n) forms an orthogonal basis satisfying
        Tr(B[i] @ B[j]) = n * delta_ij.
    """
    mats = []
    for j in range(n):
        for k in range(j + 1, n):
            A = np.zeros((n, n))
            A[j, k] = A[k, j] = 1.0
            mats.append(A / np.sqrt(2 / n))
            A = np.zeros((n, n), dtype=complex)
            A[j, k] = -1j
            A[k, j] = 1j
            mats.append(A / np.sqrt(2 / n))
    for l in range(1, n):
        A = np.zeros((n, n))
        A[:l, :l] = np.eye(l)
        A[l, l] = -l
        mats.append(A * np.sqrt(2 / (l * (l + 1))) / np.sqrt(2 / n))
    return mats


####################### Phase clustering ############################



def order_phases(eigs):
    """Order eigenvalues by phase in [0, 2pi)."""
    phases = np.angle(eigs) % (2 * np.pi)
    order = np.argsort(phases)
    return eigs[order]

def phase_clustering( eigenvalues, phase_tol=1e-14):
    n = len(eigenvalues)
    # Phases in [0, 2*pi)
    theta = np.mod(np.angle(eigenvalues), 2 * np.pi)

    # Sort by phase.
    order = np.argsort(theta)
    theta = theta[order]

    # gap[j] = circular distance from theta[j] to theta[j+1]
    # with the last gap wrapping around 2*pi -> 0.
    gaps = np.diff(
        np.r_[theta, theta[0] + 2 * np.pi]
    )

    # A gap larger than phase_tol separates two eigenspaces.
    cuts = np.where(gaps > phase_tol)[0]
    if len(cuts) == 0:
        # All phases belong to one cluster.
        clusters = [np.arange(n)]

    else:
        # Start immediately after one of the large gaps.
        start = (cuts[0] + 1) % n
        circular_order = np.roll(np.arange(n), -start)
        clusters = []
        current = [circular_order[0]]
        for a, b in zip(circular_order[:-1],
                        circular_order[1:]):
            # Forward circular phase distance
            dtheta = (theta[b] - theta[a]) % (2 * np.pi)
            if dtheta <= phase_tol:
                current.append(b)
            else:
                clusters.append(order[np.array(current)])
                current = [b]
        clusters.append(order[np.array(current)])
    return clusters


####################### Correlation matrix ##########################



def correlation_matrix(rho, n):
    """
    Compute the  correlation matrix without the idenqtity terms, i.e.:
        T_ij = Tr[rho (sigma_i \otimes sigma_j)]
    """
    Paulis_A  =  pauli_basis[1:]  # skip identity 
    GellMann_B = GellMann(n)
    T = np.zeros((3, n**2 - 1), dtype=float)
    for i, sA in enumerate(Paulis_A[1:]):
        for j, sB in enumerate(GellMann_B):
            op = np.kron(sA, sB)
            T[i, j] = np.real(np.trace(rho @ op))
    return T

def build_T_correlation(rho, n, tol=1e-14):
    """Return T[i,j] = Tr(rho * (A_i ⊗ B_j))."""

    basis_A = pauli_basis
    basis_B = [np.eye(n)] + GellMann(n)
    if rho.shape != (2*n, 2*n):
        raise ValueError(f"rho must have shape {(2*n, 2*n)}")
    T = np.zeros((4, n**2), dtype=float)
    for i, A in enumerate(basis_A):
        for j, B in enumerate(basis_B):
            value = np.trace(rho @ np.kron(A, B))
            T[i, j] = np.real_if_close(value)
    T[np.abs(T) < tol] = 0.0
    return T

def build_rho_from_T_correlation(T):
    """Reconstruct rho from T[i,j] = Tr(rho * (A_i ⊗ B_j))."""

    if T.ndim != 2 or T.shape[0] != 4:
        raise ValueError("T must have shape (4, n**2)")
    n = int(round(np.sqrt(T.shape[1])))

    basis_A = pauli_basis
    basis_B = [np.eye(n)] + GellMann(n)
    rho = np.zeros((2*n, 2*n), dtype=complex)
    for i, A in enumerate(basis_A):
        norm_A = np.trace(A.conj().T @ A).real
        for j, B in enumerate(basis_B):
            norm_B = np.trace(B.conj().T @ B).real
            rho += T[i, j] / (norm_A * norm_B) * np.kron(A, B)

    return np.real_if_close(rho)

def build_C_from_correlation(T):
    """Given T (n^2 x 4), return C1, C3 (n x n) such that T = [Tr(rho sA⊗sB)]_{sA,sB}."""
    n = int(np.sqrt(T.shape[1]))
    C1 = np.zeros((n, n), dtype=complex)
    C3 = np.zeros((n, n), dtype=complex)
    GellMann_B = [np.eye(n)] + GellMann(n)
    for i, sB in enumerate(GellMann_B):
        C1 += T[1, i] * sB
        C3 += T[3, i] * sB
    return np.real_if_close(C1), np.real_if_close(C3)

def build_rho_from_T_contraction(T):
    C1 = (T + T.conj().T) / 2
    C3 = (T - T.conj().T) / (2j)

    rho = np.kron(I2, np.eye(C1.shape[0])) + np.kron(X, C1) + np.kron(Z, C3)
    return np.real_if_close(rho) / (2 * C1.shape[0])




# Reconstruct rho from found decomposition
def bloch_to_dmA(avec):
    return 0.5 * (I2 + avec[0] * X + avec[1] * Y + avec[2] * Z)


def bloch_to_dmB(bvec):
    return 0.5 * (I2 + bvec[0] * X + bvec[1] * Z)



############################# Basis #################################


def hs_inner(A, B):
    """Hilbert-Schmidt inner product Tr(A† B), real part (both Hermitian)."""
    return np.real(np.trace(A.conj().T @ B))

def gs_orthonormalize(basis):
    """Gram-Schmidt over a list of Hermitian matrices."""
    ortho = []
    for v in basis:
        w = v.copy()
        for u in ortho:
            w = w - hs_inner(u, w) * u
        n = np.sqrt(hs_inner(w, w))
        if n > 1e-12:
            ortho.append(w / n)
    return ortho



def is_psd(M, tol=1e-9):
    """Check if M is positive semidefinite."""
    eigvals = np.linalg.eigvalsh(M)
    return bool(np.all(eigvals >= -tol))


def reconstruct(lambdas, Es, Fs, m):
    """Reconstruct A = sum lambda_i E_i ⊗ F_i with E_i in M2, F_i in Mm."""
    dim = 2 * m
    A = np.zeros((dim, dim), dtype=complex)
    for lam, E, F in zip(lambdas, Es, Fs):
        A += lam * np.kron(E, F)
    return A



def realignement_matrix(rho, n):
    """Realignment of a 4x4 matrix rho.
    rank R = OSR(rho) #TODO: check if this is correct 
    """
    return rho.reshape(2, n, 2, n).transpose(0, 2, 1, 3).reshape(4, n * n)

def osr(rho, tol=1e-15):  # rank of the realignment R(rho): a 4 x n^2 matrix
    n = rho.shape[0] // 2
    R = realignement_matrix(rho, n)
    s = np.linalg.svd(R, compute_uv=False)
    return int(np.sum(s > tol * s[0]))

def partial_trace_A(rho, m):
    """Partial trace over the first subsystem (A) of a bipartite state rho in M2⊗Mm."""
    rho = rho.reshape(2, m, 2, m)
    return np.trace(rho, axis1=0, axis2=2).reshape(m, m)

def partial_trace_B(rho, m):
    """Partial trace over the second subsystem (B) of a bipartite state rho in M2⊗Mm."""
    rho = rho.reshape(2, m, 2, m)
    return np.trace(rho, axis1=1, axis2=3).reshape(2, 2)


def partial_transpose_A(rho, m):
    """Partial transpose over the first subsystem (A) of a bipartite state rho in M2⊗Mm."""
    dim = 2 * m
    rho = rho.reshape(2, m, 2, m)
    return rho.transpose(2, 1, 0, 3).reshape(dim, dim)

def partial_transpose_B(rho, m):
    """Partial transpose over the second subsystem (B) of a bipartite state rho in M2⊗Mm."""
    dim = 2 * m
    rho = rho.reshape(2, m, 2, m)
    return rho.transpose(0, 3, 2, 1).reshape(dim, dim)


########################## Numerical Range ##########################


def numerical_range(A, K=400):
    """
    Compute the boundary of W(A) exactly via the supporting-line method
    (Johnson's algorithm). For each angle theta, the support of W(A) in
    direction e^{i theta} is the largest eigenvalue of the Hermitian part
    of e^{-i theta} A; the corresponding eigenvector x gives x* A x on dW(A).
    """
    pts = []
    for theta in np.linspace(0, 2 * np.pi, K, endpoint=False):
        B = np.exp(-1j * theta) * A
        H = (B + B.conj().T) / 2  # Hermitian part
        w, V = np.linalg.eigh(H)  # ascending eigenvalues
        x = V[:, -1]  # eigenvector of largest eigenvalue
        pts.append((x.conj() @ A @ x))
    return np.array(pts)