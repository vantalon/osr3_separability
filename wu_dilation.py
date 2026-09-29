import numpy as np
from numpy.linalg import svd, eigh, inv

dag = lambda M: M.conj().T


def _nullspace(M, tol):
    """Orthonormal basis (columns) of ker M."""
    if M.shape[0] == 0:
        return np.eye(M.shape[1], dtype=complex)
    U, s, Vh = svd(M)
    r = int((s > tol).sum())
    return Vh[r:].conj().T


def canonical_split(T, tol=1e-14):
    """
    Return (Qu, Qa, U, A): orthonormal bases of the maximal unitary reducing
    subspace H_u and its complement H_a, and the blocks U = T|H_u, A = T|H_a.

    H_u = {x : ||T^k x|| = ||x|| = ||T*^k x||  for all k}.
    Since ker C^{1/2} = ker C for C >= 0, we may use the defect *operators*
    C = I - T*T and C* = I - TT* directly and avoid the square roots (whose
    numerical error is the square root of that of C, i.e. ~1e-8).
    """
    T = np.asarray(T, dtype=complex)
    n = T.shape[0]
    C = np.eye(n) - dag(T) @ T
    Cs = np.eye(n) - T @ dag(T)

    blocks, P, Ps = [], np.eye(n, dtype=complex), np.eye(n, dtype=complex)
    for _ in range(n):
        blocks += [C @ P, Cs @ Ps]
        P, Ps = T @ P, dag(T) @ Ps
    M = np.vstack(blocks)
    Qu = _nullspace(M, tol * max(1.0, np.linalg.norm(M, 2)))
    Qa = _nullspace(dag(Qu), 1e-14) if Qu.shape[1] else np.eye(n, dtype=complex)

    U = dag(Qu) @ T @ Qu if Qu.shape[1] else np.zeros((0, 0), complex)
    A = dag(Qa) @ T @ Qa if Qa.shape[1] else np.zeros((0, 0), complex)
    return Qu, Qa, U, A


def distinct_eigs(U, tol=1e-7):
    """Distinct eigenvalues of the unitary block and their multiplicities."""
    if U.shape[0] == 0:
        return np.array([], dtype=complex), []
    ev = np.linalg.eigvals(U)
    alphas, mults = [], []
    for z in ev:
        for i, a in enumerate(alphas):
            if abs(z - a) < tol:
                mults[i] += 1
                break
        else:
            alphas.append(z)
            mults.append(1)
    return np.array(alphas, dtype=complex), mults


def minimal_polynomial_roots(A, tol=1e-8, mode="minimal"):
    """
    Roots (with multiplicity) of the minimal polynomial of A (mode="minimal"),
    or of the characteristic polynomial (mode="characteristic").

    "minimal" finds the first d with I, A, ..., A^d linearly dependent and
    falls back to "characteristic" if the resulting phi does not annihilate A
    (Cayley-Hamilton makes the characteristic choice always admissible; it only
    makes the model space larger).
    """
    if A.shape[0] == 0:
        return np.array([], dtype=complex)
    n = A.shape[0]
    if mode == "characteristic":
        return np.linalg.eigvals(A)

    cols, P = [], np.eye(n, dtype=complex)
    for d in range(n + 1):
        cols.append(P.reshape(-1))
        M = np.array(cols).T
        s = svd(M, compute_uv=False)
        if s[-1] <= tol * s[0]:
            c = _nullspace(M, tol * s[0])[:, 0]
            k = max(i for i in range(len(c)) if abs(c[i]) > 1e-8 * np.max(np.abs(c)))
            lam = np.roots((c[: k + 1] / c[k])[::-1])
            if np.max(np.abs(lam)) < 1 - 1e-10 and _phi_norm(A, lam) < 1e-7:
                return lam
            break
        P = A @ P
    return np.linalg.eigvals(A)


def _phi_norm(A, lam):
    """||phi(A)|| for phi = prod b_{lambda_j}; must be ~0."""
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    P = I.copy()
    for l in lam:
        R = I - np.conj(l) * A
        if np.linalg.cond(R) > 1e12:
            return np.inf
        P = P @ (l * I - A) @ inv(R)
    return np.linalg.norm(P, 2)


def S_of_Lambda(lam):
    """Livsic triangular form S(Lambda); matrix of the compressed shift."""
    lam = np.asarray(lam, dtype=complex)
    m = len(lam)
    d = np.sqrt(np.clip(1 - np.abs(lam) ** 2, 0, None))
    S = np.zeros((m, m), dtype=complex)
    for j in range(m):
        S[j, j] = lam[j]
        for k in range(j + 1, m):
            prod = 1.0 + 0j
            for l in range(j + 1, k):
                prod *= -np.conj(lam[l])
            S[j, k] = d[j] * prod * d[k]
    return S


def _malmquist_walsh_of_A(A, lam):
    """
    e_j(A) for the (suffix-ordered, sign-gauged) Malmquist-Walsh basis

        e_j(z) = (-1)^j  d_j (1 - conj(l_j) z)^{-1} prod_{l>j} b_{l_l}(z),

    which is the basis in which the compressed shift has the upper
    triangular Livsic form S(Lambda) of the statement.
    """
    n, m = A.shape[0], len(lam)
    I = np.eye(n, dtype=complex)
    d = np.sqrt(np.clip(1 - np.abs(lam) ** 2, 0, None))
    suf = [None] * (m + 1)
    suf[m] = I.copy()
    for j in range(m - 1, -1, -1):
        lj = lam[j]
        suf[j] = suf[j + 1] @ (lj * I - A) @ inv(I - np.conj(lj) * A)
    return [
        ((-1.0) ** j) * d[j] * inv(I - np.conj(lam[j]) * A) @ suf[j + 1]
        for j in range(m)
    ]


def isometry_A(A, lam, tol=1e-9):
    """
    W : H_a -> (+)^{r_A} K_phi,   (Wx)(z) = D_{A*}(I - z A*)^{-1} x.
    Rows are indexed by rho = i*m + j  (i = copy, j = Malmquist-Walsh index).
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0), complex), 0
    Def = np.eye(n) - A @ dag(A)
    w, Q = eigh((Def + dag(Def)) / 2)
    idx = [i for i in range(n) if w[i].real > tol]
    rA = len(idx)
    F = Q[:, idx] * np.sqrt(np.clip(w[idx].real, 0, None))  # columns D_{A*} f_i

    E = _malmquist_walsh_of_A(A, lam)
    m = len(lam)
    W = np.zeros((rA * m, n), dtype=complex)
    for i in range(rA):
        for j in range(m):
            W[i * m + j, :] = dag(E[j] @ F[:, i])
    return W, rA


def isometry_U(U, alphas, mults, N, tol=1e-7):
    """V_U : H_u -> (+)^N C^p,  u_{k,i} |-> e_k in copy i."""
    q, p = U.shape[0], len(alphas)
    if q == 0:
        return np.zeros((N * p, 0), complex)
    ev, Vec = np.linalg.eig(U)
    VU = np.zeros((N * p, q), dtype=complex)
    used = {k: 0 for k in range(p)}
    cols = []
    for t in range(q):
        k = int(np.argmin([abs(ev[t] - a) for a in alphas]))
        cols.append((k, used[k]))
        used[k] += 1
    # orthonormalise eigenvectors inside each eigenspace
    X = Vec.copy()
    for k in range(p):
        ids = [t for t in range(q) if cols[t][0] == k]
        if ids:
            Qk, _ = np.linalg.qr(X[:, ids])
            X[:, ids] = Qk
    Xinv = inv(X)  # coordinates of H_u basis in eigenbasis
    for t in range(q):
        k, i = cols[t]
        VU[i * p + k, :] += Xinv[t, :]
    return VU


def dilation(T, tol=1e-14, verbose=False, mode="minimal"):
    """Return a dict with V, V_U, V_A, B and all intermediate data."""
    T = np.asarray(T, dtype=complex)
    n = T.shape[0]
    if np.max(svd(T, compute_uv=False)) > 1 + 1e-8:
        raise ValueError("T is not a contraction")

    Qu, Qa, U, A = canonical_split(T, tol)
    us, mults = distinct_eigs(U)
    p = len(us)
    rU = max(mults) if mults else 0

    lam = minimal_polynomial_roots(A, tol, mode)
    m = len(lam)
    W, rA = isometry_A(A, lam, 1e-9)
    N = max(rU, rA, 1)
    if verbose:
        print("alphas   =", np.round(us, 4), " mults =", mults)
        print("Lambda   =", np.round(lam, 4))
        print("r_U =", rU, " r_A =", rA, " N =", N, " p =", p, " m =", m)

    S = S_of_Lambda(lam)
    Dal = np.diag(us) if p else np.zeros((0, 0), complex)
    block = np.zeros((p + m, p + m), dtype=complex)
    block[:p, :p], block[p:, p:] = Dal, S
    B = np.kron(np.eye(N), block)

    VU = isometry_U(U, us, mults, N)
    # embed into (+)^N (C^p (+) K_phi)
    V = np.zeros((N * (p + m), n), dtype=complex)
    for i in range(N):
        if p:
            V[i * (p + m) : i * (p + m) + p, :] += VU[i * p : (i + 1) * p, :] @ dag(Qu)
        if m and i < rA:
            V[i * (p + m) + p : (i + 1) * (p + m), :] += W[
                i * m : (i + 1) * m, :
            ] @ dag(Qa)

    Vu_full = np.zeros_like(V)
    Va_full = np.zeros_like(V)
    for i in range(N):
        if p:
            Vu_full[i * (p + m) : i * (p + m) + p, :] = VU[
                i * p : (i + 1) * p, :
            ] @ dag(Qu)
        if m and i < rA:
            Va_full[i * (p + m) + p : (i + 1) * (p + m), :] = W[
                i * m : (i + 1) * m, :
            ] @ dag(Qa)

    return dict(
        V=V,
        V_U=Vu_full,
        V_A=Va_full,
        B=B,
        W=W,
        Dal=Dal,
        S=S,
        U=U,
        A=A,
        mults=mults,
        Lambda=lam,
        r_U=rU,
        r_A=rA,
        N=N,
        p=p,
        m=m,
    )


def check(res, T, kmax=8):
    """Max errors: isometry of V, dilation identity, intertwining."""
    V, B = res["V"], res["B"]
    n = T.shape[0]
    e_iso = np.max(np.abs(dag(V) @ V - np.eye(n)))
    e_dil = 0.0
    Bk, Tk = np.eye(B.shape[0], dtype=complex), np.eye(n, dtype=complex)
    for _ in range(kmax + 1):
        e_dil = max(e_dil, np.max(np.abs(dag(V) @ Bk @ V - Tk)))
        Bk, Tk = B @ Bk, T @ Tk
    W, S, A, rA = res["W"], res["S"], res["A"], res["r_A"]
    if W.size:
        Sb = np.kron(np.eye(rA), S)
        e_int = np.max(np.abs(W @ dag(A) - dag(Sb) @ W))
    else:
        e_int = 0.0
    return dict(isometry=e_iso, dilation=e_dil, intertwining=e_int)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    # T = U (+) A  in a rotated basis
    n = 5
    Ud = np.diag([1.0 + 0j, -1j])
    Ab = np.array(
        [[0.42, 0.33, 0.11], [0.08, 0.52, 0.19], [0.21, 0.13, 0.49]], dtype=complex
    )
    assert len(Ud) + len(Ab) == n
    T0 = np.zeros((n, n), dtype=complex)
    T0[: len(Ud), : len(Ud)] = Ud
    T0[len(Ud) :, len(Ud) :] = Ab
    rng = np.random.default_rng(0)
    Q, R = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    Q = Q @ np.diag(np.diag(R) / abs(np.diag(R)))
    T = Q @ T0 @ dag(Q)

    res = dilation(T)
    print("T = U \oplus A")
    print(
        "eigenvalues of U :", np.round(np.diag(res["Dal"]), 4), " mults =", res["mults"]
    )
    print("eigenvalues of A :", np.round(res["Lambda"], 4))
    print(
        "r_U =",
        res["r_U"],
        " r_A =",
        res["r_A"],
        " N =",
        res["N"],
        " p =",
        res["p"],
        " m =",
        res["m"],
    )
    print("\nchecks:", {k: f"{v:.2e}" for k, v in check(res, T).items()})
