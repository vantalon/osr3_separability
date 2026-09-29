import numpy as np
from sampler import sample_bipartite_dm
from filtering import *
from utils import *
from dilation import (
    minimal_unitary_dilation,
    extract_decomposition,
    unitary_dilation_with_shift,
)

seed = np.random.randint(0, 1000)
# seed = 523
print("Random seed:", seed)
rng = np.random.default_rng(seed)

n = 4

pure_decomposition = True
rho = sample_bipartite_dm(n, d=4, rng=rng)


rho_PT, V1 = cariello_decomposition(rho, n)

assert np.allclose(
    rho_PT, partial_transpose_A(rho_PT, n)
), "rho is not invariant under partial transpose"

rho_full_rk, U1 = full_rank_B(rho_PT, n)
m = rho_full_rk.shape[0] // 2
assert (
    np.linalg.matrix_rank(partial_trace_A(rho_full_rk, m)) == m
), "rho_full_rk is not full rank"
print("New dimension to be full rank:", m)

rho_mm, inv_tauB = maximally_mixed_B_marginals(rho_full_rk, m)

C1, C3 = build_C_from_correlation(build_T_correlation(rho_mm, m))
T = C1 + 1j * C3


if pure_decomposition:
    U, J, d = minimal_unitary_dilation(T, tol=1e-12)
    print("Defect index:", d)
else:
    U, J = unitary_dilation_with_shift(T, tol=1e-12)


p, alpha, beta = extract_decomposition(U, J, m, T)
print("sum p_k = ", np.sum(p))
print("Length of decomposition : ", len(p))
reconstructed_rho = sum(
    p_k * np.kron(a_k, b_k) for p_k, a_k, b_k in zip(p, alpha, beta)
)
print(
    "Check reconstruction of normal form:\n ",
    np.allclose(rho_mm, reconstructed_rho),
    np.max(np.abs(rho_mm - reconstructed_rho)),
)


p, alpha, beta = inverse_all_filterings(p, alpha, beta, V1, U1, inv_tauB, n)
reconstructed_rho = sum(
    p_k * np.kron(a_k, b_k) for p_k, a_k, b_k in zip(p, alpha, beta)
)
print(
    "Check reconstruction after inverse filtering:\n ",
    np.allclose(rho, reconstructed_rho),
    np.max(np.abs(rho - reconstructed_rho)),
)
