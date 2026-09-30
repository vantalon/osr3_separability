import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from dilation import unitary_dilation_with_shift, minimal_unitary_dilation
from wu_dilation import dilation
from sampler import *
from utils import *
from filtering import *


def circle_handle(width, height, xdescent, ydescent, **kwargs):
    r = min(width, height) / 2
    return mpatches.Circle((width / 2 - xdescent, height / 2 - ydescent), r)


def figure_numerical_range(ax, T):
    n = len(T)
    W = numerical_range(T)
    center = np.trace(T) / n
    eig = np.linalg.eigvals(T)

    ax.fill(W.real, W.imag, alpha=0.2, label="W(T)")
    ax.plot(np.append(W.real, W.real[0]), np.append(W.imag, W.imag[0]), lw=1)
    ax.plot(center.real, center.imag, "x", color="k", ms=8, label=r"$\it{a}$")
    ax.scatter(eig.real, eig.imag, c="navy", zorder=3, label=r"spec$(T)$")

    circ = plt.Circle(
        (0, 0), 1, fill=False, ls="--", color="gray", label=r"$\partial\mathbb{D}$"
    )
    ax.add_patch(circ)

    ax.set_aspect("equal")
    ax.axhline(0, lw=0.3)
    ax.axvline(0, lw=0.3)
    # ax.set_title(f"Numerical range, ||A||={np.linalg.norm(A,2):.3f}, n = {n}, seed={seed}")
    ax.set_title(f"n = {n}")
    ax.legend(
        fontsize=12,
        loc="upper left",
        handler_map={circ: HandlerPatch(patch_func=circle_handle)},
        handlelength=1.2,
        handleheight=1.2,
    )
    ax.set_xlabel("X")
    ax.xaxis.set_ticks(np.linspace(-1, 1, 5))
    ax.set_ylabel("Z")
    ax.yaxis.set_ticks(np.linspace(-1, 1, 5))


def figure_pure(ax, T, i):
    n = len(T)
    W = numerical_range(T)
    center = np.trace(T) / n
    eig = np.linalg.eigvals(T)

    U, _, d = minimal_unitary_dilation(T, tol=1e-13)
    eig_U = order_phases(np.linalg.eigvals(U))

    ax.fill(W.real, W.imag, alpha=0.2, label="W(T)")
    ax.plot(np.append(W.real, W.real[0]), np.append(W.imag, W.imag[0]), lw=1)
    ax.plot(center.real, center.imag, "x", color="k", ms=8, label=r"$\it{a}$")
    ax.scatter(eig.real, eig.imag, c="navy", zorder=3, label=r"spec$(T)$")
    ax.plot(eig_U.real, eig_U.imag, "o", color="g", ms=6, label=rf"spec$(U)$")
    ax.plot(
        np.append(eig_U.real, eig_U.real[0]),
        np.append(eig_U.imag, eig_U.imag[0]),
        "-",
        color="g",
        ms=6,
    )
    circ = plt.Circle(
        (0, 0), 1, fill=False, ls="--", color="gray", label=r"$\partial\mathbb{D}$"
    )
    ax.add_patch(circ)

    ax.set_aspect("equal")
    ax.axhline(0, lw=0.3)
    ax.axvline(0, lw=0.3)
    # ax.set_title(f"Numerical range, ||A||={np.linalg.norm(A,2):.3f}, n = {n}, seed={seed}")
    ax.set_title(rf"n = {n}, rank($\rho$) = {n+d}", fontsize=14)
    if i == 0:
        ax.legend(
            loc="upper left",
            handler_map={circ: HandlerPatch(patch_func=circle_handle)},
            handlelength=1.2,
            handleheight=1.2,
            fontsize=12,
        )
    ax.set_xlabel("X")
    ax.xaxis.set_ticks(np.linspace(-1, 1, 5))
    ax.set_ylabel("Z")
    ax.yaxis.set_ticks(np.linspace(-1, 1, 5))


def figure_mixed(ax, T, i):
    n = len(T)
    W = numerical_range(T)
    center = np.trace(T) / n
    eig = np.linalg.eigvals(T)

    res = dilation(T)
    B = res["B"]

    W_B = numerical_range(B)

    U, _ = unitary_dilation_with_shift(T, tol=1e-13)
    eig_U_tot = order_phases(np.linalg.eigvals(U))
    cluster = phase_clustering(eig_U_tot)
    eig_U = np.array([np.mean(eig_U_tot[idx]) for idx in cluster])

    ax.fill(W.real, W.imag, alpha=0.2, label="W(T)")
    ax.plot(np.append(W.real, W.real[0]), np.append(W.imag, W.imag[0]), lw=1)
    ax.plot(center.real, center.imag, "x", color="k", ms=8, label=r"$\it{a}$")
    ax.scatter(eig.real, eig.imag, c="navy", zorder=3, label=r"spec$(T)$")
    ax.fill(W_B.real, W_B.imag, alpha=0.1, zorder=3, color="indigo", label="W(B)")
    ax.plot(
        np.append(W_B.real, W_B.real[0]),
        np.append(W_B.imag, W_B.imag[0]),
        lw=1,
        color="indigo",
    )
    ax.plot(
        np.append(eig_U.real, eig_U.real[0]),
        np.append(eig_U.imag, eig_U.imag[0]),
        "o-",
        color="g",
        ms=6,
        label=rf"spec $(U)$",
    )

    circ = plt.Circle(
        (0, 0), 1, fill=False, ls="--", color="gray", label=r"$\partial\mathbb{D}$"
    )
    ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.axhline(0, lw=0.3)
    ax.axvline(0, lw=0.3)
    ax.set_title(f"n = {n}", fontsize=14)
    if i == 0:
        ax.legend(
            loc="upper left",
            handler_map={circ: HandlerPatch(patch_func=circle_handle)},
            handlelength=1.2,
            handleheight=1.2,
            fontsize=12,
        )
    ax.set_xlabel("X")
    ax.xaxis.set_ticks(np.linspace(-1, 1, 5))
    ax.set_ylabel("Z")
    ax.yaxis.set_ticks(np.linspace(-1, 1, 5))


def figure_UA(ax, T):
    ax0 = ax[0]
    ax1 = ax[1]
    ax2 = ax[2]
    n = len(T)
    center = np.trace(T) / n

    res = dilation(T)
    U, A, B, S, Dal = res["U"], res["A"], res["B"], res["S"], res["Dal"]
    U_S, J_S, d = minimal_unitary_dilation(S)

    W_T = numerical_range(T)
    eig_T = np.linalg.eigvals(T)
    for ax in [ax0, ax1, ax2]:
        ax.plot(center.real, center.imag, "x", color="k", ms=8, label=r"$\it{a}$")
        ax.scatter(eig_T.real, eig_T.imag, zorder=3, color="navy", label=r"spec$(T)$")
        ax.fill(W_T.real, W_T.imag, alpha=0.2, zorder=3, label="W(T)")
        ax.plot(
            np.append(W_T.real, W_T.real[0]), np.append(W_T.imag, W_T.imag[0]), lw=1
        )

    names = [
        "U",
        "A",
        f"S($\Lambda$)",
        "D",
        "V",
        "B",
    ]
    axes = [ax0, ax0, ax1, ax1, ax1, ax2]
    colors = ["deepskyblue", "crimson", "orange", "deepskyblue", "lightcoral", "purple"]
    for i, C in enumerate([U, A, S, Dal, U_S, B]):
        W_T = numerical_range(C)
        eig_T = np.linalg.eigvals(C)
        axes[i].plot(
            np.append(W_T.real, W_T.real[0]),
            np.append(W_T.imag, W_T.imag[0]),
            lw=1,
            label=rf"W({names[i]})",
            ls="-",
            color=colors[i],
        )
    ax2.fill(W_T.real, W_T.imag, alpha=0.1, zorder=3, color="purple")

    U, _ = unitary_dilation_with_shift(T, tol=1e-13)
    eig_U_tot = order_phases(np.linalg.eigvals(U))
    cluster = phase_clustering(eig_U_tot)
    eig_U = np.array([np.mean(eig_U_tot[idx]) for idx in cluster])
    ax2.plot(
        np.append(eig_U.real, eig_U.real[0]),
        np.append(eig_U.imag, eig_U.imag[0]),
        "o-",
        color="g",
        ms=6,
        label=r"spec $(U_{dil})$",
        zorder=3,
    )

    for ax in [ax0, ax1, ax2]:
        circ = plt.Circle(
            (0, 0), 1, fill=False, ls="--", color="gray", label=r"$\partial\mathbb{D}$"
        )
        ax.add_patch(circ)

        ax.set_aspect("equal")
        ax.axhline(0, lw=0.3)
        ax.axvline(0, lw=0.3)

        ax.legend(
            loc="upper left",
            handler_map={circ: HandlerPatch(patch_func=circle_handle)},
            handlelength=1.2,
            handleheight=1.2,
        )
        ax.set_xlabel("X")
        ax.xaxis.set_ticks(np.linspace(-1, 1, 5))
        ax.set_ylabel("Z")
        ax.yaxis.set_ticks(np.linspace(-1, 1, 5))
    # ax0.set_title(rf"$T= U \oplus A$")


def figure_both(ax, T):
    n = len(T)
    W = numerical_range(T)
    center = np.trace(T) / n
    eig = np.linalg.eigvals(T)

    V, _, d = minimal_unitary_dilation(T)
    eig_V = order_phases(np.linalg.eigvals(V))

    U, _ = unitary_dilation_with_shift(T)
    eig_U_tot = order_phases(np.linalg.eigvals(U))
    cluster = phase_clustering(eig_U_tot)
    eig_U = np.array([np.mean(eig_U_tot[idx]) for idx in cluster])

    ax.fill(W.real, W.imag, alpha=0.2, label="W(T)")
    ax.plot(np.append(W.real, W.real[0]), np.append(W.imag, W.imag[0]), lw=1)
    ax.plot(center.real, center.imag, "x", color="k", ms=8, label=r"$\it{a}$")
    ax.scatter(eig.real, eig.imag, c="navy", zorder=3, label=r"spec$(T)$")
    ax.plot(
        eig_U.real,
        eig_U.imag,
        "o",
        color="g",
        ms=6,
        label=rf"spec$(U)$, $|\text{{spec}}(U)|$ = {len(eig_U)}",
    )
    ax.plot(
        np.append(eig_U.real, eig_U.real[0]),
        np.append(eig_U.imag, eig_U.imag[0]),
        "-",
        color="g",
        ms=6,
    )
    ax.plot(
        eig_V.real,
        eig_V.imag,
        "o",
        color="indigo",
        ms=6,
        label=rf"spec$(V)$, $|\text{{spec}}(V)|$ = {len(eig_V)}",
    )
    ax.plot(
        np.append(eig_V.real, eig_V.real[0]),
        np.append(eig_V.imag, eig_V.imag[0]),
        "-",
        color="indigo",
        ms=6,
    )
    circ = plt.Circle(
        (0, 0), 1, fill=False, ls="--", color="gray", label=r"$\partial\mathbb{D}$"
    )
    ax.add_patch(circ)

    ax.set_aspect("equal")
    ax.axhline(0, lw=0.3)
    ax.axvline(0, lw=0.3)
    # ax.set_title(f"Numerical range, ||A||={np.linalg.norm(A,2):.3f}, n = {n}, seed={seed}")
    ax.set_title(rf"n = {n}, rank($\rho$) = {n+d}")
    ax.legend(
        loc="upper left",
        handler_map={circ: HandlerPatch(patch_func=circle_handle)},
        handlelength=1.2,
        handleheight=1.2,
    )
    ax.set_xlabel("X")
    ax.xaxis.set_ticks(np.linspace(-1, 1, 5))
    ax.set_ylabel("Z")
    ax.yaxis.set_ticks(np.linspace(-1, 1, 5))


def figure_poncelet_pure(ax, T):
    n = len(T)
    W = numerical_range(T)
    center = np.trace(T) / n
    eig = np.linalg.eigvals(T)

    ax.fill(W.real, W.imag, alpha=0.2, label="W(T)")
    ax.plot(np.append(W.real, W.real[0]), np.append(W.imag, W.imag[0]), lw=1)
    ax.plot(center.real, center.imag, "x", color="k", ms=8, label=r"$\it{a}$")
    ax.scatter(eig.real, eig.imag, c="navy", zorder=3, label=r"spec$(T)$")

    for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):

        U, _, d = minimal_unitary_dilation(T, theta=theta, tol=1e-13)
        eig_U = order_phases(np.linalg.eigvals(U))
        ax.plot(
            np.append(eig_U.real, eig_U.real[0]),
            np.append(eig_U.imag, eig_U.imag[0]),
            "go-",
            ms=6,
        )

    circ = plt.Circle(
        (0, 0), 1, fill=False, ls="--", color="gray", label=r"$\partial\mathbb{D}$"
    )
    ax.add_patch(circ)

    ax.set_aspect("equal")
    ax.axhline(0, lw=0.3)
    ax.axvline(0, lw=0.3)
    # ax.set_title(f"Numerical range, ||A||={np.linalg.norm(A,2):.3f}, n = {n}, seed={seed}")
    ax.set_title(rf"n = {n}")
    ax.legend(
        fontsize=12,
        loc="upper left",
        handler_map={circ: HandlerPatch(patch_func=circle_handle)},
        handlelength=1.2,
        handleheight=1.2,
    )
    ax.set_xlabel("X")
    ax.xaxis.set_ticks(np.linspace(-1, 1, 5))
    ax.set_ylabel("Z")
    ax.yaxis.set_ticks(np.linspace(-1, 1, 5))


def figure_n2(ax, T):
    n = len(T)
    W = numerical_range(T)
    center = np.trace(T) / n
    eig = np.linalg.eigvals(T)

    ax.fill(W.real, W.imag, alpha=0.2, label="W(T)")
    ax.plot(np.append(W.real, W.real[0]), np.append(W.imag, W.imag[0]), lw=1)
    ax.plot(center.real, center.imag, "x", color="k", ms=8, label=r"$\it{a}$")
    ax.scatter(eig.real, eig.imag, c="navy", zorder=3, label=r"spec$(T)$")

    U, _, d = minimal_unitary_dilation(T / np.linalg.norm(T, 2), tol=1e-13)
    eig_U = order_phases(np.linalg.norm(T, 2) * np.linalg.eigvals(U))
    ax.plot(
        np.append(eig_U.real, eig_U.real[0]),
        np.append(eig_U.imag, eig_U.imag[0]),
        "go-",
        ms=6,
    )

    circ1 = plt.Circle(
        (0, 0), 1, fill=False, ls="--", color="gray", label=r"$\partial\mathbb{D}$"
    )
    ax.add_patch(circ1)
    circ2 = plt.Circle(
        (0, 0),
        np.linalg.norm(T, 2),
        fill=False,
        ls="--",
        color="b",
        label=r"$\mathbb{B}(0, ||T||)$",
    )
    ax.add_patch(circ2)

    ax.set_aspect("equal")
    ax.axhline(0, lw=0.3)
    ax.axvline(0, lw=0.3)

    ax.set_title(rf"n = {n}")
    ax.legend(
        fontsize=12,
        loc="upper left",
        handler_map={
            circ1: HandlerPatch(patch_func=circle_handle),
            circ2: HandlerPatch(patch_func=circle_handle),
        },
        handlelength=1.2,
        handleheight=1.2,
    )
    ax.set_xlabel("X")
    ax.xaxis.set_ticks(np.linspace(-1, 1, 5))
    ax.set_ylabel("Z")
    ax.yaxis.set_ticks(np.linspace(-1, 1, 5))


if __name__ == "__main__":
    figure_1 = True
    figure_jn = False
    figure_poncelet = False
    figure_mixed_2 = False
    figure_n_2 = False
    figure_U_A = False
    figure_halmos = False

    if figure_1:
        nb_figs = 3
        fig, axs = plt.subplots(1, nb_figs, figsize=(18, 5))
        seed = np.random.randint(0, 1000)
        # seed = 523
        print("Random seed:", seed)
        rng = np.random.default_rng(seed)
        n = -2
        for i in range(2):
            n += 4
            rho = sample_bipartite_dm(n, d=2 * n, rng=rng)
            rho_new, F = cariello_filtering(rho, n)
            rho_mm, F_mm = maximally_mixed_B_marginals(rho_new, n)
            print(
                f"Trace of rho: {np.trace(rho_mm):.4f}",
                "eigenvalues of rho:",
                np.linalg.eigvalsh(rho_mm),
            )
            C1, C3 = build_C_from_correlation(build_T_correlation(rho_mm, n))
            T = C1 + 1j * C3
            figure_numerical_range(axs[i], T)

        n = 5
        U = haar(n, rng)
        figure_numerical_range(axs[2], U)
        plt.tight_layout()
        plt.show()

    if figure_halmos:
        nb_figs = 3
        fig, axs = plt.subplots(1, nb_figs, figsize=(16, 6))
        seed = np.random.randint(0, 1000)
        # seed = 523
        print("Random seed:", seed)
        rng = np.random.default_rng(seed)
        n = 5
        for i in range(nb_figs):
            rho = sample_bipartite_dm(n, d=n + i * 2 + 1, rng=rng)
            rho_new, F = cariello_filtering(rho, n)
            rho_mm, F_mm = maximally_mixed_B_marginals(rho_new, n)
            print(
                f"Trace of rho: {np.trace(rho_mm):.4f}",
                "eigenvalues of rho:",
                np.linalg.eigvalsh(rho_mm),
            )
            C1, C3 = build_C_from_correlation(build_T_correlation(rho_mm, n))
            T = C1 + 1j * C3
            figure_pure(axs[i], T, i)

        plt.tight_layout()
        plt.show()

    if figure_jn:
        n = 3
        nb_figs = 1
        fig, axs = plt.subplots(1, nb_figs, figsize=(6, 6))
        T = np.diag(np.ones(n - 1), k=1)
        rho = build_rho_from_T_contraction(T)
        print(rho)
        print(
            "Trace of rho:",
            np.trace(rho),
            "eigenvalues of rho:",
            np.linalg.eigvalsh(rho),
            "rank of rho: n +",
            np.linalg.matrix_rank(rho) - n,
        )
        print(np.linalg.eigvals(T))
        figure_mixed(axs, T, 0)
        plt.tight_layout()
        plt.show()

    if figure_poncelet:
        n = 2
        nb_figs = 1
        fig, axs = plt.subplots(1, nb_figs, figsize=(6, 6))
        seed = np.random.randint(0, 1000)
        # seed = 250
        print("Random seed:", seed)
        rng = np.random.default_rng(seed)
        rho = sample_bipartite_dm(n, d=n + 1, rng=rng)
        rho_new, F = cariello_filtering(rho, n)
        rho_mm, F_mm = maximally_mixed_B_marginals(rho_new, n)
        print(
            "Trace of rho:",
            np.trace(rho),
            "eigenvalues of rho:",
            np.linalg.eigvalsh(rho_mm),
            "rank of rho: n +",
            np.linalg.matrix_rank(rho_mm) - n,
        )
        C1, C3 = build_C_from_correlation(build_T_correlation(rho_mm, n))
        T = C1 + 1j * C3

        print(np.linalg.eigvals(T))
        figure_poncelet_pure(axs, T)
        plt.tight_layout()
        plt.show()

    if figure_mixed_2:
        nb_figs = 3
        fig, axs = plt.subplots(1, nb_figs, figsize=(16, 6))
        seed = np.random.randint(0, 1000)
        # seed = 503
        print("Random seed:", seed)
        rng = np.random.default_rng(seed)
        n = -1
        for i in range(nb_figs):
            n += 3
            rho = sample_bipartite_dm(n, d=2 * n, rng=rng)
            rho_new, F = cariello_filtering(rho, n)
            rho_mm, F_mm = maximally_mixed_B_marginals(rho_new, n)
            print(
                f"Trace of rho: {np.trace(rho_mm):.4f}",
                "eigenvalues of rho:",
                np.linalg.eigvalsh(rho_mm),
            )
            C1, C3 = build_C_from_correlation(build_T_correlation(rho_mm, n))
            T = C1 + 1j * C3
            reconstructed_rho = build_rho_from_T_contraction(T)
            figure_mixed(axs[i], T, i)
            # figure_both(axs[i], T)
        plt.tight_layout()
        plt.show()

    if figure_U_A:
        nb_figs = 3
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 6))
        fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6))
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 6))
        axes = [ax1, ax2, ax3]
        n = 6
        Ud = np.diag([1.0 + 0j, (1 + 1j) / np.sqrt(2), -1j])
        Ab = np.array(
            [[0.42j, 0.33, 0.11], [-0.08, 0.52j, -0.19], [0.21, 0.13, 0.49]],
            dtype=complex,
        )
        assert len(Ud) + len(Ab) == n
        T0 = np.zeros((n, n), dtype=complex)
        T0[: len(Ud), : len(Ud)] = Ud
        T0[len(Ud) :, len(Ud) :] = Ab
        rng = np.random.default_rng(0)
        Q, R = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
        Q = Q @ np.diag(np.diag(R) / abs(np.diag(R)))
        T = Q @ T0 @ Q.conj().T
        reconstructed_rho = build_rho_from_T_contraction(T)
        figure_UA(axes, T)
        plt.show()

    if figure_n_2:
        n = 2
        nb_figs = 1
        fig, axs = plt.subplots(1, nb_figs, figsize=(6, 6))
        seed = np.random.randint(0, 1000)
        # seed = 553
        print("Random seed:", seed)
        rng = np.random.default_rng(seed)
        rho = sample_bipartite_dm(n, d=2 * n, rng=rng)
        rho_new, F = cariello_filtering(rho, n)
        rho_mm, F_mm = maximally_mixed_B_marginals(rho_new, n)
        print(
            "Trace of rho:",
            np.trace(rho_mm),
            "eigenvalues of rho:",
            np.linalg.eigvalsh(rho_mm),
            "rank of rho: n +",
            np.linalg.matrix_rank(rho_mm) - n,
        )
        C1, C3 = build_C_from_correlation(build_T_correlation(rho_mm, n))
        T = C1 + 1j * C3

        print(np.linalg.eigvals(T))
        figure_n2(axs, T)
        plt.tight_layout()
        plt.show()
