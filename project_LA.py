"""
MATB32 Linear Algebra, Programming Project (HT 2026).

Group 3: Folke Adolfsson, Bruno Benyamine Remahl, Arvid Brenner, Sixten Midsem.

Task 1: Linear regression as a least squares problem, projections and
        reflections.
Task 2: Eigenvalues, eigenvectors and the recurrence z_{n+1} = A z_n.

Run with
    python project_LA.py
The data files regression_1.dat and regression_2.dat must be in the same
directory as this script. All figures are shown and also saved to the
subdirectory figures/.

Libraries: numpy, scipy, matplotlib (nothing else).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, optimize

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

np.set_printoptions(precision=6, suppress=False)


def save(fig, name):
    """Save a figure as PDF (for the report) and PNG in figures/."""
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150, bbox_inches="tight")


# =============================================================================
# Task 1: Linear regression, projections, reflections, least squares
# =============================================================================

def load_datasets():
    """
    Read the two data files and return a list of (name, x, y) tuples.

    regression_1.dat has columns x, y.
    regression_2.dat has columns x, y1, y2, y3 (three series, same x).
    """
    d1 = np.loadtxt(HERE / "regression_1.dat", delimiter=",", skiprows=1)
    d2 = np.loadtxt(HERE / "regression_2.dat", delimiter=",", skiprows=1)
    datasets = [("Dataset 1 (regression_1)", d1[:, 0], d1[:, 1])]
    for k in range(1, 4):
        datasets.append((f"Dataset 2.{k} (regression_2, y{k})",
                         d2[:, 0], d2[:, k]))
    return datasets


def design_matrix(x):
    """Return the n x 2 matrix A = [x, 1] so that A @ [a, b] = a x + b."""
    return np.column_stack((x, np.ones_like(x)))


def squared_error(beta, A, y):
    """Objective ||A beta - y||^2 of the least squares problem."""
    r = A @ beta - y
    return r @ r


def fit_fmin(A, y):
    """Minimise ||A beta - y||^2 directly with scipy.optimize.fmin."""
    return optimize.fmin(squared_error, x0=np.zeros(2), args=(A, y),
                         xtol=1e-12, ftol=1e-12, maxiter=10_000,
                         maxfun=20_000, disp=False)


def fit_normal_equations(A, y):
    """Solve the normal equations A^T A beta = A^T y (Elfstrom 3.5)."""
    return np.linalg.solve(A.T @ A, A.T @ y)


def fit_lstsq(A, y):
    """Solve the least squares problem with numpy.linalg.lstsq."""
    return np.linalg.lstsq(A, y, rcond=None)[0]


def projection_matrix(A):
    """Orthogonal projection on im A: P = A (A^T A)^{-1} A^T (Elfstrom 3.4)."""
    return A @ np.linalg.solve(A.T @ A, A.T)


def is_projection(P, tol=1e-10):
    """P is a projection iff P^2 = P (Elfstrom Thm 5.16)."""
    return np.allclose(P @ P, P, atol=tol)


def is_orthogonal_projection(P, tol=1e-10):
    """P is an orthogonal projection iff P^2 = P and P = P^T (Thm 5.27 (i))."""
    return is_projection(P, tol) and np.allclose(P, P.T, atol=tol)


def is_reflection(R, tol=1e-10):
    """R is a reflection iff R^2 = I (Elfstrom Thm 5.20)."""
    return np.allclose(R @ R, np.eye(R.shape[0]), atol=tol)


def is_orthogonal_reflection(R, tol=1e-10):
    """R is an orthogonal reflection iff R^2 = I and R = R^T (Thm 5.27)."""
    return is_reflection(R, tol) and np.allclose(R, R.T, atol=tol)


def r_squared(y, y_hat):
    """Coefficient of determination, used only to comment on the fit."""
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def analyse_dataset(name, x, y):
    """Run subtasks 1.1, 1.5, 1.6, 1.8 and 1.9 for one dataset."""
    A = design_matrix(x)
    beta_fmin = fit_fmin(A, y)
    beta_ne = fit_normal_equations(A, y)
    beta_ls = fit_lstsq(A, y)

    P = projection_matrix(A)
    y_hat = P @ y
    R = 2 * P - np.eye(len(y))
    Ry = R @ y
    residual = y - y_hat

    print(f"\n--- {name} (n = {len(x)}) ---")
    print(
        f"  fmin:             a = {beta_fmin[0]: .10f}, "
        f"b = {beta_fmin[1]: .10f}")
    print(
        f"  normal equations: a = {beta_ne[0]: .10f}, b = {beta_ne[1]: .10f}")
    print(
        f"  numpy lstsq:      a = {beta_ls[0]: .10f}, b = {beta_ls[1]: .10f}")
    print(
        "  |beta_fmin - beta_lstsq| = "
        f"{np.linalg.norm(beta_fmin - beta_ls):.2e}")
    print(
        f"  min error ||A beta - y|| = {np.linalg.norm(A @ beta_ls - y):.6f}")
    print(f"  R^2 = {r_squared(y, y_hat):.4f}")
    print(f"  P y equals A beta:            {np.allclose(y_hat, A @ beta_ls)}")
    print(f"  P is a projection (P^2 = P):  {is_projection(P)}")
    print(f"  P is orthogonal (P = P^T):    {is_orthogonal_projection(P)}")
    print(
        f"  rank P = trace P = {np.linalg.matrix_rank(P)}, {np.trace(P):.6f}")
    # 1.8: only y and y_hat are used here
    print(f"  <y_hat, y - y_hat> = {y_hat @ residual: .2e}")
    print(f"  ||y||^2 - ||y_hat||^2 - ||y - y_hat||^2 = "
          f"{y @ y - y_hat @ y_hat - residual @ residual: .2e}")
    print(f"  R = 2P - I is a reflection (R^2 = I):    {is_reflection(R)}")
    print(f"  R is an orthogonal reflection (R = R^T): "
          f"{is_orthogonal_reflection(R)}")
    print(f"  ||R y|| - ||y|| = {np.linalg.norm(Ry) - np.linalg.norm(y): .2e}")

    return dict(name=name, x=x, y=y, A=A, beta=beta_ls, beta_fmin=beta_fmin,
                P=P, y_hat=y_hat, R=R, Ry=Ry)


def plot_fits(results, extended=False):
    """
    Subtask 1.2 (extended=False): data points and fitted line.
    Subtasks 1.7 and 1.9 (extended=True): also y_hat = P y and R y.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, res in zip(axes.flat, results):
        x, y = res["x"], res["y"]
        a, b = res["beta"]
        xs = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
        ax.plot(xs, a * xs + b, "C0-", lw=1.5,
                label=f"$f(x) = {a:.3f}x {b:+.3f}$")
        ax.plot(x, y, "ko", ms=5, label="data $y$")
        if extended:
            ax.plot(x, res["y_hat"], "C1s", ms=5, mfc="none", mew=1.5,
                    label=r"$\hat y = Py$")
            ax.plot(x, res["Ry"], "C3^", ms=5, label="$Ry = (2P - I)y$")
            for xi, yi, ri in zip(x, y, res["Ry"]):
                ax.plot([xi, xi], [yi, ri], color="0.75", lw=0.8, zorder=0)
        ax.set_title(res["name"])
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "task1_fit_extended" if extended else "task1_fit")
    return fig


def plot_error_surfaces(results):
    """Subtask 1.3: surface of eps(a, b) = ||A [a, b]^T - y||, all datasets."""
    fig = plt.figure(figsize=(12, 9.5))
    for k, res in enumerate(results):
        A, y = res["A"], res["y"]
        a0, b0 = res["beta"]
        a = np.linspace(a0 - 0.6, a0 + 0.6, 121)
        b = np.linspace(b0 - 4.0, b0 + 4.0, 121)
        AA, BB = np.meshgrid(a, b)
        # residual for every grid point: r = a x + b - y, shape (len(b),
        # len(a), n)
        resid = AA[..., None] * res["x"] + BB[..., None] - y
        E = np.linalg.norm(resid, axis=-1)

        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        ax.plot_surface(AA, BB, E, cmap="viridis", alpha=0.75,
                        linewidth=0, antialiased=True)
        ax.contour(AA, BB, E, levels=12, cmap="viridis", offset=0)
        e_min = np.linalg.norm(A @ res["beta"] - y)
        ax.scatter([a0], [b0], [e_min], color="red", s=40, depthshade=False,
                   label=f"$(a, b) = ({a0:.3f}, {b0:.3f})$")
        ax.scatter([a0], [b0], [0], color="red", marker="x", s=40)
        ax.set_xlabel("$a$")
        ax.set_ylabel("$b$")
        ax.set_zlabel(r"$\epsilon(a, b)$", labelpad=2)
        ax.set_box_aspect(None, zoom=0.88)
        ax.set_zlim(0, E.max())
        ax.set_title(res["name"], fontsize=10)
        ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    save(fig, "task1_error_surface")
    return fig


def discuss_fit_quality(results):
    """Numbers used in the discussion of subtask 1.2."""
    print("\n--- Subtask 1.2: quality of the fit ---")
    for res in results:
        x, y, y_hat = res["x"], res["y"], res["y_hat"]
        r = y - y_hat
        k = np.argmax(np.abs(r))
        # quadratic fit for comparison
        c2 = np.polyfit(x, y, 2)
        r2_quad = r_squared(y, np.polyval(c2, x))
        # refit without the point with the largest residual
        mask = np.arange(len(x)) != k
        beta_wo = fit_lstsq(design_matrix(x[mask]), y[mask])
        r2_wo = r_squared(y[mask], design_matrix(x[mask]) @ beta_wo)
        print(f"{res['name']}:")
        print(f"  R^2 linear = {r_squared(y, y_hat):.4f}, "
              f"R^2 quadratic = {r2_quad:.4f} (coef {c2.round(4)})")
        print(f"  largest residual {r[k]: .3f} at x = {x[k]}, "
              f"next largest {np.sort(np.abs(r))[-2]:.3f}")
        print(
            f"  without that point: a = {beta_wo[0]:.4f}, "
            f"b = {beta_wo[1]:.4f},"
            f" R^2 = {r2_wo:.6f}")


def task1():
    print("=" * 70)
    print("TASK 1")
    print("=" * 70)
    results = [analyse_dataset(*d) for d in load_datasets()]
    discuss_fit_quality(results)
    plot_fits(results, extended=False)
    plot_error_surfaces(results)
    plot_fits(results, extended=True)
    return results


# =============================================================================
# Task 2: Eigenvalues, eigenvectors, recurrence relations
# =============================================================================

A_REC = np.array([[1.0, 3.0, 2.0],
                  [-3.0, 4.0, 3.0],
                  [2.0, 3.0, 1.0]])


def initial_values():
    """The five initial values z0 used in Task 2, with a label for each."""
    cases = [("[9, 1, 9]", np.array([9.0, 1.0, 9.0]))]
    for alpha, alpha_txt in [(1.0, "1"), (1.0 / 19.0, "1/19")]:
        cases.append((f"{alpha_txt}*[1, 0, 1]",
                      alpha * np.array([1.0, 0.0, 1.0])))
        cases.append((f"{alpha_txt}*[1, 12, -19]",
                      alpha * np.array([1.0, 12.0, -19.0])))
    return cases


# Integer eigenvectors of A_REC, read off from the scipy result below
# (columns scaled to integers) and verified exactly in task2().
EIGVALS_EXACT = np.array([4.0, 3.0, -1.0])
EIGVECS_EXACT = np.array([[3.0, 1.0, 1.0],
                          [1.0, 0.0, 12.0],
                          [3.0, 1.0, -19.0]])


def eigen_decomposition(A):
    """
    Eigenvalues and eigenvectors of A with scipy.linalg.eig, sorted by
    decreasing absolute value. The eigenvalues of A_REC are real, so the
    (zero) imaginary parts are dropped. Each eigenvector is scaled so that
    its first component is 1.
    """
    lam, V = linalg.eig(A)
    assert np.allclose(lam.imag, 0) and np.allclose(V.imag, 0)
    lam, V = lam.real, V.real
    order = np.argsort(-np.abs(lam))
    return lam[order], V[:, order] / V[0, order]


def unit(z):
    """
    z / ||z||, computed without overflow: z is first divided by its largest
    entry (|z_n| grows like 4^n, and ||z||^2 overflows for n > 255).
    """
    z = z / np.max(np.abs(z))
    return z / np.linalg.norm(z)


def eigen_coordinates(E, z0):
    """Coordinates c of z0 in the eigenvector basis: z0 = E c."""
    return np.linalg.solve(E, z0)


def closed_form(lam, E, c, n):
    """z_n = sum_i c_i lambda_i^n e_i = E diag(lambda^n) c (Elfstrom 6.4)."""
    return E @ (lam ** n * c)


def iterate(A, z0, N):
    """Compute z_0, ..., z_N with z_{n+1} = A z_n. Returns (N+1, 3) array."""
    Z = np.empty((N + 1, len(z0)))
    Z[0] = z0
    for n in range(N):
        Z[n + 1] = A @ Z[n]
    return Z


def normalized_iterate(A, z0, N):
    """Compute v_0 = z0/||z0||, v_n = A v_{n-1}/||A v_{n-1}||, n = 1..N."""
    Vn = np.empty((N + 1, len(z0)))
    Vn[0] = z0 / np.linalg.norm(z0)
    for n in range(N):
        w = A @ Vn[n]
        Vn[n + 1] = w / np.linalg.norm(w)
    return Vn


def rayleigh(A, Vn):
    """q_n = v_n^T A v_n for every row v_n of Vn."""
    return np.einsum("ni,ij,nj->n", Vn, A, Vn)


def theoretical_limit(lam, E, c, tol=1e-12):
    """
    Limit of v_n predicted by the eigen-expansion. The term with the largest
    |lambda_i| among those with c_i != 0 dominates, so
        v_n ~ sign(c_i lambda_i^n) e_i / ||e_i||   and   q_n -> lambda_i.
    Returns (v, q, oscillates). If lambda_i < 0 the sign alternates and v_n
    has no limit (oscillates = True); v is then the limit of v_{2m}.
    """
    scale = np.max(np.abs(c))
    i = next(j for j in range(len(c)) if abs(c[j]) > tol * scale)
    return np.sign(c[i]) * unit(E[:, i]), lam[i], lam[i] < 0


def line_distance(Vn, v):
    """Distance from each unit vector v_n to the line spanned by v."""
    return np.minimum(np.linalg.norm(Vn - v, axis=1),
                      np.linalg.norm(Vn + v, axis=1))


def steps_to_tolerance(err, eps):
    """
    Smallest n such that err[m] < eps for all m >= n (m <= N).
    Returns None if the tolerance is never reached permanently.
    """
    bad = np.nonzero(err >= eps)[0]
    if len(bad) == 0:
        return 0
    n = bad[-1] + 1
    return n if n < len(err) else None


def task2(N=400):
    print("\n" + "=" * 70)
    print("TASK 2")
    print("=" * 70)
    A = A_REC
    lam, V = eigen_decomposition(A)
    print("Eigenvalues (scipy.linalg.eig):", lam)
    print("Eigenvectors (columns, scaled so the first entry is 1):")
    print(V)
    print("Check A V = V diag(lam):", np.allclose(A @ V, V * lam))
    E, lam_ex = EIGVECS_EXACT, EIGVALS_EXACT
    print("Integer eigenvectors e1, e2, e3 (columns):")
    print(E)
    print("Exact check A E == E diag(4, 3, -1):",
          np.array_equal(A @ E, E * lam_ex))
    print("Parallel to the scipy eigenvectors:",
          np.allclose(E / E[0], V))

    out = []
    for label, z0 in initial_values():
        c = eigen_coordinates(E, z0)
        Z = iterate(A, z0, N)
        Vn = normalized_iterate(A, z0, N)
        q = rayleigh(A, Vn)
        v_th, q_th, osc = theoretical_limit(lam_ex, E, c)
        out.append(dict(label=label, z0=z0, c=c, Z=Z, Vn=Vn, q=q,
                        v_th=v_th, q_th=q_th, osc=osc))

        print(f"\n--- z0 = {label} ---")
        print(f"  z0 = c1 e1 + c2 e2 + c3 e3 with c = {np.round(c, 12) + 0.0}")
        print(
            f"  largest entry of z_N:              {np.max(np.abs(Z[N])):.3e}")
        print(f"  z_N / ||z_N||   (iteration, 2.2):  {unit(Z[N])}")
        print(f"  z_N-1 / ||z_N-1||:                 {unit(Z[N - 1])}")
        print(
            "  closed form, z_N / ||z_N||:        "
            f"{unit(closed_form(lam_ex, E, c, N))}")
        print(f"  v_N   (normalized, 2.4):           {Vn[N]}")
        print(f"  v_N-1:                             {Vn[N - 1]}")
        print(f"  theoretical v (2.3):               {v_th}"
              f"{'  (sign alternates)' if osc else ''}")
        print(f"  q_N = {q[N]:.16f}, q_N-1 = {q[N - 1]:.16f}, "
              f"theoretical q = {q_th}")
        for nm, seq in [("z_n", np.array([unit(z) for z in Z])), ("v_n", Vn)]:
            d = line_distance(seq, v_th)
            if d[-1] > 1e-6:
                print(f"  {nm} leaves the line through v (distance > 1e-6) "
                      f"at n = {np.argmax(d > 1e-6)}")
    return out


def plot_line_distance(out):
    """Distance from v_n to the line through the theoretical limit v."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, d in enumerate(out):
        dist = line_distance(d["Vn"], d["v_th"])
        ax.semilogy(
            np.arange(
                len(dist)),
            np.maximum(
                dist,
                1e-17),
            f"C{k}-",
            lw=1.4,
            label=f"$z_0 = {d['label']}$".replace(
                "*",
                r"\cdot"))
    ax.set_xlabel("$n$")
    ax.set_ylabel(r"$\min(\|v_n - v\|, \|v_n + v\|)$")
    ax.set_title(
        "Distance from $v_n$ to the line through the theoretical limit $v$")
    ax.set_xlim(0, 120)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "task2_line_distance")
    return fig


def convergence_table(out, eps=1e-8):
    """Subtask 2.7: iterations needed until ||v_n - v*|| < eps."""
    print("\n--- Subtask 2.7: smallest n with ||v_m - v*|| < 1e-8 "
          "for all m >= n ---")
    print(f"{'z0':>18} | {'v* = v (theory)':>16} | {'v* = v_400':>10} |"
          f" {'q* = q':>7} | {'q* = q_400':>10}")
    rows = []
    for d in out:
        Vn, q = d["Vn"], d["q"]
        n_v_th = steps_to_tolerance(
            np.linalg.norm(
                Vn - d["v_th"], axis=1), eps)
        n_v_num = steps_to_tolerance(np.linalg.norm(Vn - Vn[-1], axis=1), eps)
        n_q_th = steps_to_tolerance(np.abs(q - d["q_th"]), eps)
        n_q_num = steps_to_tolerance(np.abs(q - q[-1]), eps)
        rows.append((d["label"], n_v_th, n_v_num, n_q_th, n_q_num))
        def fmt(n): return "never" if n is None else str(n)
        print(f"{d['label']:>18} | {fmt(n_v_th):>16} | {fmt(n_v_num):>10} |"
              f" {fmt(n_q_th):>7} | {fmt(n_q_num):>10}")
    return rows


def plot_convergence(out):
    """Subtask 2.8: number of iterations versus tolerance eps (semilogy)."""
    eps_values = np.logspace(-1, -14, 60)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    for ax, d in zip(axes.flat, out):
        Vn, q = d["Vn"], d["q"]
        errors = {
            r"$\|v_n - v\|$ (theory)": np.linalg.norm(Vn - d["v_th"], axis=1),
            r"$\|v_n - v_{400}\|$": np.linalg.norm(Vn - Vn[-1], axis=1),
            r"$|q_n - q|$ (theory)": np.abs(q - d["q_th"]),
        }
        if q[-1] != d["q_th"]:
            errors[r"$|q_n - q_{400}|$"] = np.abs(q - q[-1])
        styles = ["C0o-", "C1s--", "C2^-", "C3v--"]
        for (lbl, err), st in zip(errors.items(), styles):
            n_needed = [steps_to_tolerance(err, e) for e in eps_values]
            ok = [(n, e)
                  for n, e in zip(n_needed, eps_values) if n is not None]
            if ok:
                ns, es = zip(*ok)
                ax.semilogy(ns, es, st, ms=3, lw=1, label=lbl)
            else:
                ax.semilogy([], [], st, label=lbl + ": never")
        title = f"$z_0 = {d['label']}$".replace("*", r"\cdot")
        if d["c"][0] == 0 and not d["osc"]:
            title += "  (exact eigenvector: $n = 0$ for every $\\varepsilon$)"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-3, 120)
        ax.set_xlabel("iterations $n$ needed")
        ax.set_ylabel(r"tolerance $\varepsilon$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    save(fig, "task2_convergence")
    return fig


def harmonic_example(N=10**6):
    """Subtask 2.6: x_n = sum_{j<=n} 1/j has small steps but diverges."""
    n = np.arange(1, N + 1)
    x = np.cumsum(1.0 / n)
    print("\n--- Subtask 2.6: harmonic series ---")
    for k in [10, 1000, 10**6]:
        print(
            f"  n = {k:>7}: x_n = {x[k - 1]:.4f}, x_n - x_(n-1) = {1 / k:.1e},"
            f" x_(n+1) - x_(n-1) = {1 / k + 1 / (k + 1):.1e}")


def main():
    task1()
    out = task2()
    plot_line_distance(out)
    harmonic_example()
    convergence_table(out)
    plot_convergence(out)
    plt.show()


if __name__ == "__main__":
    main()
