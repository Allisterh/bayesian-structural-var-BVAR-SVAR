import numpy as np
import matplotlib.pyplot as plt


def plot_a0_traces_and_hists(A0, mode="optimized", L=None, param_names=None, bins="auto"):
    """
    Visualiza traza + histograma por parámetro.

    Parámetros
    ----------
    A0 : np.ndarray
        - mode="optimized": array (P, 2*L) con cadenas concatenadas c1|c2 por fila.
        - mode="plain"    : array (P, N) con una sola cadena (o concatenación que NO se separa).
    mode : {"optimized","plain"}
        - "optimized": separa c1 y c2 (mitades iguales o longitud L dada).
        - "plain"    : no separa; todo es una única cadena por parámetro.
    L : int, opcional
        - Solo para mode="optimized". Si None, se infiere como N//2.
    param_names : list[str], opcional
        - Nombres para los θ en títulos/ejes. Si None, usa θ1, θ2, ...
    bins : int o "auto"
        - Bins para los histogramas.

    Notas
    -----
    - En "optimized", dibuja línea vertical en el corte y superpone histogramas de c1 y c2.
    - Marca medias por cadena y media total.
    """
    A0 = np.asarray(A0)
    P, N = A0.shape

    if param_names is None:
        param_names = [f"θ{i+1}" for i in range(P)]

    if mode not in {"optimized", "plain"}:
        raise ValueError("mode debe ser 'optimized' o 'plain'.")

    if mode == "optimized":
        if L is None:
            if N % 2 != 0:
                raise ValueError("Para mode='optimized' y L=None, N debe ser par (concatenación c1|c2).")
            L = N // 2
        if L <= 0 or L > N:
            raise ValueError("L inválido para mode='optimized'.")

    for i in range(P):
        chain = A0[i, :]
        pname = param_names[i] if i < len(param_names) else f"θ{i+1}"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # ---------------- TRACE ----------------
        ax_trace = axes[0]

        if mode == "optimized":
            # divide en dos cadenas
            c1 = chain[:L]
            c2 = chain[L:L*2]  # por si N != 2*L, recorta con seguridad
            idx1 = np.arange(0, len(c1))
            idx2 = np.arange(L, L + len(c2))

            ax_trace.plot(idx1, c1, lw=0.9, label="Cadena 1")
            ax_trace.plot(idx2, c2, lw=0.9, label="Cadena 2")
            ax_trace.axvline(L, linestyle="--", alpha=0.7)
            ax_trace.legend()

        else:  # plain
            ax_trace.plot(chain, lw=0.9)

        ax_trace.set_title(f"Traza del parámetro {pname}")
        ax_trace.set_xlabel("Iteraciones")
        ax_trace.set_ylabel(pname)

        # ---------------- HIST ----------------
        ax_hist = axes[1]

        if mode == "optimized":
            c1 = chain[:L]
            c2 = chain[L:L*2]
            m1 = float(np.mean(c1))
            m2 = float(np.mean(c2))
            mall = float(np.mean(chain))

            u1 = np.unique(c1)
            u2 = np.unique(c2)

            if (u1.size == 1) and (u2.size == 1) and (u1[0] == u2[0]):
                # todo constante
                x0 = u1[0]
                ax_hist.bar(x0, 1.0, width=1e-6, alpha=0.6, label="Cadena 1/2 (const.)")
                ax_hist.set_xlim(x0 - 1e-3, x0 + 1e-3)
            else:
                ax_hist.hist(c1, bins=bins, density=True, alpha=0.5, label="Cadena 1")
                ax_hist.hist(c2, bins=bins, density=True, alpha=0.5, label="Cadena 2")

            ax_hist.axvline(m1, linestyle="--", linewidth=1.4, label=f"Mean c1 = {m1:.4f}")
            ax_hist.axvline(m2, linestyle="--", linewidth=1.4, label=f"Mean c2 = {m2:.4f}")
            ax_hist.axvline(mall, linestyle="-",  linewidth=1.4, label=f"Mean total = {mall:.4f}")

        else:
            m = float(np.mean(chain))
            u = np.unique(chain)
            if u.size == 1:
                x0 = u[0]
                ax_hist.bar(x0, 1.0, width=1e-6, alpha=0.6, label="Constante")
                ax_hist.set_xlim(x0 - 1e-3, x0 + 1e-3)
            else:
                ax_hist.hist(chain, bins=bins, density=True, alpha=0.6, label="Cadena")
            ax_hist.axvline(m, linestyle="--", linewidth=1.4, label=f"Mean = {m:.4f}")

        ax_hist.set_title(f"Histograma de {pname}")
        ax_hist.set_xlabel(pname)
        ax_hist.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax_hist.ticklabel_format(style="plain", axis="y")
        ax_hist.legend()

        plt.tight_layout()
        plt.show()
