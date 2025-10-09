import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.lp_utils import *
import math
import matplotlib.pyplot as plt


def make_lags(df: pd.DataFrame, L: int) -> pd.DataFrame:
    cols = {}
    for c in df.columns:
        for l in range(1, L+1):
            cols[f"{c}_L{l}"] = df[c].shift(l)
    return pd.DataFrame(cols, index=df.index)


def align_for_horizon(y: pd.Series, shock: pd.Series, X: pd.DataFrame, h: int, cumulative: bool=False):
    y_dep = (y.shift(-h) - y.shift(-1)) if cumulative else y.shift(-h)
    df = pd.concat([y_dep, shock, X], axis=1).dropna()
    return df.iloc[:,0], df.iloc[:,1], df.iloc[:,2:]


def lp_irf_single_beta_se(y, shock, X_controls, H=20):
    betas, ses = [], []
    for h in range(H+1):
        y_h, s_t, X_t = align_for_horizon(y, shock, X_controls, h, cumulative=False)
        Xreg = sm.add_constant(pd.concat([s_t.rename('shock'), X_t], axis=1))
        res = sm.OLS(y_h, Xreg).fit(cov_type='HAC', cov_kwds={'maxlags': max(h+1,1)})
        betas.append(res.params['shock'])
        ses.append(res.bse['shock'])
    return np.asarray(betas), np.asarray(ses)


def lp_hac_from_ref_draw(
    order,
    H,
    p_lags,
    datos_var,      # df fuente solo para index
    YY,             # panel VAR (T x n) ya transformado
    X, Y,           # matrices del VAR (para residuales)
    Posterior_B_MCMC,          # (R,k,n)
    Posterior_A0_MCMC,         # (q_params, R)
    A0_Mtx,                     # callable: A0_Mtx(theta_r)->(n x n)
    r_ref=0,
    shock_var_name='tbills',
    include_shock_lags=False,
    normalize_shock='sd',       # None | 'sd' (1 s.d.)
    z68=1.0,
    z80=1.28155):

    """
    Devuelve:
      - betas: dict[var] -> array (H+1,)
      - ses:   dict[var] -> array (H+1,)
      - q50, lo68, up68, lo80, up80: matrices (H x n) listas para graficar (h=1..H).
    Requiere make_lags(...) y lp_irf_single_beta_se(...).
    """
    n = len(order)
    IDX_SHOCK = order.index(shock_var_name)

    # datos alineados con el VAR
    df_q  = datos_var.copy()
    df_lp = pd.DataFrame(YY, index=df_q.index[:len(YY)], columns=order)
    T_eff = X.shape[0]
    df_lp_eff = df_lp.iloc[-T_eff:].copy()
    idx_eff = df_lp_eff.index

    # draw de referencia
    B_ref  = Posterior_B_MCMC[r_ref, :, :]
    yhat   = X @ B_ref
    Y_eff  = Y[-T_eff:, :]
    u_ref  = Y_eff - yhat

    A0_ref = A0_Mtx(Posterior_A0_MCMC[:, r_ref])
    e_ref  = (A0_ref @ u_ref.T).T                         # (T_eff x n)

    shock_ref = pd.Series(e_ref[:, IDX_SHOCK], index=idx_eff, name='shock_tbills')
    if normalize_shock == 'sd':
        sd = shock_ref.std(ddof=1)
        if sd > 0:
            shock_ref = shock_ref / sd

    # controles LP (comparables con VAR)
    if include_shock_lags:
        tmp = df_lp_eff.copy()
        tmp['shock_tbills'] = shock_ref
        X_controls = make_lags(tmp[order + ['shock_tbills']], p_lags)
    else:
        X_controls = make_lags(df_lp_eff[order], p_lags)

    # β y SE por variable/horizonte
    betas, ses = {}, {}
    for var in order:
        b, s = lp_irf_single_beta_se(df_lp_eff[var], shock_ref, X_controls, H=H)
        betas[var] = b
        ses[var]   = s

    # construir matrices (H x n), descartando h=0 para el plot
    q50 = np.zeros((H, n))
    lo68 = np.zeros((H, n)); up68 = np.zeros((H, n))
    lo80 = np.zeros((H, n)); up80 = np.zeros((H, n))

    for j, var in enumerate(order):
        b = betas[var][1:H+1]
        s = ses[var][1:H+1]
        q50[:, j]  = b
        lo68[:, j] = b - z68*s;  up68[:, j] = b + z68*s
        lo80[:, j] = b - z80*s;  up80[:, j] = b + z80*s

    return betas, ses, q50, lo68, up68, lo80, up80

def plot_lp_hac(
    q50, lo68, up68, lo80, up80,
    order, titles=None,
    ncols=3, figsize_base=(5,3),
    title='Respuesta a Shock en Tbills (LP) — Bandas HAC',
    smooth=True, smooth_method='savgol', smooth_window=7, smooth_polyorder=2,
    show_original_center=False,
    savepath=None):
    
    """
    Grafica bandas 68%/80% (H x n) y la mediana q50 (H x n).
    """
    n = len(order)
    t = q50.shape[0]
    x = np.arange(1, t+1)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(figsize_base[0]*ncols, figsize_base[1]*nrows),
                             sharex=True)
    axes = np.atleast_2d(axes)
    fig.suptitle(title, fontsize=17, fontweight='bold', y=0.98)

    for j, var in enumerate(order):
        r, c = divmod(j, ncols)
        ax = axes[r, c]

        if smooth:
            lo80_s, med_s, up80_s = smooth_band(lo80[:, j], q50[:, j], up80[:, j],
                                                 method=smooth_method, window=smooth_window,
                                                 polyorder=smooth_polyorder)
            lo68_s, _,     up68_s = smooth_band(lo68[:, j], q50[:, j], up68[:, j],
                                                 method=smooth_method, window=smooth_window,
                                                 polyorder=smooth_polyorder)
        else:
            lo80_s, med_s, up80_s = lo80[:, j], q50[:, j], up80[:, j]
            lo68_s, up68_s        = lo68[:, j], up68[:, j]

        if show_original_center and smooth:
            ax.plot(x, q50[:, j], color='black', alpha=0.25, linewidth=1)

        ax.fill_between(x, lo80_s, up80_s, color='lightgray', label='80 % intervalo' if j==0 else None)
        ax.fill_between(x, lo68_s, up68_s, color='gray',       label='68 % intervalo' if j==0 else None)
        ax.plot(x, med_s, color='black', linewidth=2)

        ax.axhline(0, color='k', lw=1, alpha=0.7)
        ttl = (titles or {}).get(var, var)
        ax.set_title(ttl, fontsize=11, fontweight='bold')

    for k in range(n, nrows*ncols):
        rr, cc = divmod(k, ncols)
        axes[rr, cc].set_visible(False)
    for cc in range(ncols):
        axes[-1, cc].set_xlabel("Horizonte")

    axes[0, 0].legend(loc="upper right", frameon=False)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.show()
    return fig