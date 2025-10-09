import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.lp_utils import *
import math
import matplotlib.pyplot as plt



def _stack_draws(draw_list):
    """Convierte lista de vectores (H+1,) a matriz (R, H+1) sin recalcular nada."""
    A = np.asarray(draw_list)
    if A.ndim == 1:     
        A = A[None, :]
    if A.ndim == 2:
        return A
    return np.vstack(draw_list)


def make_lags(df: pd.DataFrame, L: int):
    cols = {}
    for c in df.columns:
        for l in range(1, L+1):
            cols[f"{c}_L{l}"] = df[c].shift(l)
    return pd.DataFrame(cols, index=df.index)

def align_for_horizon(y: pd.Series, shock: pd.Series, X: pd.DataFrame, h: int, cumulative: bool=False):
    y_dep = (y.shift(-h) - y.shift(-1)) if cumulative else y.shift(-h)
    df = pd.concat([y_dep, shock, X], axis=1).dropna()
    return df.iloc[:,0], df.iloc[:,1], df.iloc[:,2:]


def lp_irf_single(y, shock, X_controls, H=20, hac=True):
    out = []
    for h in range(H+1):
        y_h, s_t, X_t = align_for_horizon(y, shock, X_controls, h, cumulative=False)
        Xreg = sm.add_constant(pd.concat([s_t.rename('shock'), X_t], axis=1))
        res = sm.OLS(y_h, Xreg).fit(cov_type='HAC', cov_kwds={'maxlags': max(h+1,1)}) if hac else sm.OLS(y_h, Xreg).fit()
        out.append(res.params['shock'])
    return np.asarray(out) 


def lp_betas_from_draws(
    order,
    H,
    p_lags,
    datos_var,      # dataframe fuente (para index)
    YY,             # panel usado en el VAR (T x n) ya transformado como en el VAR
    X, Y,           # matrices del VAR para yhat y residuales
    Posterior_B_MCMC,           # (R, k, n)
    Posterior_A0_MCMC,          # (q_params, R)  ó (n*n, R) según tu A0_Mtx
    A0_Mtx,                      # callable: A0_Mtx(theta_r) -> (n x n)
    include_shock_lags=False,
    shock_var_name='tbills',
    R_cap=2000,
    hac=True,
    normalize_shock='sd'         # None | 'sd'
):
    """
    Corre LP para cada draw del SBVAR y devuelve:
      - betas_lp: dict[var] -> list of arrays (R_used elementos, cada uno (H+1,))
      - idx_eff, R_used
    """
    n = len(order)
    df_q = datos_var.copy()
    df_lp = pd.DataFrame(YY, index=df_q.index[:len(YY)], columns=order)

    T_eff = X.shape[0]
    df_lp_eff = df_lp.iloc[-T_eff:].copy()
    idx_eff = df_lp_eff.index

    R_total = Posterior_B_MCMC.shape[0]
    R_used = min(R_cap, R_total)

    IDX_SHOCK = order.index(shock_var_name)
    betas_lp = {y: [] for y in order}

    # preconstruye controles sin shock (comparables con VAR)
    from pandas import concat
    def make_controls(base_df, p):
        return make_lags(base_df, p)

    for r in range(R_used):
        B_r  = Posterior_B_MCMC[r, :, :]
        yhat = X @ B_r
        Y_eff = Y[-T_eff:, :]
        u_r   = Y_eff - yhat

        A0_r  = A0_Mtx(Posterior_A0_MCMC[:, r])
        e_r   = (A0_r @ u_r.T).T

        # serie del shock estructural
        shock_r = pd.Series(e_r[:, IDX_SHOCK], index=idx_eff, name='shock_struct')

        # normalización (importante para comparar con IRF del SBVAR por 1 s.d.)
        if normalize_shock == 'sd':
            sd = shock_r.std(ddof=1)
            if sd > 0:
                shock_r = shock_r / sd

        if include_shock_lags:
            tmp = df_lp_eff.copy()
            tmp['shock_struct'] = shock_r
            X_controls = make_controls(tmp[order + ['shock_struct']], p_lags)
        else:
            X_controls = make_controls(df_lp_eff[order], p_lags)

        # LP por variable objetivo
        for yname in order:
            b_vec = lp_irf_single(df_lp_eff[yname], shock_r, X_controls, H=H, hac=hac)
            betas_lp[yname].append(b_vec)

    return betas_lp, idx_eff, R_used


def lp_bands_from_betas(betas_lp, order, H):
    """
    Devuelve:
      - bands_68: dict[var] -> DataFrame(h, p16, p50, p84)
      - bands_80: dict[var] -> dict(p10, p90 arrays)
    """
    bands_68 = {}
    bands_80 = {}
    for var in order:
        B = _stack_draws(betas_lp[var])        # (R_used, H+1)
        H_eff = min(H+1, B.shape[1])
        B = B[:, :H_eff]
        h = np.arange(H_eff)
        bands_68[var] = pd.DataFrame({
            'h':   h,
            'p16': np.percentile(B, 16, axis=0),
            'p50': np.percentile(B, 50, axis=0),
            'p84': np.percentile(B, 84, axis=0),
        })
        bands_80[var] = {
            'p10': np.percentile(B, 10, axis=0),
            'p90': np.percentile(B, 90, axis=0)}
        
    return bands_68, bands_80


def lp_prepare_plot_mats(bands_68, bands_80, order, H):
    n = len(order)
    q10 = np.zeros((H, n))
    q16 = np.zeros((H, n))
    q50 = np.zeros((H, n))
    q84 = np.zeros((H, n))
    q90 = np.zeros((H, n))

    for j, var in enumerate(order):
        dfq = bands_68[var]
        mask = (dfq['h'].values >= 1) & (dfq['h'].values <= H)
        q16[:, j] = dfq.loc[mask, 'p16'].to_numpy()
        q50[:, j] = dfq.loc[mask, 'p50'].to_numpy()
        q84[:, j] = dfq.loc[mask, 'p84'].to_numpy()
        q10[:, j] = bands_80[var]['p10'][1:H+1]
        q90[:, j] = bands_80[var]['p90'][1:H+1]
    return q10, q16, q50, q84, q90


def plot_lp_percentiles(
    q10, q16, q50, q84, q90, order, titles=None,
    figsize_base=(5, 3), ncols=3, title='Respuesta a Shock en Tbills (LP) — Percentiles',
    smooth=True, smooth_method='savgol', smooth_window=7, smooth_polyorder=2,
    show_original_center=False, savepath=None):
    """
    Grafica bandas 68% y 80% con mediana. Suavizado opcional para visualización.
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
            lo80_s, med_s, up80_s = smooth_band(q10[:, j], q50[:, j], q90[:, j],
                                                 method=smooth_method, window=smooth_window,
                                                 polyorder=smooth_polyorder)
            lo68_s, _,     up68_s = smooth_band(q16[:, j], q50[:, j], q84[:, j],
                                                 method=smooth_method, window=smooth_window,
                                                 polyorder=smooth_polyorder)
        else:
            lo80_s, med_s, up80_s = q10[:, j], q50[:, j], q90[:, j]
            lo68_s, up68_s        = q16[:, j], q84[:, j]

        if show_original_center and smooth:
            ax.plot(x, q50[:, j], color='black', alpha=0.25, linewidth=1)

        ax.fill_between(x, lo80_s, up80_s, color='lightgray', label='80 % intervalo' if j==0 else None)
        ax.fill_between(x, lo68_s, up68_s, color='gray',       label='68 % intervalo' if j==0 else None)
        ax.plot(x, med_s, color='black', linewidth=2)

        ax.axhline(0, color='k', lw=1, alpha=0.7)
        ttl = (titles or {}).get(var, var)
        ax.set_title(ttl, fontsize=11, fontweight='bold')

    # ocultar subplots vacíos
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