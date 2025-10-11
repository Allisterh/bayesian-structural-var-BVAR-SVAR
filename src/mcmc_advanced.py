import arviz as az
import xarray as xr
from contextlib import contextmanager
import numpy as np 
from src.metropolis_sampler import *

@contextmanager
def seeded_default_rng(seed: int):
    """
    Context manager that temporarily overrides NumPy's `np.random.default_rng`
    factory so that calls within the block return a `Generator` seeded with `seed`.

    This is useful when the MCMC routine internally calls `np.random.default_rng()`
    without exposing a seed parameter; the context ensures reproducibility of draws
    made inside your MCMC function.

    Parameters
    ----------
    seed : int
        Seed used to initialize `np.random.PCG64` for the temporary `Generator`.

    Yields
    ------
    None
        Executes the code block with a deterministic `default_rng`; restores the
        original factory upon exit.
    """

    orig = np.random.default_rng

    def _factory():
        return np.random.Generator(np.random.PCG64(seed))
    np.random.default_rng = _factory
    try:
        yield
    finally:
        np.random.default_rng = orig

def as_idata_from_numpy(chain1, chain2, var_name="theta"):
    """
    Build an ArviZ `InferenceData` from two NumPy chains of the same variable.

    Parameters
    ----------
    chain1 : np.ndarray
        Array with shape (T, D): T draws by D parameters for chain 1.
    chain2 : np.ndarray
        Array with shape (T, D): T draws by D parameters for chain 2.
    var_name : str, default "theta"
        Name of the variable to store in the posterior group.

    Returns
    -------
    az.InferenceData
        InferenceData with a `posterior` Dataset containing coordinates
        `chain={0,1}`, `draw=range(T)`, `dim=range(D)` and variable `<var_name>`
        of shape (chain=2, draw=T, dim=D).

    """
    T, D = chain1.shape
    posterior = xr.Dataset(
        {var_name: (("chain", "draw", "dim"),
                    np.stack([chain1, chain2], axis=0))},
        coords={"chain": [0, 1], "draw": np.arange(T), "dim": np.arange(D)})
    return az.InferenceData(posterior=posterior)


def rolling_rhat(idata, var_name="theta", window=10_000, step=2_000, threshold=1.01):
    """
    Compute rank-based R-hat over sliding windows and find the first
    index where the maximum R-hat across dimensions drops below `threshold`.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData containing the target variable in the `posterior` group.
    var_name : str, default "theta"
        Variable name within `posterior` to diagnose.
    window : int, default 10_000
        Window length (number of draws) used for each rolling R-hat calculation.
    step : int, default 2_000
        Step size (in draws) between consecutive windows.
    threshold : float, default 1.01
        Convergence cutoff. Windows with max R-hat below this value are considered stable.

    Returns
    -------
    windows : list of (start, end, rhat_max)
        For each window, the start draw index (inclusive), end index (exclusive),
        and the maximum rank-R-hat across dimensions.
    first_stable : int
        The earliest window start index whose `rhat_max` < `threshold`.
        If none meet the criterion, returns 0.

    Notes
    -----
    - R-hat is computed via `az.rhat(..., method="rank")` to improve robustness.
    - This diagnostic is complementary to mean-stability checks; both should be
      satisfied before discarding warmup.
    """

    post = idata.posterior[var_name]  
    n_draws = post.sizes["draw"]
    windows, first_stable = [], None
    for start in range(0, max(1, n_draws - window + 1), step):
        end = min(n_draws, start + window)
        sub = idata.sel(draw=slice(start, end - 1))
        r = az.rhat(sub, var_names=[var_name], method="rank").to_array().values
        rmax = np.nanmax(r)
        windows.append((start, end, float(rmax)))
        if rmax < threshold and first_stable is None:
            first_stable = start

    return windows, (first_stable or 0)

def burnin_by_cummean(idata, var_name="theta", win=10_000, tol=1e-3):
    """
    Determine the minimal burn-in index where the cumulative mean becomes stable,
    simultaneously across chains and dimensions.

    Stability criterion (applied component-wise over a trailing window `win`):
        |mean_t - mean_{t-win}| / |mean_{t-win}| < tol

    The function returns the *maximum* index across chains/dimensions that
    satisfies the criterion (worst-case), i.e., a conservative burn-in.

    Parameters
    ----------
    idata : az.InferenceData
        InferenceData containing the target variable in the `posterior` group.
    var_name : str, default "theta"
        Variable name within `posterior` to check.
    win : int, default 10_000
        Window length used in the cumulative-mean stability test.
    tol : float, default 1e-3
        Relative tolerance (e.g., 1e-3 = 0.1%).

    Returns
    -------
    int
        Conservative burn-in index `t*`. If the chain is too short (T < 2*win),
        falls back to `T//2`.

    Notes
    -----
    - The test uses relative differences with a small epsilon in the denominator
      to avoid division by zero.
    - This is complementary to R-hat; use both before trimming.

    """
    arr = idata.posterior[var_name].values  # (C, T, D)
    C, T, D = arr.shape
    def first_stable_1d(x, w, tol):
        if T < 2*w: 
            return T//2
        means = np.cumsum(x)/np.arange(1, T+1)
        diffs = np.abs(means[w:] - means[:-w])/(np.abs(means[:-w]) + 1e-12)
        idxs = np.where(diffs < tol)[0]
        return (idxs[0] if len(idxs)>0 else T//2)
    tstars = []

    for c in range(C):
        t_c = max(first_stable_1d(arr[c, :, d], win, tol) for d in range(D))
        tstars.append(t_c)
    return int(max(tstars))

def trim_numpy_chains(cut_idx, theta_s1, theta_s2, theta_s3):

    """
    Apply a synchronous left-trim to three NumPy outputs from  MCMC.

    Expected shapes (as produced by your function with `burning=0`):
      - `theta_s1`: (nalpha, T)  — structural parameters (e.g., A0) by draw
      - `theta_s2`: (n, T)       — variance/scale parameters by draw
      - `theta_s3`: (T, k, n)    — coefficient matrices by draw

    Parameters
    ----------
    cut_idx : int
        Number of initial draws to drop (the burn-in).
    theta_s1 : np.ndarray
        Array of shape (nalpha, T).
    theta_s2 : np.ndarray
        Array of shape (n, T).
    theta_s3 : np.ndarray
        Array of shape (T, k, n).

    Returns
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        The three arrays trimmed consistently from `cut_idx` onward:
        `(theta_s1[:, cut_idx:], theta_s2[:, cut_idx:], theta_s3[cut_idx:, :, :])`.
    """
    return (theta_s1[:, cut_idx:], theta_s2[:, cut_idx:], theta_s3[cut_idx:, :, :])


def _mean_stable_inside_block(idata_block, var_name="theta", tol=1e-3):
    """
    Verifica estabilidad de media dentro del bloque:
    comparamos media de la primera mitad vs segunda mitad (por dim y cadena).
    Exige diff relativa < tol en TODAS las dims.
    """
    x = idata_block.posterior[var_name].values  # (chain, draw, dim)
    C, T, D = x.shape
    mid = T // 2
    m1 = x[:, :mid, :].mean(axis=1)  # (C, D)
    m2 = x[:, mid:, :].mean(axis=1)  # (C, D)
    denom = np.abs(m1) + 1e-12
    rel = np.abs(m2 - m1) / denom     # (C, D)
    return bool(np.all(rel < tol))

def find_stable_block(idata, var_name="theta", segment_draws=2000, rhat_threshold=1.05, mean_tol=1e-3):
    """
    Busca un ÚNICO bloque contiguo de longitud segment_draws tal que:
      1) max R-hat (rank) dentro del bloque < rhat_threshold
      2) medias estables dentro del bloque (1a mitad vs 2a mitad) < mean_tol
    Devuelve (start, end) si lo encuentra; None si no hay.
    """

    T = idata.posterior[var_name].sizes["draw"]
    if segment_draws is None or segment_draws <= 0 or segment_draws > T:
        return None
    for start in range(0, T - segment_draws + 1):
        end = start + segment_draws
        sub = idata.sel(draw=slice(start, end - 1))
        r = az.rhat(sub, var_names=[var_name], method="rank").to_array().values
        rmax = float(np.nanmax(r))
        if rmax >= rhat_threshold:
            continue
        if not _mean_stable_inside_block(sub, var_name=var_name, tol=mean_tol):
            continue
        return (start, end)
    return None

def run_two_chains_and_trim(mcmc_func,
    R,  b , 
    theta_init_chain1,
    theta_init_chain2,
    args_base: dict,
    seed1=123,
    seed2=456,
    rhat_window=10_000,
    rhat_step=2_000,
    rhat_threshold=1.01,
    mean_win=10_000,
    mean_tol=1e-3,
    validator=None , segment_draws=None , enforce_global_contiguous=True,):

    """
    Run two independent MCMC chains with controlled seeds, optionally filter
    invalid draws, compute rolling R-hat and cumulative-mean diagnostics to
    determine a conservative burn-in `t*`, trim all outputs, and return
    diagnostics plus trimmed arrays.

    Parameters
    ----------
    mcmc_func : Callable
        The function implementing your sampler, called as:
        `mcmc_func(R, theta_init, **args_base, b=0, burning=0)`,
        and returning `(theta_s1, theta_s2, theta_s3)` with shapes described
        in `trim_numpy_chains`.
    R : int
        Number of iterations/draws to run per chain (with `burning=0`).
    theta_init_chain1 : np.ndarray
        Initial values for chain 1 (passed through to `mcmc_func`).
    theta_init_chain2 : np.ndarray
        Initial values for chain 2 (passed through to `mcmc_func`).
    args_base : dict
        Keyword arguments forwarded to `mcmc_func` (e.g., `Y, X, M, phi0, S0,
        omega, S2, Vmo, ...`). `b` and `burning` are forced to 0 here.

    rhat_window : int, default 10_000
        Window size for rolling rank-based R-hat.
    rhat_step : int, default 2_000
        Step between windows for R-hat.
    rhat_threshold : float, default 1.01
        Convergence cutoff for R-hat (lower is stricter).
    mean_win : int, default 10_000
        Window length for the cumulative-mean stability test.
    mean_tol : float, default 1e-3
        Relative tolerance for mean stability (e.g., 0.1%).
    validator : callable, optional
        A function `f(t) -> bool` evaluated on draw index `t` that returns
        whether the draw is valid (e.g., stability, PSD, invertibility).
        If provided, invalid draws are filtered out *before* diagnostics,
        synchronously across both chains and all returned arrays.

    Notes
    -----
    - This routine enforces two complementary criteria for trimming: (i) a
      rolling rank-R-hat below `rhat_threshold` and (ii) cumulative-mean
      stability within `mean_tol`. The final burn-in index is the maximum
      of both, yielding a conservative choice.
    - Use stricter settings (smaller threshold, larger window) for higher
      assurance; relax if chains are long and stable.
    """

    # Correr ambas cadenas con semillas controladas
    with seeded_default_rng(seed1):
        print('Empezando el muestro de la primera cadena')
        th1_c1, th2_c1, th3_c1 = mcmc_func(R, theta_init_chain1, **args_base, b=b, burning=0)
    with seeded_default_rng(seed2):
        print('Empezando el muestro de la segunda cadena')
        th1_c2, th2_c2, th3_c2 = mcmc_func(R, theta_init_chain2, **args_base, b=b, burning=0)

    print(f'Emezando el proceso de seleccion de cadenas')

    # Despues de ambas cadenas... Esto solo lo hacemos para posterior de A0
    chain1 = th1_c1.T  
    chain2 = th1_c2.T 
    T = chain1.shape[0]
    P = th1_c1.shape[0]

    # Filtro de draws inválidos
    if validator is not None:
        mask = np.array([validator(t) for t in range(T)], dtype=bool)
        chain1, chain2 = chain1[mask], chain2[mask]
        th1_c1, th2_c1, th3_c1 = th1_c1[:, mask], th2_c1[:, mask], th3_c1[mask, :, :]
        th1_c2, th2_c2, th3_c2 = th1_c2[:, mask], th2_c2[:, mask], th3_c2[mask, :, :]
        T = mask.sum()

    idata = as_idata_from_numpy(chain1, chain2, var_name="theta")

    # =========================
    # MODO 1 (PRIORITARIO): BLOQUES CONTIGUOS POR PARÁMETRO (con fallback t* por parámetro)
    # =========================

    if segment_draws is not None:
        per_param = []
        n_segment_ok = 0
        blocks = []

        for k in range(P):
            # idata del parámetro k con todas las draws (ambas cadenas)
            c1_k = th1_c1[k, :][None, :].T  # (T,1)
            c2_k = th1_c2[k, :][None, :].T
            idata_k = as_idata_from_numpy(c1_k, c2_k, var_name="theta")

            entry = {"param_index": int(k)}

            # Intentar bloque contiguo estable
            chosen_block = find_stable_block(
                idata_k,
                var_name="theta",
                segment_draws=segment_draws,
                rhat_threshold=rhat_threshold,
                mean_tol=mean_tol)

            if chosen_block is not None:
                start, end = chosen_block
                entry.update({"mode": "segment",
                    "segment_block": (int(start), int(end)),
                    "cut_index": int(start)})

                # Recortes por parámetro (NO mezclamos ni sobrescribimos)
                th1_c1_blk = th1_c1[k:k+1, start:end]
                th2_c1_blk = th2_c1[:, start:end]
                th3_c1_blk = th3_c1[start:end, :, :]

                th1_c2_blk = th1_c2[k:k+1, start:end]
                th2_c2_blk = th2_c2[:, start:end]
                th3_c2_blk = th3_c2[start:end, :, :]

                # idata y resumen por parámetro
                chain1_blk = th1_c1_blk.T
                chain2_blk = th1_c2_blk.T
                idata_trim_k = as_idata_from_numpy(chain1_blk, chain2_blk, var_name="theta")
                summary_final_k = az.summary(idata_trim_k, var_names=["theta"], kind="stats", round_to=4)

                entry.update({
                    "idata_trim": idata_trim_k,
                    "summary_trim": summary_final_k,
                    "theta_s1_chain1": th1_c1_blk,
                    "theta_s2_chain1": th2_c1_blk,
                    "theta_s3_chain1": th3_c1_blk,
                    "theta_s1_chain2": th1_c2_blk,
                    "theta_s2_chain2": th2_c2_blk,
                    "theta_s3_chain2": th3_c2_blk})

                n_segment_ok += 1
                blocks.append((int(start), int(end)))

            else:
                # Fallback SOLO para este parámetro: corte único t*_k
                windows_k, t_rhat_k = rolling_rhat(
                    idata_k, var_name="theta",
                    window=rhat_window, step=rhat_step, threshold=rhat_threshold)
                t_mean_k = burnin_by_cummean(idata_k, var_name="theta", win=mean_win, tol=mean_tol)
                t_star_k = int(max(int(t_rhat_k), int(t_mean_k), 0))

                entry.update({
                    "mode": "t_star",
                    "cut_index": int(t_star_k)})

                # Recortes por parámetro con t*_k
                th1_c1_trim = th1_c1[k:k+1, t_star_k:]
                th2_c1_trim = th2_c1[:, t_star_k:]
                th3_c1_trim = th3_c1[t_star_k:, :, :]

                th1_c2_trim = th1_c2[k:k+1, t_star_k:]
                th2_c2_trim = th2_c2[:, t_star_k:]
                th3_c2_trim = th3_c2[t_star_k:, :, :]

                chain1_trim_k = th1_c1_trim.T
                chain2_trim_k = th1_c2_trim.T
                idata_trim_k = as_idata_from_numpy(chain1_trim_k, chain2_trim_k, var_name="theta")
                summary_final_k = az.summary(idata_trim_k, var_names=["theta"], kind="stats", round_to=4)

                entry.update({
                    "idata_trim": idata_trim_k,
                    "summary_trim": summary_final_k,
                    "theta_s1_chain1": th1_c1_trim,
                    "theta_s2_chain1": th2_c1_trim,
                    "theta_s3_chain1": th3_c1_trim,
                    "theta_s1_chain2": th1_c2_trim,
                    "theta_s2_chain2": th2_c2_trim,
                    "theta_s3_chain2": th3_c2_trim})
            per_param.append(entry)

        # Intentar GLOBAL sólo si TODOS lograron bloque y existe intersección no vacía
        global_out = None
        if n_segment_ok == P:
            max_start = max(s for s, e in blocks)
            min_end   = min(e for s, e in blocks)
            if min_end > max_start:
                # Intersección contigua común [max_start, min_end)
                s, e = int(max_start), int(min_end)
                chain1_trim = th1_c1[:, s:e].T
                chain2_trim = th1_c2[:, s:e].T
                idata_trim_global = as_idata_from_numpy(chain1_trim, chain2_trim, var_name="theta")
                summary_global = az.summary(idata_trim_global, var_names=["theta"], kind="stats", round_to=4)

                global_out = {
                    "mode": "intersection_segment",
                    "segment_block": (s, e),
                    "idata_trim": idata_trim_global,
                    "summary_trim": summary_global,
                    "theta_s1_chain1": th1_c1[:, s:e],
                    "theta_s2_chain1": th2_c1[:, s:e],
                    "theta_s3_chain1": th3_c1[s:e, :, :],
                    "theta_s1_chain2": th1_c2[:, s:e],
                    "theta_s2_chain2": th2_c2[:, s:e],
                    "theta_s3_chain2": th3_c2[s:e, :, :]}
            else:
                print("ℹ️ Todos los parámetros tienen bloque, pero no hay intersección contigua común.")
                global_out = {"mode": "no_global_intersection"}

        # Reporte R-hat global (si hay selección global)
        diagnostics = {}
        if global_out is not None and "idata_trim" in global_out:
            rhat_vals = az.rhat(global_out["idata_trim"], method="rank")
            rhat_array = rhat_vals.to_array().values.flatten()
            threshold = max(1.01, rhat_threshold)
            n_total = rhat_array.size
            n_good = int(np.sum(rhat_array < threshold))
            n_bad = n_total - n_good
            diagnostics = {
                "rhat_total": int(n_total),
                "rhat_good": int(n_good),
                "rhat_bad": int(n_bad),
                "rhat_threshold": float(threshold)}

        out = {
            "mode": "per_param",
            "idata_full": idata,
            "per_param": per_param,              # <<< resultados independientes por parámetro
            "global": global_out,                # <<< sólo si todos tuvieron bloque y hay intersección
            "selection_summary": {
                "n_params": int(P),
                "n_with_segment": int(n_segment_ok),
                "n_with_t_star": int(P - n_segment_ok),
                "segment_draws_requested": int(segment_draws)},
            "diagnostics": diagnostics}

        return out

    # =========================
    # MODO 2 (FALLBACK GLOBAL): CORTE ÚNICO t* (comportamiento original)
    # =========================
    windows, t_rhat = rolling_rhat(idata, var_name="theta",
        window=rhat_window, step=rhat_step, threshold=rhat_threshold)
    
    t_mean = burnin_by_cummean(idata, var_name="theta", win=mean_win, tol=mean_tol)
    t_star = int(max(t_rhat, t_mean, 0))

    th1_c1_trim, th2_c1_trim, th3_c1_trim = trim_numpy_chains(t_star, th1_c1, th2_c1, th3_c1)
    th1_c2_trim, th2_c2_trim, th3_c2_trim = trim_numpy_chains(t_star, th1_c2, th2_c2, th3_c2)

    chain1_trim = th1_c1_trim.T
    chain2_trim = th1_c2_trim.T
    idata_trim = as_idata_from_numpy(chain1_trim, chain2_trim, var_name="theta")

    summary_final = az.summary(idata_trim, var_names=["theta"], kind="stats", round_to=4)

    # Diagnóstico R-hat final 
    rhat_vals = az.rhat(idata_trim, method="rank")
    rhat_array = rhat_vals.to_array().values.flatten()
    threshold = max(1.01, rhat_threshold)
    n_total = rhat_array.size
    n_good = int(np.sum(rhat_array < threshold))
    n_bad = n_total - n_good

    print("Diagnóstico R-hat en la selección final (t* global):")
    print(f" Total de parámetros: {n_total}")
    print(f"Convergieron (<{threshold}): {n_good}")
    print(f"No convergieron (≥{threshold}): {n_bad}")
    if n_bad > 0:
        idx_bad = np.where(rhat_array >= threshold)[0]
        print(f"Parámetros sin convergencia (índices): {idx_bad.tolist()}")

    out = {
        "mode": "global_t_star",
        "cut_index": t_star,
        "windows_rhat": windows,
        "idata_full": idata,
        "idata_trim": idata_trim,
        "summary_trim": summary_final,
        "theta_s1_chain1": th1_c1_trim,
        "theta_s2_chain1": th2_c1_trim,
        "theta_s3_chain1": th3_c1_trim,
        "theta_s1_chain2": th1_c2_trim,
        "theta_s2_chain2": th2_c2_trim,
        "theta_s3_chain2": th3_c2_trim,
        "diagnostics": {
            "rhat_total": int(n_total),
            "rhat_good": int(n_good),
            "rhat_bad": int(n_bad),
            "rhat_threshold": float(threshold)}}

    return out

