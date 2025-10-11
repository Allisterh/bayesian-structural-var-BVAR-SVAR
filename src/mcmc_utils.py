import numpy as np 

def diagnose_out(out, expected_P=16, print_report=True):
    """
    Genera un diagnóstico legible del objeto 'out' producido por run_two_chains_and_trim.
    - Reporta modo, número de parámetros, verificación contra expected_P,
      y para cada parámetro: modo de selección, n_draws y métricas claves (si disponibles).
    - Si existe selección global (intersección de bloques), también la reporta.

    Returns
    -------
    report : dict con resúmenes y tablas por parámetro/global.
    """
    rep = {"mode": out.get("mode", None)}

    def _safe_summary_dict(summary):
        # summary es un DataFrame de az.summary para 1 dim (parametro k)
        # devolvemos métricas clave si existen
        try:
            row = summary.iloc[0] if summary.shape[0] >= 1 else None
            if row is None:
                return {}
            d = {}
            for col in ["r_hat", "ess_bulk", "ess_tail", "mcse_mean", "mean", "sd"]:
                if col in summary.columns:
                    d[col] = float(row[col])
            return d
        except Exception:
            return {}

    if rep["mode"] == "per_param":
        per_param = out.get("per_param", [])
        P = len(per_param)
        rep["n_params"] = P
        rep["expected_ok"] = (P == expected_P)

        # resumen por parámetro
        rows = []
        for entry in per_param:
            k = entry.get("param_index", None)
            mode_k = entry.get("mode", None)

            if mode_k == "segment":
                s, e = entry.get("segment_block", (None, None))
                n_draws = (e - s) if (s is not None and e is not None) else None
            else:
                t_star_k = entry.get("cut_index", None)
                # intentar inferir n_draws desde la forma de theta_s1_chain1
                th = entry.get("theta_s1_chain1", None)
                n_draws = int(th.shape[1]) if (th is not None and hasattr(th, "shape")) else None

            # formas típicas disponibles
            shapes = {}
            for key in ["theta_s1_chain1", "theta_s1_chain2",
                        "theta_s2_chain1", "theta_s2_chain2",
                        "theta_s3_chain1", "theta_s3_chain2"]:
                arr = entry.get(key, None)
                shapes[key] = tuple(arr.shape) if (arr is not None and hasattr(arr, "shape")) else None

            summary = entry.get("summary_trim", None)
            metrics = _safe_summary_dict(summary) if summary is not None else {}

            rows.append({
                "param_index": k,
                "mode": mode_k,
                "n_draws_selected": n_draws,
                "segment_block": entry.get("segment_block", None) if mode_k == "segment" else None,
                "cut_index": entry.get("cut_index", None) if mode_k == "t_star" else None,
                "r_hat": metrics.get("r_hat", None),
                "ess_bulk": metrics.get("ess_bulk", None),
                "ess_tail": metrics.get("ess_tail", None),
                "mcse_mean": metrics.get("mcse_mean", None),
                "mean": metrics.get("mean", None),
                "sd": metrics.get("sd", None),
                "shapes": shapes,})

        rep["per_param_table"] = rows
        rep["selection_summary"] = out.get("selection_summary", {})

        # diagnóstico global (si existe intersección)
        global_out = out.get("global", None)
        if global_out is not None:
            gmode = global_out.get("mode", None)
            rep["global"] = {"mode": gmode}
            if gmode == "intersection_segment":
                seg = global_out.get("segment_block", None)
                rep["global"]["segment_block"] = seg
                g_shapes = {}

                for key in ["theta_s1_chain1", "theta_s1_chain2",
                            "theta_s2_chain1", "theta_s2_chain2",
                            "theta_s3_chain1", "theta_s3_chain2"]:
                    arr = global_out.get(key, None)
                    g_shapes[key] = tuple(arr.shape) if (arr is not None and hasattr(arr, "shape")) else None
                rep["global"]["shapes"] = g_shapes
                gsum = global_out.get("summary_trim", None)
                rep["global"]["metrics"] = _safe_summary_dict(gsum)
            else:
                rep["global"]["note"] = "No hay intersección contigua global usable."

        # diagnóstico agregado (si lo calculaste)
        rep["diagnostics"] = out.get("diagnostics", {})

        if print_report:
            print(f"[MODO] per_param  |  P={P}  |  esperado={expected_P}  |  OK={rep['expected_ok']}")
            sel = rep.get("selection_summary", {})
            if sel:
                print(f"  -> with_segment={sel.get('n_with_segment')} | with_t_star={sel.get('n_with_t_star')} | L={sel.get('segment_draws_requested')}")
            for r in rows:
                k = r["param_index"]
                mode_k = r["mode"]
                nd = r["n_draws_selected"]
                msg = f"  Param {k:02d}: mode={mode_k:8s} | draws={nd}"
                if mode_k == "segment":
                    msg += f" | block={r['segment_block']}"
                else:
                    msg += f" | t*={r['cut_index']}"
                if r["r_hat"] is not None:
                    msg += f" | r_hat={r['r_hat']:.3f} | ESS={r['ess_bulk']:.0f}"
                print(msg)
            if rep.get("global"):
                g = rep["global"]
                print(f"[GLOBAL] mode={g.get('mode')}")
                if g.get("mode") == "intersection_segment":
                    print(f"  block={g.get('segment_block')} | shapes={g.get('shapes')}")
                    m = g.get("metrics", {})
                    if m:
                        print(f"  r_hat={m.get('r_hat', None)} | ESS={m.get('ess_bulk', None)} | mcse_mean={m.get('mcse_mean', None)}")

        return rep

    #  Caso: global_t_star 
    elif rep["mode"] == "global_t_star":
        rep["cut_index"] = out.get("cut_index", None)
        rep["windows_rhat"] = out.get("windows_rhat", None)
        rep["diagnostics"] = out.get("diagnostics", {})

        # shapes globales recortados
        g_shapes = {}
        for key in ["theta_s1_chain1", "theta_s1_chain2",
                    "theta_s2_chain1", "theta_s2_chain2",
                    "theta_s3_chain1", "theta_s3_chain2"]:
            arr = out.get(key, None)
            g_shapes[key] = tuple(arr.shape) if (arr is not None and hasattr(arr, "shape")) else None
        rep["shapes"] = g_shapes

        # métricas globales
        gsum = out.get("summary_trim", None)
        rep["metrics"] = _safe_summary_dict(gsum)

        if print_report:
            print("[MODO] global_t_star")
            print(f"  t* = {rep['cut_index']}")
            print(f"  shapes = {g_shapes}")
            if rep["metrics"]:
                print(f"  r_hat={rep['metrics'].get('r_hat', None)} | ESS={rep['metrics'].get('ess_bulk', None)} | mcse_mean={rep['metrics'].get('mcse_mean', None)}")
            diag = rep.get("diagnostics", {})
            if diag:
                print(f"  diag: rhat_total={diag.get('rhat_total')} | good={diag.get('rhat_good')} | bad={diag.get('rhat_bad')} | thr={diag.get('rhat_threshold')}")

        return rep

    else:
        if print_report:
            print("Modo desconocido en 'out'. Claves presentes:", list(out.keys()))
        return rep
    

def build_A0_hybrid_1000_from_out(out, L=1000):
    """
    SOLO A₀ (estructural):
      - Si un parámetro k está en mode='segment': toma los primeros L draws de [start_k, end_k)
      - Si está en mode='t_star': toma los primeros L draws desde t*_k:
    Concatena horizontalmente cadenas 1 y 2 -> shape final (P, 2*L).
    """

    if out.get("mode") != "per_param":
        raise ValueError("Se esperaba out['mode']=='per_param' para el híbrido por parámetro (A0).")

    per_param = out["per_param"]
    P = len(per_param)

    A0_c1_rows, A0_c2_rows = [], []
    for k, entry in enumerate(per_param):
        th1_c1_k = entry.get("theta_s1_chain1")  # (1, Tk_sel)  A0 para cadena 1 YA recortado para k
        th1_c2_k = entry.get("theta_s1_chain2")  # (1, Tk_sel)  A0 para cadena 2 YA recortado para k
        if th1_c1_k is None or th1_c2_k is None:
            raise ValueError(f"Faltan arrays de A0 para el parámetro {k} en out['per_param'].")

        Tk = th1_c1_k.shape[1]
        if Tk < L:
            raise ValueError(f"El parámetro {k} solo tiene {Tk} draws seleccionadas; L={L} no cabe.")

        A0_c1_rows.append(th1_c1_k[:, :L])  # (1, L)
        A0_c2_rows.append(th1_c2_k[:, :L])  # (1, L)

    A0_c1 = np.vstack(A0_c1_rows)              # (P, L)
    A0_c2 = np.vstack(A0_c2_rows)              # (P, L)
    Posterior_A0_MCMC = np.hstack([A0_c1, A0_c2])  # (P, 2*L)

    info = {
        "selection_mode": "A0_hybrid_L_per_param",
        "P": P,
        "L": int(L),
        "note": (
            "Cada parámetro usa su propia ventana (segment o t*_k). "
            "Úsalo para análisis marginal de A₀; NO para objetos conjuntos.")}
    
    return Posterior_A0_MCMC, info


def build_D_B_concat_from_out(out):
    """
    D (Σ) y B (forma reducida) SIN optimización:
      - Si hay bloque global en out['global'], usa esos arrays y concatena c1|c2.
      - Si no, toma la PRIMERA entrada de out['per_param'] que tenga theta_s2_* y theta_s3_*,
        y concatena c1|c2 de ESA entrada. (sin buscar mejor nada)
    Devuelve:
      Posterior_D_MCMC: (n, 2*T_sel)
      Posterior_B_MCMC: (2*T_sel, kB, n)
      info: metadatos mínimos de qué tramo se usó.
    """
    # Preferir bloque global si existe
    g = out.get("global")
    if g and g.get("theta_s2_chain1") is not None:
        D_c1 = g["theta_s2_chain1"]  
        D_c2 = g["theta_s2_chain2"]  
        B_c1 = g["theta_s3_chain1"]  
        B_c2 = g["theta_s3_chain2"] 

        Posterior_D_MCMC = np.hstack([D_c1, D_c2])               
        Posterior_B_MCMC = np.concatenate([B_c1, B_c2], axis=0)  

        info = {"selection_mode": "global_block",
            "segment_block": g.get("segment_block", None),
            "draws_per_chain": int(D_c1.shape[1]),
            "total_draws_concat": int(2 * D_c1.shape[1])}
        return Posterior_D_MCMC, Posterior_B_MCMC, info

    # 2) Sin global: tomar la PRIMERA entrada válida en per_param (sin optimizar)
    if out.get("mode") != "per_param":
        raise ValueError("No hay bloque global y out['mode'] != 'per_param'; no hay de dónde tomar D/B.")

    for entry in out["per_param"]:
        D_c1 = entry.get("theta_s2_chain1")
        D_c2 = entry.get("theta_s2_chain2")
        B_c1 = entry.get("theta_s3_chain1")
        B_c2 = entry.get("theta_s3_chain2")
        if all(x is not None for x in [D_c1, D_c2, B_c1, B_c2]):
            Posterior_D_MCMC = np.hstack([D_c1, D_c2])               
            Posterior_B_MCMC = np.concatenate([B_c1, B_c2], axis=0)  
            info = {
                "selection_mode": "first_per_param_entry",
                "param_index_ref": entry.get("param_index"),
                "mode_ref": entry.get("mode"),
                "segment_block_ref": entry.get("segment_block") if entry.get("mode") == "segment" else None,
                "t_star_ref": entry.get("cut_index") if entry.get("mode") == "t_star" else None,
                "draws_per_chain": int(D_c1.shape[1]),
                "total_draws_concat": int(2 * D_c1.shape[1])}
            
            return Posterior_D_MCMC, Posterior_B_MCMC, info

    raise RuntimeError("No se encontraron arrays D/B en 'out' para concatenar.")