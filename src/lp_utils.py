from scipy.signal import savgol_filter
import numpy as np 


def _odd_leq(n):
    return n if n % 2 == 1 else n-1

def _choose_window(H, base=7):
    # ventana impar, al menos 3 y <= H-1
    w = max(3, base)
    w = _odd_leq(min(w, H-1))
    if w < 3: w = 3
    if w >= H: w = _odd_leq(H-1)
    return w

def smooth_1d(y, method="savgol", window=None, polyorder=2):
    y = np.asarray(y, float)
    H = y.size
    if H < 5:
        return y.copy()
    if method == "savgol":
        w = _choose_window(H, base=(window or 7))
        poly = min(polyorder, w-1)  # debe ser < window
        ys = savgol_filter(y, window_length=w, polyorder=poly, mode="interp")
        ys[0] = y[0]                 # fijar h=0
        return ys
    elif method == "ma":
        w = window or 5
        w = max(3, w)
        if w > H: w = H
        k = w//2
        pad = (k, w-1-k)
        ypad = np.pad(y, pad, mode="edge")
        c = np.convolve(ypad, np.ones(w)/w, mode="valid")
        c[0] = y[0]
        return c
    else:
        return y.copy()

def smooth_band(lo, med, up, **kwargs):
    lo_s  = smooth_1d(lo,  **kwargs)
    med_s = smooth_1d(med, **kwargs)
    up_s  = smooth_1d(up,  **kwargs)
    # Asegurar orden de bandas
    lo_s  = np.minimum.reduce([lo_s, med_s, up_s])
    up_s  = np.maximum.reduce([lo_s, med_s, up_s])
    med_s = np.clip(med_s, lo_s, up_s)
    return lo_s, med_s, up_s

