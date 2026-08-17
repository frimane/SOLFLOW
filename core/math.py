# ===== SECTION: mathutil ====================================================
# ============================================================================
# Standard normal CDF and inverse CDF, vectorized, no scipy. The CDF uses the
# Abramowitz & Stegun 7.1.26 erf approximation (abs error < 1.5e-7); the
# inverse uses Acklam's rational approximation refined by one Halley step,
# which is far below the noise floor of anything we score here.

def norm_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    x = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
              + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-x * x)
    return 0.5 * (1.0 + np.sign(z) * erf)


def norm_ppf(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    p = np.clip(p, 1e-300, 1.0 - 1e-16)
    x = np.empty_like(p)

    lo = p < plow
    hi = p > phigh
    mid = ~(lo | hi)

    if lo.any():
        q = np.sqrt(-2.0 * np.log(p[lo]))
        x[lo] = ((((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0))
    if hi.any():
        q = np.sqrt(-2.0 * np.log(1.0 - p[hi]))
        x[hi] = -((((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                  ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0))
    if mid.any():
        q = p[mid] - 0.5
        r = q * q
        x[mid] = ((((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q /
                  (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0))

    # one Halley refinement step against the accurate CDF
    e = norm_cdf(x) - p
    u = e * np.sqrt(2.0 * np.pi) * np.exp(0.5 * x * x)
    x = x - u / (1.0 + 0.5 * x * u)
    return x


# ============================================================================
