"""Agreement statistics for paired measurements.

Used for the prone/supine pillar. Kept here rather than in a script so it can be
unit-tested against published worked examples, which matters because an ICC
reported with the wrong model is a silent error: the three models differ by a
factor of two on the same data.

Model used is **ICC(2,1)** in Shrout and Fleiss's notation -- two-way random
effects, absolute agreement, single measurement. That is the right choice here
because the two positions are not interchangeable replicates of one another and
a systematic offset between them *should* count against agreement. ICC(3,1),
which treats the raters as fixed and ignores such an offset, would flatter the
result.

References:
Shrout PE, Fleiss JL. Intraclass correlations: uses in assessing rater
reliability. Psychol Bull 1979;86:420-428.
McGraw KO, Wong SP. Forming inferences about some intraclass correlation
coefficients. Psychol Methods 1996;1:30-46.
Bland JM, Altman DG. Statistical methods for assessing agreement between two
methods of clinical measurement. Lancet 1986;1:307-310.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ICCResult", "BlandAltmanResult", "icc21", "bland_altman"]


@dataclass
class ICCResult:
    icc: float
    lower: float
    upper: float
    n: int
    k: int
    confidence: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"ICC(2,1) = {self.icc:.3f} "
            f"({self.confidence:.0%} CI {self.lower:.3f}-{self.upper:.3f}), n={self.n}"
        )


@dataclass
class BlandAltmanResult:
    bias: float
    sd_diff: float
    lower_loa: float
    upper_loa: float
    n: int
    bias_ci: tuple[float, float]
    mean_values: np.ndarray
    differences: np.ndarray


def icc21(ratings: np.ndarray, confidence: float = 0.95) -> ICCResult:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement.

    ``ratings`` is ``(n_subjects, k_raters)`` with no missing values. For the
    prone/supine analysis the two columns are the two positions.
    """
    from scipy import stats

    x = np.asarray(ratings, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("ratings must be (n_subjects >= 2, k_raters >= 2)")
    if not np.isfinite(x).all():
        raise ValueError("ratings must not contain NaN or inf")

    n, k = x.shape
    grand = x.mean()
    row_means = x.mean(axis=1)
    col_means = x.mean(axis=0)

    # Mean squares: rows (subjects), columns (raters), residual.
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_total = ((x - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols

    ms_r = ss_rows / (n - 1)
    ms_c = ss_cols / (k - 1)
    ms_e = ss_err / ((n - 1) * (k - 1))

    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    icc = (ms_r - ms_e) / denom if denom != 0 else float("nan")

    # McGraw & Wong (1996), ICC(A,1) confidence interval.
    alpha = 1.0 - confidence
    if ms_e <= 0:
        # Perfect agreement: the residual is zero, the interval degenerates and
        # the F ratio is not finite. Report the point estimate as the interval
        # rather than propagating inf/inf through the formula below.
        return ICCResult(
            icc=float(icc), lower=float(icc), upper=float(icc),
            n=n, k=k, confidence=confidence,
        )

    # Satterthwaite degrees of freedom of McGraw & Wong (1996), case 2A --
    # equivalently Shrout & Fleiss (1979) with F_j = MS_columns / MS_error.
    # (Until 2026-09-12 this used MS_rows / MS_error, which understated the
    # lower bound; the Shrout & Fleiss worked example now pins the interval.)
    a_ = k * icc / (n * (1.0 - icc)) if icc < 1 else float("inf")
    b_ = 1.0 + k * icc * (n - 1) / (n * (1.0 - icc)) if icc < 1 else float("inf")
    num = (a_ * ms_c + b_ * ms_e) ** 2
    den = (a_ * ms_c) ** 2 / (k - 1) + (b_ * ms_e) ** 2 / ((n - 1) * (k - 1))
    v = num / den if np.isfinite(num) and den > 0 else (n - 1) * (k - 1)

    f_lower = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f_upper = stats.f.ppf(1 - alpha / 2, v, n - 1)
    lower = n * (ms_r - f_lower * ms_e) / (
        f_lower * (k * ms_c + (k * n - k - n) * ms_e) + n * ms_r
    )
    upper = n * (f_upper * ms_r - ms_e) / (
        k * ms_c + (k * n - k - n) * ms_e + n * f_upper * ms_r
    )

    return ICCResult(
        icc=float(icc),
        lower=float(np.clip(lower, -1.0, 1.0)),
        upper=float(np.clip(upper, -1.0, 1.0)),
        n=n,
        k=k,
        confidence=confidence,
    )


def bland_altman(a: np.ndarray, b: np.ndarray, loa_z: float = 1.96) -> BlandAltmanResult:
    """Bias and 95 % limits of agreement for two paired measurements.

    The difference is ``a - b``, so the sign follows the order of the arguments.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("a and b must have the same shape")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2:
        raise ValueError("need at least two complete pairs")

    diff = x - y
    mean = (x + y) / 2.0
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    n = int(diff.size)
    se_bias = sd / np.sqrt(n)

    from scipy import stats

    t = stats.t.ppf(0.975, n - 1)
    return BlandAltmanResult(
        bias=bias,
        sd_diff=sd,
        lower_loa=bias - loa_z * sd,
        upper_loa=bias + loa_z * sd,
        n=n,
        bias_ci=(bias - t * se_bias, bias + t * se_bias),
        mean_values=mean,
        differences=diff,
    )
