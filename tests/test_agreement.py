"""Tests for ctc_core.agreement.

The ICC is checked against the worked example in Shrout & Fleiss (1979), whose
published values for that dataset are ICC(1,1) = 0.17, ICC(2,1) = 0.29 and
ICC(3,1) = 0.71. Reporting the wrong model is a silent error -- they differ by
more than a factor of two on the same numbers -- so a published target matters
more here than a self-consistent one.
"""

import numpy as np
import pytest

from ctc_core.agreement import bland_altman, icc21

# Shrout & Fleiss (1979), Table 1: 6 targets rated by 4 judges.
SHROUT_FLEISS = np.array(
    [
        [9, 2, 5, 8],
        [6, 1, 3, 2],
        [8, 4, 6, 8],
        [7, 1, 2, 6],
        [10, 5, 6, 9],
        [6, 2, 4, 7],
    ],
    dtype=float,
)


def test_icc21_matches_the_published_worked_example():
    res = icc21(SHROUT_FLEISS)
    assert res.icc == pytest.approx(0.290, abs=0.005)
    assert res.n == 6
    assert res.k == 4


def test_icc21_interval_matches_the_shrout_fleiss_formula():
    """The interval, computed independently from Shrout & Fleiss (1979).

    Their ICC(2,1) interval uses Satterthwaite degrees of freedom built from
    F_j = JMS / EMS, the *judges* mean square. An earlier version used the
    targets' mean square instead and understated every lower bound; on this
    dataset the correct interval is [0.019, 0.761].
    """
    from scipy import stats

    x = SHROUT_FLEISS
    n, k = x.shape
    g, r, c = x.mean(), x.mean(axis=1), x.mean(axis=0)
    bms = k * ((r - g) ** 2).sum() / (n - 1)
    jms = n * ((c - g) ** 2).sum() / (k - 1)
    ems = (((x - g) ** 2).sum() - k * ((r - g) ** 2).sum() - n * ((c - g) ** 2).sum()) / (
        (n - 1) * (k - 1))
    icc = (bms - ems) / (bms + (k - 1) * ems + k * (jms - ems) / n)
    fj = jms / ems
    v = (k - 1) * (n - 1) * (k * icc * fj + n * (1 + (k - 1) * icc) - k * icc) ** 2 / (
        (n - 1) * k**2 * icc**2 * fj**2 + (n * (1 + (k - 1) * icc) - k * icc) ** 2)
    fs, fss = stats.f.ppf(0.975, n - 1, v), stats.f.ppf(0.975, v, n - 1)
    lo = n * (bms - fs * ems) / (fs * (k * jms + (k * n - k - n) * ems) + n * bms)
    hi = n * (fss * bms - ems) / (k * jms + (k * n - k - n) * ems + n * fss * bms)

    res = icc21(x)
    assert res.lower == pytest.approx(lo, abs=1e-6)
    assert res.upper == pytest.approx(hi, abs=1e-6)
    assert res.lower == pytest.approx(0.019, abs=0.002)
    assert res.upper == pytest.approx(0.761, abs=0.002)


def test_icc21_confidence_interval_brackets_the_estimate():
    res = icc21(SHROUT_FLEISS)
    assert res.lower < res.icc < res.upper
    assert -1.0 <= res.lower <= 1.0
    assert -1.0 <= res.upper <= 1.0


def test_icc21_is_one_for_identical_columns():
    x = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0], [3.0, 3.0]])
    assert icc21(x).icc == pytest.approx(1.0, abs=1e-9)


def test_icc21_penalises_a_systematic_offset():
    """This is why ICC(2,1) and not ICC(3,1): a constant bias must count."""
    base = np.array([1.0, 5.0, 9.0, 3.0, 7.0])
    perfect = np.column_stack([base, base])
    shifted = np.column_stack([base, base + 3.0])

    assert icc21(perfect).icc == pytest.approx(1.0, abs=1e-9)
    assert icc21(shifted).icc < 0.9


def test_icc21_is_near_zero_for_unrelated_columns():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 2))
    assert abs(icc21(x).icc) < 0.35


def test_icc21_rejects_bad_input():
    with pytest.raises(ValueError):
        icc21(np.array([1.0, 2.0, 3.0]))  # not 2-D
    with pytest.raises(ValueError):
        icc21(np.array([[1.0, 2.0]]))  # one subject
    with pytest.raises(ValueError):
        icc21(np.array([[1.0, np.nan], [2.0, 3.0]]))


# ---------------------------------------------------------------------------
# Bland-Altman
# ---------------------------------------------------------------------------


def test_bland_altman_recovers_a_known_bias_and_spread():
    rng = np.random.default_rng(1)
    truth = rng.normal(100.0, 20.0, size=500)
    a = truth + rng.normal(0.0, 2.0, size=500)
    b = truth + rng.normal(-5.0, 2.0, size=500)  # b reads 5 low

    res = bland_altman(a, b)
    assert res.bias == pytest.approx(5.0, abs=0.5)
    assert res.sd_diff == pytest.approx(np.sqrt(2) * 2.0, abs=0.5)
    assert res.lower_loa < res.bias < res.upper_loa
    assert res.n == 500


def test_bland_altman_sign_follows_argument_order():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])
    assert bland_altman(a, b).bias == pytest.approx(-1.0)
    assert bland_altman(b, a).bias == pytest.approx(1.0)


def test_bland_altman_limits_cover_about_95_percent():
    rng = np.random.default_rng(2)
    a = rng.normal(size=4000)
    b = a + rng.normal(0.0, 1.0, size=4000)
    res = bland_altman(a, b)
    inside = np.mean((res.differences >= res.lower_loa) & (res.differences <= res.upper_loa))
    assert inside == pytest.approx(0.95, abs=0.02)


def test_bland_altman_drops_incomplete_pairs():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 3.0, 3.0, 5.0])
    res = bland_altman(a, b)
    assert res.n == 3


def test_bland_altman_rejects_too_few_pairs():
    with pytest.raises(ValueError):
        bland_altman(np.array([1.0]), np.array([2.0]))
    with pytest.raises(ValueError):
        bland_altman(np.array([1.0, 2.0]), np.array([1.0]))
