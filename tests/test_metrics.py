import numpy as np
from drift_radar.app.metrics import psi, js_divergence, ks_from_hist


def test_psi_zero_for_identical():
    a = np.array([0.2, 0.3, 0.5])
    assert psi(a, a) == 0.0


def test_jsd_zero_for_identical():
    a = np.array([0.2, 0.3, 0.5])
    assert js_divergence(a, a) == 0.0


def test_ks_from_hist_high_pvalue_for_similar():
    bins = [0, 1, 2, 3]
    a = np.array([0.3, 0.4, 0.3])
    p = ks_from_hist(bins, a, a)
    assert p > 0.9
