import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def prices():
    """Six synthetic assets, about five years of business days, different volatilities."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1300)
    vols = np.array([0.05, 0.08, 0.12, 0.16, 0.20, 0.30]) / np.sqrt(252)
    common = rng.normal(0, 1, len(idx))[:, None]
    r = 0.0003 + vols * (0.5 * common + rng.normal(0, 1, (len(idx), 6)))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=list("ABCDEF"))
