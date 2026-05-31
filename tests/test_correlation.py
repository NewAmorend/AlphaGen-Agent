from __future__ import annotations
from wq_agent.config import Settings


def test_self_corr_settings_defaults():
    s = Settings(_env_file=None)
    assert s.SELF_CORR_THRESHOLD == 0.7
    assert s.SELF_CORR_SHARPE_MARGIN == 0.10
    assert s.SELF_CORR_MIN_OVERLAP == 60
