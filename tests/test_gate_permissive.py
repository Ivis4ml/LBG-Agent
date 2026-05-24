"""(ii) GateConfig.permissive() preset.

The default GateConfig embodies PROPOSAL §7's locked thresholds and must
remain the source for any H1 claim. The permissive preset is the
experiment-time escape valve documented in the §7 commentary.
"""

from __future__ import annotations

from lbg.gate import GateConfig


def test_default_gate_matches_proposal_thresholds():
    """If you change any default, double-check PROPOSAL.html § 9 first."""
    cfg = GateConfig()
    assert cfg.min_trades == 20
    assert cfg.utility_lcb_alpha == 0.20
    assert cfg.drawdown_regression_factor == 1.15


def test_permissive_is_strictly_looser_than_default():
    default = GateConfig()
    perm = GateConfig.permissive()
    assert perm.min_trades < default.min_trades
    assert perm.utility_lcb_alpha > default.utility_lcb_alpha
    assert perm.drawdown_regression_factor > default.drawdown_regression_factor


def test_permissive_preserves_other_fields():
    """The relaxation must be surgical -- complexity / turnover bounds
    must not silently move."""
    default = GateConfig()
    perm = GateConfig.permissive()
    assert perm.max_turnover == default.max_turnover
    assert perm.max_complexity_delta == default.max_complexity_delta
    assert perm.max_train_val_sharpe_gap == default.max_train_val_sharpe_gap
    assert perm.eps == default.eps


def test_permissive_returns_a_new_instance_each_call():
    """The preset must not be a shared mutable singleton."""
    a = GateConfig.permissive()
    b = GateConfig.permissive()
    assert a is not b
