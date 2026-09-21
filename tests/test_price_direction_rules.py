from __future__ import annotations

import numpy as np
import pandas as pd

import tushare_prediction_pipeline as pipeline
from tools.price_direction_rules import price_rule_cache, selector


core = pipeline.prediction_core


def inputs(rows=160):
    positions = np.arange(rows)
    close = 100 * np.cumprod(1 + 0.006 * np.sin(positions * 0.7))
    base = pd.DataFrame({"close": close, "pre_close": np.r_[100, close[:-1]],
                         "open": close * (1 - 0.005 * np.cos(positions * 0.4))})
    labels = (base.close.shift(-1) > base.close).to_numpy(dtype=int)
    cache = core.NestedRuleCache(predictions=np.ones((1, rows), dtype=np.int8),
                                labels=labels, rule_names=["constant"], valid_mask=np.ones(rows, dtype=bool))
    return base, cache


def test_new_rules_are_exact_complements_and_old_cache_is_preserved():
    base, cache = inputs()
    original = cache.predictions.copy()
    result = price_rule_cache(core, base, cache)
    assert result.predictions.shape == (43, 160)
    np.testing.assert_array_equal(cache.predictions, original)
    np.testing.assert_array_equal(result.predictions[:1], original)
    assert (result.predictions[1:, :40] == -1).all()
    for index in range(1, 43, 2):
        valid = result.predictions[index] >= 0
        assert valid.sum() >= 119
        np.testing.assert_array_equal(valid, result.predictions[index + 1] >= 0)
        np.testing.assert_array_equal(result.predictions[index, valid] + result.predictions[index + 1, valid], np.ones(valid.sum()))


def test_price_rules_and_complete_signal_have_causal_prefix_parity():
    base, cache = inputs()
    kwargs = {"idx": 120, "calibration_window": 120, "min_calibration_rows": 30, "top_k": 5}
    full_cache = price_rule_cache(core, base, cache)
    full = selector(core, "current_price_rules")(base=base, cache=cache, **kwargs)
    prefix_cache = core.NestedRuleCache(cache.predictions[:, :121], cache.rule_names, cache.labels[:121], cache.valid_mask[:121])
    prefix = selector(core, "current_price_rules")(base=base.iloc[:121], cache=prefix_cache, **kwargs)
    assert full == prefix
    pd.testing.assert_frame_equal(pd.DataFrame(full_cache.predictions[:, :121]),
                                  pd.DataFrame(price_rule_cache(core, base.iloc[:121], prefix_cache).predictions))
    cache.labels[120:] = 1 - cache.labels[120:]
    base.loc[121:, ["close", "open"]] *= 5
    changed = selector(core, "current_price_rules")(base=base, cache=cache, **kwargs)
    assert full == changed


def test_appended_current_price_rule_can_change_an_uninformative_vote():
    base, cache = inputs()
    expanded = price_rule_cache(core, base, cache)
    # The synthetic target deliberately follows current intraday direction.
    cache.labels[:] = (base.close > base.open).to_numpy(dtype=int)
    compute = selector(core, "current_price_rules")
    changes = 0
    for idx in range(100, 130):
        kwargs = dict(base=base, cache=cache, idx=idx, calibration_window=80, min_calibration_rows=30, top_k=5)
        proposed = compute(**kwargs)
        original = core._nested_volatility_rule_signal(**kwargs)
        changes += proposed.predicted_label != original.predicted_label
    assert changes > 0
    assert "intraday_return>rolling_q0.5" in expanded.rule_names
