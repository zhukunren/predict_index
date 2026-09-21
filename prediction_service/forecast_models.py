"""Fixed production and shadow algorithms, independent of web and scheduling code.

These are the selected historical experiments, with their original timing,
training masks, thresholds and calibration. There is no direction quota.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import tushare_prediction_pipeline as pipeline
from tools.downside_specialist import (
    specialist_features, downside_probabilities, mean_downside_probabilities,
    specialist_predictions,
)
from tools.moneyflow_features import FEATURE_COLUMNS as FLOW_COLUMNS, PRICE_FEATURE_COLUMNS, moneyflow_features
from tools.option_features import FEATURE_COLUMNS as OPTION_COLUMNS, option_features
from tools.option_position_features import FEATURE_COLUMNS as POSITION_COLUMNS, position_features


@dataclass(frozen=True)
class ForecastModel:
    key: str
    name: str
    experiment: str | None
    threshold: float | None


MODELS = (
    ForecastModel("option_moneyflow", "期权＋资金流组合", "option_moneyflow_ensemble_v1", 0.55),
    ForecastModel("baseline", "原生产基线", None, None),
    ForecastModel("moneyflow", "资金流信号", "moneyflow_downside_v1", 0.60),
    ForecastModel("moneyflow_price", "资金流＋价格特征组合", "ensemble_moneyflow_price_downside_v1", 0.55),
)
PRODUCTION_KEY = "option_moneyflow"
MODEL_KEYS = tuple(model.key for model in MODELS)
CONTEXT_NAMES = ("breadth", "moneyflow", "options", "spx", "nasdaq")
PARITY_COLUMNS = (
    "trade_date", "predicted_label", "predicted_pct_change", "predicted_close",
    "confidence", "calibrated_confidence", "return_calibration_scale",
)


def assert_parity(expected, actual):
    aligned = actual.set_index("trade_date").reindex(expected.trade_date).reset_index()
    pd.testing.assert_frame_equal(
        expected.loc[:, PARITY_COLUMNS].reset_index(drop=True),
        aligned.loc[:, PARITY_COLUMNS], check_exact=True, check_dtype=False,
    )


def calculate_models(market, baseline, context):
    """Calculate all four streams once on an identical chronological baseline."""
    config, _ = pipeline._default_calculation_options()
    core = pipeline.prediction_core
    common = specialist_features(core, market, baseline, config, context["breadth"],
                                {name: context[name] for name in ("spx", "nasdaq")})
    calendar = pd.to_datetime(market.trade_date).dt.strftime("%Y%m%d").astype(int)
    flow = moneyflow_features(baseline.trade_date, context["moneyflow"], calendar, price_context=True)
    flow_features = pd.concat([common, flow.loc[:, FLOW_COLUMNS]], axis=1)
    price_features = pd.concat([flow_features, flow.loc[:, PRICE_FEATURE_COLUMNS]], axis=1)
    flow_mask = flow.moneyflow_available.to_numpy()
    flow_probability = downside_probabilities(baseline, flow_features, "downside_logistic", available=flow_mask)
    price_probability = downside_probabilities(baseline, price_features, "downside_logistic", available=flow_mask)

    # The selected option experiment uses current-session activity AND positions.
    options = option_features(baseline.trade_date, context["options"], calendar,
                              lag_sessions=0, publication_hour=20)
    positions = position_features(baseline.trade_date, context["options"], calendar,
                                 lag_sessions=0, publication_hour=20)
    shared_mask = flow_mask & positions.option_position_available.to_numpy()
    option_features_only = pd.concat([common, options.loc[:, OPTION_COLUMNS],
                                     positions.loc[:, POSITION_COLUMNS]], axis=1)
    # This mask is deliberately distinct from the standalone moneyflow model.
    paired_flow = downside_probabilities(baseline, flow_features, "downside_logistic", available=shared_mask)
    option_probability = downside_probabilities(baseline, option_features_only, "downside_logistic", available=shared_mask)
    combined = mean_downside_probabilities(paired_flow, option_probability).rename(columns={
        "downside_base_probability": "downside_moneyflow_probability",
        "downside_extended_probability": "downside_option_probability",
    })
    return {
        "baseline": baseline.copy(),
        "moneyflow": specialist_predictions(core, baseline, flow_probability, 0.60),
        "moneyflow_price": specialist_predictions(core, baseline, mean_downside_probabilities(flow_probability, price_probability), 0.55),
        "option_moneyflow": specialist_predictions(core, baseline, combined, 0.55),
    }
