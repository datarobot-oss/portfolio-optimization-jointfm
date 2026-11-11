# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The JointFM sampler: request contents, return conversion, and refusals."""

import numpy as np
import pytest
from portfolio_optimization.settings import ReturnsComputeSettings
from portfolio_optimization.utils import calculate_returns

from portfolio_optimization_jointfm import (
    JointFMSettings,
    generate_horizon_samples_jointfm,
    generate_samples_jointfm,
)

from .conftest import TICKERS


def _sample(returns_dict, return_type="LOG", freq=1, num_scen=25, seed=3):
    return generate_samples_jointfm(
        num_scen, returns_dict["returns"], return_type, freq, JointFMSettings(), seed
    )


def _assert_log_prices_up_to_constant(history, prices):
    """The history is the log of ``prices`` shifted by one constant per asset."""
    assert list(history.columns) == list(prices.columns)
    offsets = history.to_numpy() - np.log(prices.to_numpy())
    np.testing.assert_allclose(
        offsets, np.broadcast_to(offsets[0], offsets.shape), rtol=0, atol=1e-12
    )


def test_one_joint_next_step_request(returns_dict, fake_jointfm):
    """All assets go into one sample request for the step after the history."""
    _sample(returns_dict, num_scen=25, seed=11)
    (call,) = fake_jointfm.forecast_calls
    assert call["n_samples"] == 25
    assert call["seed"] == 11
    assert call["return_mode"] == "samples"
    assert call["target_columns"] == TICKERS
    assert call["requested_columns"] == TICKERS
    assert call["query_times"] == [len(call["history"])]


@pytest.mark.parametrize("return_type", ["LOG", "LINEAR"])
def test_history_is_log_price_path(price_data, fake_jointfm, return_type):
    """Returns are sent as the log-price path they imply, starting from 0."""
    returns_dict = calculate_returns(
        price_data,
        regime_dict=None,
        returns_compute_settings=ReturnsComputeSettings(
            return_type=return_type, freq=1
        ),
    )
    fake_jointfm.envelope.n_input = 1000
    _sample(returns_dict, return_type=return_type)
    history = fake_jointfm.forecast_calls[0]["history"]
    _assert_log_prices_up_to_constant(history, price_data)
    np.testing.assert_array_equal(history.iloc[0], 0.0)


def test_history_cut_to_n_input(price_data, returns_dict, fake_jointfm):
    """Only the most recent n_input points of the path are sent."""
    n_input = fake_jointfm.envelope.n_input
    assert len(returns_dict["returns"]) > n_input
    _sample(returns_dict)
    history = fake_jointfm.forecast_calls[0]["history"]
    _assert_log_prices_up_to_constant(history, price_data.iloc[-n_input:])


@pytest.mark.parametrize("return_type", ["LOG", "LINEAR"])
def test_forecast_log_price_becomes_next_return(
    returns_dict, fake_jointfm, return_type
):
    """A draw minus the last log price is the next log return, converted."""
    samples = _sample(returns_dict, return_type=return_type)
    (call,) = fake_jointfm.forecast_calls
    last_log_prices = call["history"].iloc[-1].to_numpy()
    assert np.all(np.abs(last_log_prices) > 1e-3)
    next_log_returns = call["samples"][:, 0, :] - last_log_prices
    expected = next_log_returns if return_type == "LOG" else np.expm1(next_log_returns)
    np.testing.assert_allclose(samples, expected)
    assert samples.shape == (25, len(TICKERS))
    assert samples.flags.writeable


@pytest.mark.parametrize("return_type", ["LOG", "LINEAR"])
def test_horizon_samples_are_returns_over_each_horizon(
    returns_dict, fake_jointfm, return_type
):
    """Each horizon's draw minus the last log price is its h-period return."""
    samples = generate_horizon_samples_jointfm(
        25, returns_dict["returns"], return_type, 1, JointFMSettings(), (1, 5, 21), 3
    )
    (call,) = fake_jointfm.forecast_calls
    last = len(call["history"]) - 1
    assert call["query_times"] == [last + 1, last + 5, last + 21]
    log_returns = call["samples"] - call["history"].iloc[-1].to_numpy()
    expected = log_returns if return_type == "LOG" else np.expm1(log_returns)
    assert samples.shape == (25, 3, len(TICKERS))
    np.testing.assert_allclose(samples, expected)


def test_scenarios_never_imply_non_positive_prices(returns_dict, fake_jointfm):
    """A strongly negative forecast log return still maps to a price above 0."""
    fake_jointfm.sample_scale = 5.0
    samples = _sample(returns_dict, return_type="LINEAR")
    assert samples.min() < -0.9
    assert np.all(1.0 + samples > 0.0)


def test_client_reads_settings_environment(returns_dict, fake_jointfm):
    settings = JointFMSettings(dotenv_path=None, config_path="c.yaml")
    generate_samples_jointfm(25, returns_dict["returns"], "LOG", 1, settings)
    assert fake_jointfm.from_env_kwargs == {
        "dotenv_path": None,
        "config_path": "c.yaml",
    }


@pytest.mark.parametrize("return_type", ["ABSOLUTE", "PNL"])
def test_return_type_without_price_level_is_refused(
    returns_dict, fake_jointfm, return_type
):
    with pytest.raises(ValueError, match="return_type"):
        _sample(returns_dict, return_type=return_type)
    assert fake_jointfm.from_env_kwargs is None


def test_overlapping_returns_are_refused(returns_dict, fake_jointfm):
    with pytest.raises(ValueError, match="freq=5"):
        _sample(returns_dict, freq=5)
    assert fake_jointfm.from_env_kwargs is None


@pytest.mark.parametrize("horizons", [(), (0, 1), (5, 5), (5, 1)])
def test_invalid_horizons_are_refused(returns_dict, fake_jointfm, horizons):
    """Empty, non-positive, or non-increasing horizons never reach the client."""
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_horizon_samples_jointfm(
            25, returns_dict["returns"], "LOG", 1, JointFMSettings(), horizons
        )
    assert fake_jointfm.from_env_kwargs is None


def test_horizon_beyond_n_output_is_refused(returns_dict, fake_jointfm):
    fake_jointfm.envelope.n_output = 20
    with pytest.raises(ValueError, match="n_output=20"):
        generate_horizon_samples_jointfm(
            25, returns_dict["returns"], "LOG", 1, JointFMSettings(), (1, 21)
        )
    assert fake_jointfm.forecast_calls == []


@pytest.mark.parametrize("min_series,max_series", [(1, 2), (4, 10)])
def test_universe_outside_capacity_is_refused(
    returns_dict, fake_jointfm, min_series, max_series
):
    fake_jointfm.envelope.min_series = min_series
    fake_jointfm.envelope.max_series = max_series
    with pytest.raises(ValueError, match="got 3 assets"):
        _sample(returns_dict)
    assert fake_jointfm.forecast_calls == []


def test_missing_capacity_envelope_is_refused(returns_dict, fake_jointfm):
    fake_jointfm.envelope = None
    with pytest.raises(ValueError, match="data_generation"):
        _sample(returns_dict)
    assert fake_jointfm.forecast_calls == []


def test_response_with_other_columns_is_refused(returns_dict, fake_jointfm):
    forecast = fake_jointfm.forecast

    def reversed_columns(history, **kwargs):
        response = forecast(history, **kwargs)
        response.requested_columns = tuple(reversed(response.requested_columns))
        return response

    fake_jointfm.forecast = reversed_columns
    with pytest.raises(ValueError, match="returned columns"):
        _sample(returns_dict)


def test_response_with_too_few_samples_is_refused(returns_dict, fake_jointfm):
    forecast = fake_jointfm.forecast

    def short(history, *, n_samples, **kwargs):
        return forecast(history, n_samples=n_samples - 1, **kwargs)

    fake_jointfm.forecast = short
    with pytest.raises(ValueError, match="samples of shape"):
        _sample(returns_dict)
