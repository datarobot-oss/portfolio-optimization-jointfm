# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The JointFM sampler: request contents, return conversion, and refusals."""

import numpy as np
import pytest
from portfolio_optimization.settings import ReturnsComputeSettings
from portfolio_optimization.utils import calculate_returns

from portfolio_optimization_jointfm import (
    JointFMSettings,
    generate_horizon_samples_jointfm,
    generate_samples_and_mean_jointfm,
    generate_samples_jointfm,
)

from .conftest import TICKERS

FORECAST_SERIES = ["log_price", "price", "log_return", "linear_return"]
SERIES_WITH_LINEAR_MEAN = [
    ("log_price", "LOG"),
    ("price", "LINEAR"),
    ("log_return", "LOG"),
    ("linear_return", "LINEAR"),
]
SERIES_WITHOUT_LINEAR_MEAN = [
    ("log_price", "LINEAR"),
    ("price", "LOG"),
    ("log_return", "LINEAR"),
    ("linear_return", "LOG"),
]


def _sample(
    returns_dict,
    return_type="LOG",
    freq=1,
    num_scen=25,
    seed=3,
    forecast_series="log_price",
):
    return generate_samples_jointfm(
        num_scen,
        returns_dict["returns"],
        return_type,
        freq,
        JointFMSettings(forecast_series=forecast_series),
        seed,
    )


def _expected_returns(values, last, forecast_series, return_type):
    """Reference conversion through each forecast value's next log return."""
    match forecast_series:
        case "log_price":
            log_returns = values - last
        case "price":
            log_returns = np.log(values) - np.log(last)
        case "log_return":
            log_returns = values
        case "linear_return":
            log_returns = np.log1p(values)
    return log_returns if return_type == "LOG" else np.expm1(log_returns)


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


def test_price_history_is_price_path_from_one(price_data, returns_dict, fake_jointfm):
    """The price series is the price path scaled to start at 1."""
    fake_jointfm.envelope.n_input = 1000
    fake_jointfm.sample_scale = 0.01
    _sample(returns_dict, forecast_series="price")
    history = fake_jointfm.forecast_calls[0]["history"]
    assert list(history.columns) == TICKERS
    np.testing.assert_allclose(
        history.to_numpy(), (price_data / price_data.iloc[0]).to_numpy(), rtol=1e-12
    )


@pytest.mark.parametrize("forecast_series", ["log_return", "linear_return"])
def test_return_history_is_the_returns(
    price_data, returns_dict, fake_jointfm, forecast_series
):
    """A return series is sent as the window's returns, one row per return."""
    fake_jointfm.envelope.n_input = 1000
    fake_jointfm.sample_scale = 0.01
    _sample(returns_dict, forecast_series=forecast_series)
    history = fake_jointfm.forecast_calls[0]["history"]
    log_returns = np.diff(np.log(price_data.to_numpy()), axis=0)
    expected = log_returns if forecast_series == "log_return" else np.expm1(log_returns)
    np.testing.assert_allclose(history.to_numpy(), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("forecast_series", FORECAST_SERIES)
@pytest.mark.parametrize("return_type", ["LOG", "LINEAR"])
def test_forecast_becomes_next_return(
    returns_dict, fake_jointfm, forecast_series, return_type
):
    """Every series' draw becomes the next period's return of return_type."""
    fake_jointfm.sample_scale = 0.01
    samples = _sample(
        returns_dict, return_type=return_type, forecast_series=forecast_series
    )
    (call,) = fake_jointfm.forecast_calls
    last = call["history"].iloc[-1].to_numpy()
    expected = _expected_returns(
        call["samples"][:, 0, :], last, forecast_series, return_type
    )
    np.testing.assert_allclose(samples, expected, rtol=1e-10, atol=1e-14)
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


@pytest.mark.parametrize("forecast_series", ["price", "linear_return"])
@pytest.mark.parametrize("return_type", ["LOG", "LINEAR"])
def test_forecast_implying_non_positive_price_is_refused(
    returns_dict, fake_jointfm, forecast_series, return_type
):
    """A draw at or below price 0 never becomes a scenario."""
    fake_jointfm.sample_scale = 5.0
    with pytest.raises(ValueError, match="non-positive price"):
        _sample(returns_dict, return_type=return_type, forecast_series=forecast_series)


@pytest.mark.parametrize("forecast_series", ["log_return", "linear_return"])
def test_return_series_takes_next_period_alone(
    returns_dict, fake_jointfm, forecast_series
):
    """A return series forecasts one future period's return, not a sum."""
    with pytest.raises(ValueError, match="horizon 1 alone"):
        generate_horizon_samples_jointfm(
            25,
            returns_dict["returns"],
            "LOG",
            1,
            JointFMSettings(forecast_series=forecast_series),
            (1, 5),
        )
    assert fake_jointfm.from_env_kwargs is None


@pytest.mark.parametrize("forecast_series,return_type", SERIES_WITH_LINEAR_MEAN)
def test_exact_mean_is_the_forecast_mean_as_a_return(
    returns_dict, linear_returns_dict, fake_jointfm, forecast_series, return_type
):
    """A mean request on the sample request's history gives the mean return."""
    returns = (returns_dict if return_type == "LOG" else linear_returns_dict)["returns"]
    settings = JointFMSettings(forecast_series=forecast_series)
    fake_jointfm.sample_scale = 0.01
    samples, mean = generate_samples_and_mean_jointfm(
        25, returns, return_type, 1, settings, 11
    )
    sample_call, mean_call = fake_jointfm.forecast_calls
    assert (sample_call["return_mode"], mean_call["return_mode"]) == ("samples", "mean")
    assert mean_call["history"] is sample_call["history"]
    assert mean_call["query_times"] == sample_call["query_times"]
    assert "n_samples" not in mean_call
    assert "seed" not in mean_call
    last = mean_call["history"].iloc[-1].to_numpy()
    expected = _expected_returns(
        mean_call["mean"][0], last, forecast_series, return_type
    )
    np.testing.assert_allclose(mean, expected, rtol=1e-10, atol=1e-14)
    assert mean.shape == (len(TICKERS),)
    np.testing.assert_array_equal(
        samples, generate_samples_jointfm(25, returns, return_type, 1, settings, 11)
    )


@pytest.mark.parametrize("forecast_series,return_type", SERIES_WITHOUT_LINEAR_MEAN)
def test_exact_mean_of_a_non_linear_return_is_refused(
    returns_dict, fake_jointfm, forecast_series, return_type
):
    """The mean of a return non-linear in the forecast is never approximated."""
    with pytest.raises(ValueError, match="not linear"):
        generate_samples_and_mean_jointfm(
            25,
            returns_dict["returns"],
            return_type,
            1,
            JointFMSettings(forecast_series=forecast_series),
        )
    assert fake_jointfm.from_env_kwargs is None


def test_mean_response_of_other_shape_is_refused(returns_dict, fake_jointfm):
    forecast = fake_jointfm.forecast

    def two_horizon_mean(history, **kwargs):
        if kwargs["return_mode"] == "mean":
            kwargs["query_times"] = kwargs["query_times"] * 2
        return forecast(history, **kwargs)

    fake_jointfm.forecast = two_horizon_mean
    with pytest.raises(ValueError, match="mean of shape"):
        generate_samples_and_mean_jointfm(
            25, returns_dict["returns"], "LOG", 1, JointFMSettings()
        )


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
