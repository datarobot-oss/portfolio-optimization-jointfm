# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""fit_type='jointfm' through NVIDIA's package: discovery, CvarData, solves."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
from portfolio_optimization import cvar_optimizer, rebalance, scenario_generators
from portfolio_optimization.cvar_parameters import CvarParameters
from portfolio_optimization.cvar_utils import generate_cvar_data
from portfolio_optimization.settings import (
    ReturnsComputeSettings,
    ScenarioGenerationSettings,
)
from pydantic import ValidationError

import portfolio_optimization_jointfm

from .conftest import TICKERS

CPU_SOLVER = {"solver": cp.CLARABEL, "verbose": False}


def _settings(num_scen=25, **options):
    return ScenarioGenerationSettings(
        num_scen=num_scen, fit_type="jointfm", seed=11, generator_options=options
    )


@pytest.fixture()
def cvar_params():
    return CvarParameters(
        w_min=0.0, w_max=1.0, c_min=0.0, c_max=0.0, risk_aversion=1.0, confidence=0.95
    )


def test_installed_entry_point_is_found():
    """The package's own metadata registers the generator; nothing is patched."""
    assert "jointfm" in scenario_generators.available()
    assert scenario_generators.get("jointfm") is portfolio_optimization_jointfm.generate
    assert scenario_generators.display_name("jointfm") == "JointFM"


def test_scenarios_feed_cvar_data(returns_dict, fake_jointfm):
    cd = generate_cvar_data(returns_dict, _settings())["cvar_data"]
    assert cd.R.shape == (len(TICKERS), 25)
    np.testing.assert_allclose(cd.p, np.full(25, 1 / 25))
    # The default takes expected returns from the forecast's exact mean, which
    # the fake puts mean_step * (1, 2, 3) above the last log price.
    np.testing.assert_allclose(
        cd.mean, fake_jointfm.mean_step * np.arange(1, len(TICKERS) + 1), rtol=1e-9
    )
    sample_call, mean_call = fake_jointfm.forecast_calls
    assert (sample_call["return_mode"], mean_call["return_mode"]) == ("samples", "mean")
    assert sample_call["n_samples"] == 25
    assert sample_call["seed"] == 11


def test_scenario_expected_return_is_scenario_mean(returns_dict, fake_jointfm):
    cd = generate_cvar_data(returns_dict, _settings(expected_return="scenario"))[
        "cvar_data"
    ]
    np.testing.assert_allclose(cd.mean, cd.R.mean(axis=1))
    (call,) = fake_jointfm.forecast_calls
    assert call["return_mode"] == "samples"


def test_historical_expected_return_is_window_mean(returns_dict, fake_jointfm):
    historical_mean = returns_dict["mean"].copy()
    cd = generate_cvar_data(returns_dict, _settings(expected_return="historical"))[
        "cvar_data"
    ]
    np.testing.assert_array_equal(cd.mean, historical_mean)


@pytest.mark.parametrize("expected_return", ["exact", "scenario"])
def test_model_expected_return_leaves_returns_dict_mean(
    returns_dict, fake_jointfm, expected_return
):
    historical_mean = returns_dict["mean"].copy()
    cd = generate_cvar_data(returns_dict, _settings(expected_return=expected_return))[
        "cvar_data"
    ]
    assert not np.allclose(cd.mean, historical_mean)
    np.testing.assert_array_equal(returns_dict["mean"], historical_mean)


def test_default_exact_mean_of_linear_returns_is_refused(
    linear_returns_dict, fake_jointfm
):
    """LINEAR returns take the exact mean only from the price series."""
    with pytest.raises(ValueError, match="forecast_series='price'"):
        generate_cvar_data(linear_returns_dict, _settings())
    assert fake_jointfm.from_env_kwargs is None
    fake_jointfm.sample_scale = 0.01
    cd = generate_cvar_data(linear_returns_dict, _settings(forecast_series="price"))[
        "cvar_data"
    ]
    assert cd.mean.shape == (len(TICKERS),)


@pytest.mark.parametrize(
    "options",
    [
        {"expected_returns": "historical"},
        {"expected_return": "forecast"},
        {"forecast_series": "returns"},
    ],
)
def test_invalid_options_are_refused_before_any_request(
    returns_dict, fake_jointfm, options
):
    """A misspelled option or an unknown value fails instead of being ignored."""
    with pytest.raises(ValidationError):
        generate_cvar_data(returns_dict, _settings(**options))
    assert fake_jointfm.from_env_kwargs is None


def test_options_reach_the_client(returns_dict, fake_jointfm):
    generate_cvar_data(returns_dict, _settings(dotenv_path="x.env", config_path=None))
    assert fake_jointfm.from_env_kwargs == {"dotenv_path": "x.env", "config_path": None}


def test_cvar_problem_solves_on_jointfm_scenarios(
    returns_dict, fake_jointfm, cvar_params
):
    fake_jointfm.sample_scale = 0.01
    returns_dict = generate_cvar_data(returns_dict, _settings(num_scen=500))
    _, portfolio = cvar_optimizer.CVaR(
        returns_dict, cvar_params
    ).solve_optimization_problem(solver_settings=CPU_SOLVER, print_results=False)
    assert portfolio.weights.shape == (len(TICKERS),)
    assert np.isclose(portfolio.weights.sum() + portfolio.cash, 1.0, atol=1e-6)


def test_rebalancing_requests_jointfm_at_every_decision(
    tmp_path, fake_jointfm, cvar_params
):
    """The rebalancer reaches the generator through fit_type alone."""
    fake_jointfm.sample_scale = 0.01
    dates = pd.date_range("2023-01-02", periods=200, freq="B")
    rng = np.random.default_rng(5)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.003, 0.01, (200, 3)), axis=0))
    price_path = tmp_path / "prices.csv"
    pd.DataFrame(prices, index=dates, columns=TICKERS).to_csv(price_path)

    rebalancer = rebalance.rebalance_portfolio(
        dataset_directory=str(price_path),
        returns_compute_settings=ReturnsComputeSettings(return_type="LOG", freq=1),
        scenario_generation_settings=_settings(num_scen=200),
        trading_start="2023-05-01",
        trading_end="2023-08-31",
        look_forward_window=10,
        look_back_window=60,
        cvar_params=cvar_params,
        solver_settings=CPU_SOLVER,
        re_optimize_criteria={"type": "every_step", "threshold": None},
    )
    _, re_optimize_dates, _ = rebalancer.re_optimize()
    # The initial portfolio and every re-optimization each send a sample
    # request and an exact-mean request.
    assert len(re_optimize_dates) > 0
    assert [call["return_mode"] for call in fake_jointfm.forecast_calls] == [
        "samples",
        "mean",
    ] * (len(re_optimize_dates) + 1)
