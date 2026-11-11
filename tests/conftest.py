# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures: price data, its returns, and a fake JointFM client."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from portfolio_optimization.settings import ReturnsComputeSettings
from portfolio_optimization.utils import calculate_returns

from portfolio_optimization_jointfm import sampler

TICKERS = ["AAPL", "GOOGL", "MSFT"]


class FakeJointFMClient:
    """Stand-in for ``jointfm_client.JointFMClient`` that records its calls.

    It mirrors the parts of the SDK the sampler reads: the health metadata's
    capacity envelope and a sample forecast whose ``samples`` axis order is
    ``(sample, horizon, column)`` and whose array is read-only. Draws scatter
    around each column's last history value, as future log prices do, one
    horizon per query time.
    """

    def __init__(self):
        self.envelope = SimpleNamespace(
            min_series=1, max_series=10, n_input=40, n_output=100
        )
        self.sample_scale = 1.0
        self.from_env_kwargs = None
        self.forecast_calls = []

    def health(self, *, cache=False):
        return SimpleNamespace(data_generation=self.envelope)

    def forecast(self, history, *, n_samples, requested_columns, **kwargs):
        rng = np.random.default_rng(kwargs["seed"])
        shape = (n_samples, len(kwargs["query_times"]), len(requested_columns))
        samples = history[requested_columns].iloc[-1].to_numpy() + (
            self.sample_scale * rng.normal(size=shape)
        )
        samples.flags.writeable = False
        self.forecast_calls.append(
            dict(
                history=history,
                n_samples=n_samples,
                requested_columns=requested_columns,
                samples=samples,
                **kwargs,
            )
        )
        return SimpleNamespace(
            samples=samples, requested_columns=tuple(requested_columns)
        )


@pytest.fixture()
def fake_jointfm(monkeypatch):
    """Route every ``JointFMClient.from_env`` of the sampler to one fake client."""
    client = FakeJointFMClient()

    def from_env(**kwargs):
        client.from_env_kwargs = kwargs
        return client

    monkeypatch.setattr(sampler, "JointFMClient", SimpleNamespace(from_env=from_env))
    return client


@pytest.fixture()
def price_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    prices = np.array([150.0, 100.0, 250.0]) + np.cumsum(
        rng.normal(0.0, 0.5, (60, 3)), axis=0
    )
    return pd.DataFrame(prices, index=dates, columns=TICKERS)


@pytest.fixture()
def returns_dict(price_data):
    return calculate_returns(
        price_data,
        regime_dict=None,
        returns_compute_settings=ReturnsComputeSettings(return_type="LOG", freq=1),
    )
