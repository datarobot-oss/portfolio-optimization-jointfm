# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint return scenarios from a JointFM forecasting service."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from jointfm_client import JointFMClient

from .settings import JointFMSettings

JOINTFM_RETURN_TYPES = ("LOG", "LINEAR")


def generate_samples_jointfm(
    num_scen: int,
    returns_data: pd.DataFrame,
    return_type: str,
    freq: int,
    jointfm_settings: JointFMSettings,
    seed: int = None,
):
    """Sample joint next-period return scenarios from a JointFM model.

    Each scenario is one joint draw of the row that follows ``returns_data``,
    so it stands for the same quantity as a 'gaussian' or 'kde' scenario. It
    is the one-period case of ``generate_horizon_samples_jointfm``, which
    describes the request.

    Parameters
    ----------
    num_scen : int
        Number of scenarios to generate.
    returns_data : pd.DataFrame
        Historical one-period returns (dates x assets), oldest row first.
    return_type : str
        'LOG' or 'LINEAR', as in ReturnsComputeSettings.return_type.
    freq : int
        Return period in rows, as in ReturnsComputeSettings.freq; must be 1.
    jointfm_settings : JointFMSettings
        Where the client reads its environment.
    seed : int, optional
        Seed forwarded to the service's sampler. None (default) draws fresh
        entropy, so repeated calls return different samples.

    Returns
    -------
    np.ndarray
        Generated samples with shape (num_scen, n_assets), columns in the
        order of ``returns_data``.

    Raises
    ------
    ValueError
        As ``generate_horizon_samples_jointfm``.
    """
    return generate_horizon_samples_jointfm(
        num_scen, returns_data, return_type, freq, jointfm_settings, (1,), seed=seed
    )[:, 0, :]


def generate_horizon_samples_jointfm(
    num_scen: int,
    returns_data: pd.DataFrame,
    return_type: str,
    freq: int,
    jointfm_settings: JointFMSettings,
    horizons: Sequence[int],
    seed: int = None,
):
    """Sample joint return scenarios over several horizons from one JointFM request.

    The simulator sends each asset's log-price path, the cumulative sum of
    its log returns ('LINEAR' returns through ``log1p``) starting from 0 at
    the price before ``returns_data``, and forecasts the log price ``h``
    periods after the last one for every ``h`` in ``horizons``. The path's
    starting constant is arbitrary because the served model reads each column
    relative to its last value. Each draw minus the path's last log price is
    the log return over the next ``h`` periods, which is converted into
    ``return_type``. Any finite log return maps to a positive price, so no
    scenario implies a negative price.

    All assets go into one request because a scenario is a draw from their
    joint distribution; the ``jointfm-client`` SDK splits a large ``num_scen``
    across requests along the sample axis, never across assets. The service
    draws each horizon independently, so scenario ``s``
    at one horizon and scenario ``s`` at another are not one path: each
    horizon's scenarios describe the joint distribution of the assets at that
    horizon alone.

    Parameters
    ----------
    num_scen : int
        Number of scenarios per horizon.
    returns_data : pd.DataFrame
        Historical one-period returns (dates x assets), oldest row first.
        Only the most recent ``n_input`` points of the log-price path are
        sent when it is longer than the history the served model accepts.
    return_type : str
        'LOG' or 'LINEAR', as in ReturnsComputeSettings.return_type.
    freq : int
        Return period in rows, as in ReturnsComputeSettings.freq; must be 1.
    jointfm_settings : JointFMSettings
        Where the client reads its environment, which names the JointFM
        service to call.
    horizons : sequence of int
        Periods after the last observation, positive and strictly increasing.
    seed : int, optional
        Seed forwarded to the service's sampler. None (default) draws fresh
        entropy, so repeated calls return different samples.

    Returns
    -------
    np.ndarray
        Returns over each horizon, shape (num_scen, len(horizons), n_assets),
        columns in the order of ``returns_data``.

    Raises
    ------
    ValueError
        If ``return_type`` carries no price level ('ABSOLUTE', 'PNL') or
        ``freq`` is not 1, so no log-price path reproduces the returns; if
        ``horizons`` is empty, not positive, or not strictly increasing; if
        the service reports no capacity envelope; if the asset count lies
        outside its ``[min_series, max_series]`` or a horizon beyond its
        ``n_output``; or if the response does not hold ``num_scen`` draws of
        every asset at every horizon in order.
    """
    if return_type not in JOINTFM_RETURN_TYPES:
        raise ValueError(
            f"fit_type='jointfm' needs return_type in {JOINTFM_RETURN_TYPES}: it "
            f"forecasts log prices, and {return_type!r} returns carry no price "
            f"level to take a log of."
        )
    if freq != 1:
        raise ValueError(
            f"fit_type='jointfm' needs one-period returns (freq=1): overlapping "
            f"returns over freq={freq} rows are moving sums that no one-step "
            f"log-price path reproduces."
        )
    horizons = list(horizons)
    if (
        not horizons
        or horizons[0] < 1
        or any(b <= a for a, b in zip(horizons, horizons[1:]))
    ):
        raise ValueError(
            f"JointFM horizons {horizons} must be positive and strictly increasing."
        )

    client = JointFMClient.from_env(
        dotenv_path=jointfm_settings.dotenv_path,
        config_path=jointfm_settings.config_path,
    )
    envelope = client.health(cache=True).data_generation
    if envelope is None:
        raise ValueError(
            "The JointFM service reports no 'data_generation' capacity envelope, "
            "so the asset count and history length cannot be checked."
        )

    n_assets = returns_data.shape[1]
    if not envelope.min_series <= n_assets <= envelope.max_series:
        raise ValueError(
            f"fit_type='jointfm' forecasts all assets jointly in one request, and "
            f"the served model accepts {envelope.min_series} to "
            f"{envelope.max_series} series; got {n_assets} assets. Choose a "
            f"universe within that range."
        )
    if horizons[-1] > envelope.n_output:
        raise ValueError(
            f"The served model forecasts at most n_output={envelope.n_output} "
            f"periods ahead; got horizon {horizons[-1]}."
        )

    asset_names = [str(column) for column in returns_data.columns]
    log_returns = returns_data if return_type == "LOG" else np.log1p(returns_data)
    log_prices = np.vstack(
        [np.zeros((1, n_assets)), np.cumsum(log_returns.to_numpy(), axis=0)]
    )
    # Ordinal time is the row position, so the history needs no date index.
    history = pd.DataFrame(log_prices[-envelope.n_input :], columns=asset_names)
    forecast = client.forecast(
        history,
        query_times=[len(history) - 1 + h for h in horizons],
        return_mode="samples",
        n_samples=num_scen,
        seed=seed,
        target_columns=asset_names,
        requested_columns=asset_names,
    )

    if tuple(forecast.requested_columns) != tuple(asset_names):
        raise ValueError(
            f"JointFM returned columns {list(forecast.requested_columns)}, "
            f"expected {asset_names}."
        )
    expected_shape = (num_scen, len(horizons), n_assets)
    if forecast.samples.shape != expected_shape:
        raise ValueError(
            f"JointFM returned samples of shape {forecast.samples.shape}, "
            f"expected {expected_shape}."
        )
    horizon_log_returns = forecast.samples - history.iloc[-1].to_numpy()
    if return_type == "LOG":
        return horizon_log_returns
    return np.expm1(horizon_log_returns)
