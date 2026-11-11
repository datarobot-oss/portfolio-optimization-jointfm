# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Joint return scenarios from a JointFM forecasting service."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from jointfm_client import JointFMClient

from .settings import JointFMSettings

JOINTFM_RETURN_TYPES = ("LOG", "LINEAR")

# The return type each forecast series turns into linearly, so that the
# forecast's exact mean is an exact expected return of that type.
SERIES_RETURN_TYPES = {
    "log_price": "LOG",
    "price": "LINEAR",
    "log_return": "LOG",
    "linear_return": "LINEAR",
}
LEVEL_SERIES = ("log_price", "price")
NEXT_PERIOD = (1,)


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
        The series JointFM forecasts and where the client reads its
        environment.
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
        num_scen,
        returns_data,
        return_type,
        freq,
        jointfm_settings,
        NEXT_PERIOD,
        seed=seed,
    )[:, 0, :]


def generate_samples_and_mean_jointfm(
    num_scen: int,
    returns_data: pd.DataFrame,
    return_type: str,
    freq: int,
    jointfm_settings: JointFMSettings,
    seed: int = None,
):
    """Sample next-period return scenarios and their exact expected value.

    The scenarios are those of ``generate_samples_jointfm``: the same request,
    so the same draws for the same seed. A second request on the same history
    asks the service for the mean of its forecast distribution of the next
    value (``return_mode='mean'``) and turns it into a return the way each
    draw is turned. That return is linear in the forecast only in the return
    type of the forecast series (``SERIES_RETURN_TYPES``), so only there is
    it the exact expected return; any other pairing is refused. A JointFM 1
    service computes the mean from the forecast distribution's parameters, so
    it carries none of the Monte Carlo noise of the scenarios' average; a
    service that estimates means by sampling returns that estimate instead.

    Parameters
    ----------
    num_scen, returns_data, return_type, freq, jointfm_settings, seed
        As ``generate_samples_jointfm``.

    Returns
    -------
    tuple of np.ndarray
        The scenarios, shape (num_scen, n_assets), and the expected returns,
        shape (n_assets,), both with columns in the order of ``returns_data``.

    Raises
    ------
    ValueError
        As ``generate_horizon_samples_jointfm``; if ``return_type`` is not the
        forecast series' return type; or if the mean response does not hold
        one value per asset.
    """
    forecast_series = jointfm_settings.forecast_series
    _check_request(return_type, freq, list(NEXT_PERIOD), forecast_series)
    series_return_type = SERIES_RETURN_TYPES[forecast_series]
    if return_type != series_return_type:
        matching_series = next(
            series
            for series in LEVEL_SERIES
            if SERIES_RETURN_TYPES[series] == return_type
        )
        raise ValueError(
            f"The exact expected return of forecast_series={forecast_series!r} "
            f"is a {series_return_type} return, and a {return_type} return is not "
            f"linear in it, so its mean is unknown. Use "
            f"forecast_series={matching_series!r}, whose returns are "
            f"{return_type} returns, or take the scenario mean "
            f"(expected_return='scenario')."
        )

    client, history = _open_request(
        returns_data, return_type, jointfm_settings, NEXT_PERIOD
    )
    samples = _sample_series_returns(
        client, history, num_scen, NEXT_PERIOD, seed, forecast_series
    )
    mean = _forecast(client, history, NEXT_PERIOD, return_mode="mean").mean
    _require_shape(mean, (len(NEXT_PERIOD), history.shape[1]), "mean")
    return samples[:, 0, :], _series_returns(mean, history, forecast_series)[0]


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

    The simulator rebuilds each asset's ``jointfm_settings.forecast_series``
    from its returns (``_series_path``), sends it as history, and forecasts
    its value ``h`` periods after the last one for every ``h`` in
    ``horizons``. Each draw becomes a return in the series' own return type
    (``_series_returns``): from the last level over the next ``h`` periods
    for a price series, the forecast return itself for a return series. That
    return is then converted into ``return_type``. A return series forecasts
    the return of one future period, not the return over several, so it
    takes the next period alone.

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
        Only the most recent ``n_input`` points of the rebuilt series are
        sent when it is longer than the history the served model accepts.
    return_type : str
        'LOG' or 'LINEAR', as in ReturnsComputeSettings.return_type.
    freq : int
        Return period in rows, as in ReturnsComputeSettings.freq; must be 1.
    jointfm_settings : JointFMSettings
        The series JointFM forecasts and where the client reads its
        environment, which names the JointFM service to call.
    horizons : sequence of int
        Periods after the last observation, positive and strictly increasing;
        only ``(1,)`` for a return series.
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
        ``freq`` is not 1, so no series rebuilt from log returns reproduces
        the returns; if ``horizons`` is empty, not positive, or not strictly
        increasing, or holds more than the next period for a return series;
        if the service reports no capacity envelope; if the asset count lies
        outside its ``[min_series, max_series]`` or a horizon beyond its
        ``n_output``; if the response does not hold ``num_scen`` draws of
        every asset at every horizon in order; or if a draw implies a
        non-positive price.
    """
    horizons = list(horizons)
    forecast_series = jointfm_settings.forecast_series
    _check_request(return_type, freq, horizons, forecast_series)
    client, history = _open_request(
        returns_data, return_type, jointfm_settings, horizons
    )
    returns = _sample_series_returns(
        client, history, num_scen, horizons, seed, forecast_series
    )
    if return_type == SERIES_RETURN_TYPES[forecast_series]:
        return returns
    if return_type == "LINEAR":
        return np.expm1(returns)
    return np.log1p(returns)


def _check_request(return_type, freq, horizons, forecast_series):
    """Refuse returns or horizons that the forecast series cannot represent.

    Runs before the client is opened, so a refused request sends nothing.

    Raises
    ------
    ValueError
        As the input checks listed in ``generate_horizon_samples_jointfm``.
    """
    if return_type not in JOINTFM_RETURN_TYPES:
        raise ValueError(
            f"fit_type='jointfm' needs return_type in {JOINTFM_RETURN_TYPES}: it "
            f"rebuilds the series it forecasts from log returns, and "
            f"{return_type!r} returns carry no price level to take a log of."
        )
    if freq != 1:
        raise ValueError(
            f"fit_type='jointfm' needs one-period returns (freq=1): overlapping "
            f"returns over freq={freq} rows are moving sums that no one-step "
            f"series reproduces."
        )
    if (
        not horizons
        or horizons[0] < 1
        or any(b <= a for a, b in zip(horizons, horizons[1:]))
    ):
        raise ValueError(
            f"JointFM horizons {horizons} must be positive and strictly increasing."
        )
    if forecast_series not in LEVEL_SERIES and horizons != list(NEXT_PERIOD):
        raise ValueError(
            f"forecast_series={forecast_series!r} forecasts the return of one "
            f"future period, not the return over several, so it takes horizon 1 "
            f"alone; got horizons {horizons}. Use a price series for longer "
            f"horizons."
        )


def _open_request(returns_data, return_type, jointfm_settings, horizons):
    """Open the client, check the service's capacity, and build the history.

    Returns
    -------
    tuple
        The ``JointFMClient`` and the history to send: the forecast series
        rebuilt from ``returns_data``, cut to the served model's ``n_input``
        most recent points, one column per asset named after its
        ``returns_data`` column. Its row position is its ordinal time.

    Raises
    ------
    ValueError
        If the service reports no capacity envelope, the asset count lies
        outside its ``[min_series, max_series]``, or the last horizon lies
        beyond its ``n_output``.
    """
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
    series = _series_path(log_returns.to_numpy(), jointfm_settings.forecast_series)
    history = pd.DataFrame(series[-envelope.n_input :], columns=asset_names)
    return client, history


def _series_path(log_returns, forecast_series):
    """Rebuild each asset's forecast series from its one-period log returns.

    A price series starts at the price before the first return, at log price
    0 (price 1). The served model reads each column relative to its last
    value and scaled by its spread, so the unknown real price level does not
    change the forecast. A return series has one row per return.

    Parameters
    ----------
    log_returns : np.ndarray
        One-period log returns, shape (n_returns, n_assets).
    forecast_series : str
        A value of ``JointFMSettings.forecast_series``.

    Returns
    -------
    np.ndarray
        Shape (n_returns + 1, n_assets) for a price series, (n_returns,
        n_assets) for a return series.
    """
    match forecast_series:
        case "log_price" | "price":
            log_prices = np.vstack(
                [np.zeros((1, log_returns.shape[1])), np.cumsum(log_returns, axis=0)]
            )
            return log_prices if forecast_series == "log_price" else np.exp(log_prices)
        case "log_return":
            return log_returns
        case "linear_return":
            return np.expm1(log_returns)


def _forecast(client, history, horizons, **mode):
    """Forecast every asset of ``history`` at each horizon in one request.

    ``mode`` holds ``return_mode`` and its controls. Every column is both a
    target and a requested column, and horizon ``h`` is the ordinal time
    ``h`` rows after the history's last.

    Raises
    ------
    ValueError
        If the response names other columns than ``history``, or in another
        order.
    """
    asset_names = list(history.columns)
    forecast = client.forecast(
        history,
        query_times=[len(history) - 1 + h for h in horizons],
        target_columns=asset_names,
        requested_columns=asset_names,
        **mode,
    )
    if tuple(forecast.requested_columns) != tuple(asset_names):
        raise ValueError(
            f"JointFM returned columns {list(forecast.requested_columns)}, "
            f"expected {asset_names}."
        )
    return forecast


def _require_shape(values, expected_shape, output):
    """Refuse a response array ``output`` whose shape is not ``expected_shape``."""
    if values.shape != expected_shape:
        raise ValueError(
            f"JointFM returned {output} of shape {values.shape}, "
            f"expected {expected_shape}."
        )


def _sample_series_returns(client, history, num_scen, horizons, seed, series):
    """Draw ``num_scen`` joint scenarios and turn them into returns of ``series``.

    Returns
    -------
    np.ndarray
        Shape (num_scen, len(horizons), n_assets), in the return type
        ``SERIES_RETURN_TYPES[series]``.
    """
    samples = _forecast(
        client,
        history,
        horizons,
        return_mode="samples",
        n_samples=num_scen,
        seed=seed,
    ).samples
    _require_shape(samples, (num_scen, len(horizons), history.shape[1]), "samples")
    return _series_returns(samples, history, series)


def _series_returns(values, history, forecast_series):
    """Turn forecast values of a series into returns of its own return type.

    A price series gives the return from the history's last value to the
    forecast value: the log-price difference for 'log_price', the price
    ratio minus 1 for 'price'. A return series' forecast already is that
    return. Each is linear in the forecast value, so the same map turns a
    forecast mean into the mean return.

    Returns
    -------
    np.ndarray
        A new writable array shaped like ``values``.

    Raises
    ------
    ValueError
        If a 'LINEAR' return lies at or below -1, which implies a
        non-positive price.
    """
    last = history.iloc[-1].to_numpy()
    match forecast_series:
        case "log_price":
            returns = values - last
        case "price":
            returns = values / last - 1.0
        case "log_return" | "linear_return":
            returns = np.array(values)
    if SERIES_RETURN_TYPES[forecast_series] == "LINEAR":
        non_positive = int(np.count_nonzero(returns <= -1.0))
        if non_positive:
            raise ValueError(
                f"{non_positive} of {returns.size} JointFM forecasts of "
                f"forecast_series={forecast_series!r} imply a non-positive price, "
                f"which no return scenario may hold. Use "
                f"forecast_series='log_price', whose forecasts always imply a "
                f"positive price."
            )
    return returns
