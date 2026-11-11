# SPDX-FileCopyrightText: Copyright (c) 2026 DataRobot, Inc. and its affiliates. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Options of the JointFM scenario generator."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class JointFMSettings(BaseModel):
    """Options of ``fit_type='jointfm'``, given as ``generator_options``.

    ``ScenarioGenerationSettings(fit_type="jointfm", generator_options={...})``
    carries these options as a plain dict, and the generator validates them on
    each call. Each scenario is one joint draw of the next period's asset
    returns, so it has the same meaning as a 'gaussian' or 'kde' scenario. The
    generator rebuilds each asset's ``forecast_series`` from the look-back
    window's returns, forecasts its next value, and converts that value into
    a return of the configured return type. This needs one-period ('LOG' or
    'LINEAR', freq=1) returns. The series is sent whole when it fits the
    served model's history capacity and cut to its most recent points
    otherwise.

    Which JointFM service to call is not an option here: the
    ``jointfm-client`` SDK resolves it from its environment variables, read
    from the process environment, ``dotenv_path``, and ``config_path``.

    Unknown options are refused, so a misspelled option fails instead of
    leaving its default in place.

    Attributes
    ----------
    forecast_series : str
        Series of each asset that JointFM reads as history and forecasts,
        rebuilt from the look-back window's returns:
        - 'log_price': the log-price path; every forecast implies a positive
          price
        - 'price': the price path, starting at 1 because JointFM standardizes
          each series by its last value and spread, so the price level does
          not change the forecast; a forecast at or below 0 is refused
        - 'log_return': the log returns themselves
        - 'linear_return': the linear returns themselves; a forecast at or
          below -1, which implies a non-positive price, is refused
        JointFM centers every series on its last history value, which suits
        price series; the return series are there for comparison. Default is
        'log_price'.
    expected_return : str
        Source of the expected returns stored in ``CvarData.mean``:
        - 'exact': the exact mean of JointFM's forecast distribution, from
          one more request beside the scenarios, so it carries no Monte Carlo
          noise. The return must be linear in the forecast, so this needs
          return_type 'LOG' for the 'log_price' and 'log_return' series and
          'LINEAR' for 'price' and 'linear_return'. Exact only on a service
          that answers means exactly, such as the JointFM inference runner
          from release 0.5.0.
        - 'scenario': mean of the JointFM scenarios, which adds their Monte
          Carlo noise
        - 'historical': mean of the look-back window's returns, the value
          the built-in generators use
        Default is 'exact'.
    dotenv_path : str, optional
        ``.env`` file the client reads its environment variables from; a
        missing file is skipped. None reads only the process environment.
        Default is '.env'.
    config_path : str, optional
        Client YAML configuration file; a missing file is skipped. None uses
        the client's built-in defaults. Default is 'config.yaml'.

    Examples
    --------
    >>> JointFMSettings.model_validate({"expected_return": "historical"}).dotenv_path
    '.env'
    """

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    forecast_series: Literal["log_price", "price", "log_return", "linear_return"] = (
        Field(default="log_price", description="Series JointFM reads and forecasts")
    )
    expected_return: Literal["exact", "scenario", "historical"] = Field(
        default="exact", description="Source of CvarData.mean"
    )
    dotenv_path: Optional[str] = Field(
        default=".env", description="Client .env file; None skips it"
    )
    config_path: Optional[str] = Field(
        default="config.yaml", description="Client YAML configuration; None skips it"
    )
