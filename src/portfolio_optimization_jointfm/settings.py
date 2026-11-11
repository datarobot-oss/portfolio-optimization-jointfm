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
    generator sends the look-back window's log-price path, rebuilt from its
    returns, forecasts the next log price, and converts its difference to the
    last log price into the configured return type; prices implied by the
    scenarios are therefore always positive. This needs one-period ('LOG' or
    'LINEAR', freq=1) returns. The path is sent whole when it fits the served
    model's history capacity and cut to its most recent points otherwise.

    Which JointFM service to call is not an option here: the
    ``jointfm-client`` SDK resolves it from its environment variables, read
    from the process environment, ``dotenv_path``, and ``config_path``.

    Unknown options are refused, so a misspelled option fails instead of
    leaving its default in place.

    Attributes
    ----------
    expected_return : str
        Source of the expected returns stored in ``CvarData.mean``:
        - 'scenario': mean of the JointFM scenarios, so expected return and
          CVaR come from the same forecast distribution
        - 'historical': mean of the look-back window's returns, the value
          the built-in generators use
        Default is 'scenario'.
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

    expected_return: Literal["scenario", "historical"] = Field(
        default="scenario", description="Source of CvarData.mean"
    )
    dotenv_path: Optional[str] = Field(
        default=".env", description="Client .env file; None skips it"
    )
    config_path: Optional[str] = Field(
        default="config.yaml", description="Client YAML configuration; None skips it"
    )
