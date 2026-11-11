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

"""The ``jointfm`` scenario generator of ``portfolio_optimization``.

The package's entry point in the ``portfolio_optimization.scenario_generators``
group names :func:`generate`, so ``ScenarioGenerationSettings(fit_type="jointfm")``
finds it once this package is installed.
"""

from portfolio_optimization.scenario_generators import Scenarios

from .sampler import generate_samples_and_mean_jointfm, generate_samples_jointfm
from .settings import JointFMSettings


def generate(returns_dict: dict, settings) -> Scenarios:
    """Next-period return scenarios for CVaR optimization from JointFM.

    Parameters
    ----------
    returns_dict : dict
        The look-back window as ``utils.calculate_returns`` builds it; reads
        ``returns`` (dates x assets), ``return_type``, and ``freq``.
    settings : ScenarioGenerationSettings
        Reads ``num_scen``, ``seed`` (forwarded to the service's sampler), and
        ``generator_options``, validated as :class:`JointFMSettings`.

    Returns
    -------
    Scenarios
        ``R`` of shape (num_scen, n_assets) in the column order of
        ``returns_dict["returns"]``; ``mean`` is the forecast's exact mean
        when ``expected_return='exact'``, the scenario mean when
        ``'scenario'``, and None (the window's mean) when ``'historical'``;
        ``p`` is None (equal weights).

    Raises
    ------
    pydantic.ValidationError
        If ``generator_options`` holds an unknown option or an invalid value.
    ValueError
        As :func:`generate_samples_and_mean_jointfm` for
        ``expected_return='exact'``, else as
        :func:`generate_horizon_samples_jointfm`.
    """
    options = JointFMSettings.model_validate(settings.generator_options)
    request = (
        settings.num_scen,
        returns_dict["returns"],
        returns_dict["return_type"],
        returns_dict["freq"],
        options,
    )
    if options.expected_return == "exact":
        R, mean = generate_samples_and_mean_jointfm(*request, seed=settings.seed)
        return Scenarios(R, mean=mean)
    R = generate_samples_jointfm(*request, seed=settings.seed)
    mean = R.mean(axis=0) if options.expected_return == "scenario" else None
    return Scenarios(R, mean=mean)


generate.display_name = "JointFM"
