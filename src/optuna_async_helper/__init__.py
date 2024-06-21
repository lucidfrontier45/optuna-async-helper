import importlib.metadata
from collections.abc import Callable
from typing import Literal, ParamSpec, TypeAlias

import joblib
from optuna import Study, Trial, create_study
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler, TPESampler
from pydantic import BaseModel, Field

__version__ = importlib.metadata.version("optuna-async-helper")

DomainType: TypeAlias = Literal["int", "float", "categorical", "logint", "logfloat"]
Numeric: TypeAlias = int | float
Scalar: TypeAlias = int | float | str | bool


class SearchSpec(BaseModel):
    var_name: str
    domain_type: DomainType
    low: Numeric = 0
    high: Numeric = 0
    choices: list[int | float | str | bool] = Field(default_factory=list)

    def suggest(self, trial: Trial):
        match self.domain_type:
            case "int":
                return trial.suggest_int(self.var_name, int(self.low), int(self.high))
            case "logint":
                return trial.suggest_int(
                    self.var_name, int(self.low), int(self.high), log=True
                )
            case "float":
                return trial.suggest_float(self.var_name, self.low, self.high)
            case "logfloat":
                return trial.suggest_float(self.var_name, self.low, self.high, log=True)
            case "categorical":
                return trial.suggest_categorical(self.var_name, self.choices)


SearchSpace: TypeAlias = list[SearchSpec]


P = ParamSpec("P")


def _worker_func(
    study: Study,
    objective_func: Callable[P, float],
    search_space: SearchSpace,
    n_trials: int,
    **fn_kwargs,
) -> None:
    for _ in range(n_trials):
        trial = study.ask()
        params = {spec.var_name: spec.suggest(trial) for spec in search_space}
        value = objective_func(**params, **fn_kwargs)  # type: ignore
        study.tell(trial, value)


def optimize(
    study_name: str,
    storage: str,
    objective_func: Callable[P, float],
    search_space: SearchSpace,
    sampler: BaseSampler | None = None,
    direction: Literal["minimize", "maximize"] = "minimize",
    n_trials: int = 20,
    batch_size: int = 8,
    n_jobs: int = -1,
    load_if_exists: bool = True,
    pruner: BasePruner | None = None,
    initial_params: dict[str, Scalar] | None = None,
    **fn_kwargs,
):
    if sampler is None:
        sampler = TPESampler(constant_liar=True)

    study = create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction=direction,
        load_if_exists=load_if_exists,
        pruner=pruner,
    )

    if initial_params is not None:
        study.enqueue_trial(params=initial_params, skip_if_exists=True)

    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_worker_func)(
            study, objective_func, search_space, n_trials, **fn_kwargs
        )
        for _ in range(batch_size)
    )

    return study
