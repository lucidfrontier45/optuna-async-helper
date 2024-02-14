from collections.abc import Callable
from typing import Literal, ParamSpec, TypeAlias

from optuna import Trial, create_study
from optuna.samplers import BaseSampler, TPESampler
from pydantic import BaseModel, Field

__version__ = "0.1.0"

VariableType: TypeAlias = Literal["int", "float", "categorical", "logint", "logfloat"]


class SearchSpec(BaseModel):
    var_name: str
    var_type: VariableType
    low: int | float = 0
    high: int | float = 0
    choices: list[int | float | str | bool] = Field(default_factory=list)

    def suggest(self, trial: Trial):
        match self.var_type:
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


def optimize(
    study_name: str,
    storage: str,
    objective_func: Callable[P, float],
    search_space: SearchSpace,
    sampler: BaseSampler | None = None,
    direction: Literal["minimize", "maximize"] = "minimize",
    n_trials: int = 20,
):
    if sampler is None:
        sampler = TPESampler(constant_liar=True)

    study = create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction=direction,
    )

    for _ in range(n_trials):
        trial = study.ask()
        params = {spec.var_name: spec.suggest(trial) for spec in search_space}
        value = objective_func(**params)  # type: ignore
        study.tell(trial, value)

    return study
