import importlib.metadata
import logging
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Generic, Literal, ParamSpec, TypeAlias, TypeVar

import joblib
from optuna import Study, Trial
from optuna import create_study as _create_study
from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.storages import BaseStorage, JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
from optuna.trial import TrialState
from pydantic import BaseModel, Field

__version__ = importlib.metadata.version("optuna-async-helper")

DomainType: TypeAlias = Literal["int", "float", "categorical", "logint", "logfloat"]
Numeric: TypeAlias = int | float
Scalar: TypeAlias = int | float | str | bool

logger = logging.getLogger("optuna-async-helper")


def create_study(
    study_name: str,
    storage: str | Path | BaseStorage,
    sampler: BaseSampler | None = None,
    pruner: BasePruner | None = None,
    direction: Literal["minimize", "maximize"] = "minimize",
    load_if_exists: bool = True,
):
    if sampler is None:
        sampler = TPESampler(constant_liar=True)

    if isinstance(storage, Path):
        storage = create_journal_storage(storage)

    return _create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=load_if_exists,
    )


def create_journal_storage(file_path: str | Path) -> JournalStorage:
    file_path_ = str(file_path)
    lock_obj = JournalFileOpenLock(file_path_) if platform.system() == "win32" else None
    storage = JournalFileBackend(file_path_, lock_obj=lock_obj)
    return JournalStorage(storage)


T = TypeVar("T", int, float, str, bool)


class SearchSpec(BaseModel, Generic[T]):
    var_name: str
    domain_type: DomainType
    low: Numeric = 0
    high: Numeric = 0
    choices: Sequence[T] = Field(default_factory=list)

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


def _get_trial(study: Study, max_trials: int) -> Trial | None:
    # count running and completed trials
    it = filter(
        lambda t: t.state in [TrialState.RUNNING, TrialState.COMPLETE], study.trials
    )
    if len(list(it)) >= max_trials:
        return None
    return study.ask()


def _worker_func(
    study: Study,
    objective_func: Callable[P, float],
    search_space: SearchSpace,
    max_trials: int,
    max_retry: int = 3,
    retry_interval: float = 1.0,
    **fn_kwargs,
) -> None:
    last_exception = None
    while True:
        for _ in range(max_retry):
            try:
                trial = _get_trial(study, max_trials)
                break
            except Exception as e:
                last_exception = e
                time.sleep(retry_interval)
                continue
        else:
            if last_exception is not None:
                raise last_exception

        # max_trials reached
        if trial is None:
            break

        params = {spec.var_name: spec.suggest(trial) for spec in search_space}
        value = objective_func(**params, **fn_kwargs)  # type: ignore
        try:
            study.tell(trial, value)
        except Exception as e:
            logger.warning(f"Failed to tell the trial: {e}")


def optimize(
    study: Study,
    objective_func: Callable[P, float],
    search_space: SearchSpace,
    n_trials: int = 20,
    batch_size: int = 8,
    n_jobs: int = -1,
    initial_params: Sequence[Mapping[str, Scalar]] = [],
    max_retry: int = 3,
    retry_interval: float = 1.0,
    **fn_kwargs,
):
    for p in initial_params:
        study.enqueue_trial(params=dict(p), skip_if_exists=True)

    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_worker_func)(
            study,
            objective_func,
            search_space,
            n_trials,
            max_retry,
            retry_interval,
            **fn_kwargs,
        )
        for _ in range(batch_size)
    )

    return study
