import pathlib
import tempfile

from optuna_async_helper import (
    SearchSpace,
    SearchSpec,
    create_journal_storage,
    create_study,
    optimize,
)


def rosenbrock(x: float, y: float, z: float) -> float:
    return (z - x) ** 2 + 100 * (y - x**2) ** 2


if __name__ == "__main__":
    search_space: SearchSpace = [
        SearchSpec(var_name="x", domain_type="float", low=-5, high=5),
        SearchSpec(var_name="y", domain_type="float", low=-5, high=5),
    ]
    z = -2.0

    with tempfile.TemporaryDirectory() as tempdir:
        storage_path = pathlib.Path(tempdir, "example.db")
        storage = create_journal_storage(str(storage_path))
        study = create_study(
            study_name="rosenbrock",
            storage=storage,
        )
        study = optimize(
            study,
            objective_func=rosenbrock,
            search_space=search_space,
            n_trials=100,
            batch_size=8,
            z=z,
        )

        for t in study.trials:
            print(t.params, t.value)

        print("Best result:")
        print(study.best_params, study.best_value)
