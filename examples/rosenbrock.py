import pathlib
import tempfile

from optuna_async_helper import (
    SearchSpace,
    SearchSpec,
    create_journal_storage,
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
        study = optimize(
            study_name="rosenbrock",
            storage=storage,
            objective_func=rosenbrock,
            search_space=search_space,
            n_trials=50,
            batch_size=8,
            z=z,
        )

        print(study.best_value)
        print(study.best_params)
