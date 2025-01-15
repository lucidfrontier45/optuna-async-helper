import tempfile

from optuna_async_helper import (
    SearchSpace,
    SearchSpec,
    optimize,
    create_journal_storage,
    create_study,
)


def rosenbrock(x: float, y: float, z: float) -> float:
    return (z - x) ** 2 + 100 * (y - x**2) ** 2


def test_optimizer():
    search_space: SearchSpace = [
        SearchSpec(var_name="x", domain_type="float", low=-5, high=5),
        SearchSpec(var_name="y", domain_type="float", low=-5, high=5),
    ]
    z = 0.5
    initial_params = [
        {"x": 0, "y": 0},
        {"x": 1.0, "y": 0},
        {"x": 0, "y": 1.0},
    ]

    with tempfile.TemporaryDirectory() as tempdir:
        storage = create_journal_storage(f"{tempdir}/example.db")
        study = create_study(
            study_name="rosenbrock",
            storage=storage,
        )
        study = optimize(
            study,
            objective_func=rosenbrock,
            search_space=search_space,
            initial_params=initial_params,
            n_trials=10,
            batch_size=32,
            z=z,
        )

        assert study.best_value < 1.0
        assert abs(study.best_params["x"] - z) < 1.0
        assert abs(study.best_params["y"] - z) < 1.0
