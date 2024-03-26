import tempfile

from optuna_async_helper import SearchSpace, SearchSpec, optimize


def rosenbrock(x: float, y: float, z: float) -> float:
    return (z - x) ** 2 + 100 * (y - x**2) ** 2


def test_optimizer():
    search_space: SearchSpace = [
        SearchSpec(var_name="x", domain_type="float", low=-5, high=5),
        SearchSpec(var_name="y", domain_type="float", low=-5, high=5),
    ]
    z = 0.5

    with tempfile.TemporaryDirectory() as tempdir:
        study = optimize(
            study_name="rosenbrock",
            storage=f"sqlite:///{tempdir}/example.db",
            objective_func=rosenbrock,
            search_space=search_space,
            n_trials=50,
            batch_size=8,
            z=z,
        )

        assert study.best_value < 1.0
        assert abs(study.best_params["x"] - z) < 1.0
        assert abs(study.best_params["y"] - z) < 1.0
