# Optuna Async Helper
A Helper Library for Optuna Async Optimization

# Install

```bash
pip install optuna-async-helper
```

# Usage

```python
from optuna_async_helper import SearchSpace, SearchSpec, optimize


def rosenbrock(x: float, y: float) -> float:
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


search_space: SearchSpace = [
    SearchSpec(var_name="x", var_type="float", low=-5, high=5),
    SearchSpec(var_name="y", var_type="float", low=-5, high=5),
]

with tempfile.TemporaryDirectory() as tempdir:
    study = optimize(
        study_name="rosenbrock",
        storage=f"sqlite:///example.db",
        objective_func=rosenbrock,
        search_space=search_space,
        n_trials=50,
        batch_size=4,
    )

    assert study.best_value < 1.0
    assert abs(study.best_params["x"] - 1) < 1.0
    assert abs(study.best_params["y"] - 1) < 1.0
```

For more detail, please check `optimize` and `SearchSpec` definitions.