import numpy as np

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6


def one_d_problem(x):
    x = x.get(f"x")
    return np.square(x) - 5.*x + 9.


if __name__=="__main__":
    print("Let's begin...")
    best_parameters, values, experiment, model = optimize(
        parameters=[
            {
                "name": "x",
                "type": "range",
                "bounds": [0.0, 10.0],
                "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                "log_scale": False,  # Optional, defaults to False.
            }
        ],
        experiment_name="test",
        objective_name="one_d_problem",
        evaluation_function=one_d_problem,
        minimize=True,  # Optional, defaults to False.
        #parameter_constraints=["x <= 20"],  # Optional.
        total_trials=30,  # Optional.
    )
    print(best_parameters)
