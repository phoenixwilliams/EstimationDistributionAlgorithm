import GaussianModels
import DiscreteModels
import ProblemLibrary
import EstimatedDistributionAlgorithms
import numpy as np
import matplotlib.pyplot as plt


"""
Note that the implemented EDA algorithms are minimization, hence will minimize the given problem. Maximization problems
can be simply converted to minimization by multiplying the output with -1. Please see onemax() in ProblemLibrary 
for an example of this.
"""

def continuous_example():
    # Define the problem
    problem_dimension = 50
    problem_params = {
        "o": np.asarray([0.2] * problem_dimension)
    }
    lower_bounds = [-5] * problem_dimension
    upper_bounds = [5] *  problem_dimension
    problem = ProblemLibrary.Problem(ProblemLibrary.shifted_sphere, problem_params)
    probability_model = GaussianModels.MultivariateNormalModel()

    design = {
        "N": 200,
        "M": 100,
        "dimension": problem_dimension,
        "problem": problem,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "model": probability_model,
        "function_evaluations": 10000,
    }

    eda = EstimatedDistributionAlgorithms.ContinuousOptimizer(design)
    population, avg_fitness = eda.optimize(return_process=True)
    plt.plot(avg_fitness)
    plt.show()


def discrete_example():
    discrete_vals = [0, 1]
    model = DiscreteModels.IndependentDiscreteModel(discrete_vals)
    design = {
        "N": 200,
        "M": 100,
        "dimension": 100,
        "problem": ProblemLibrary.onemax,
        "discrete_values": discrete_vals,
        "function_evaluations": 10000,
        "model": model
    }

    umda = EstimatedDistributionAlgorithms.BinaryOptimizer(design)
    population, avg = umda.optimize(return_process=True)
    plt.plot(avg)
    plt.show()

if __name__ == "__main__":
    discrete_example()


