import numpy as np
class Problem:
    """
    During the optimization process a sample is evaluated by called a problem method e.g. problem(sample),
    if a problem relies on other parameters, these can be encoded in a dictionary which is passed to the
    problem function using pointers (**param_dictionary). This class carries this out, and can still
    called directly on a sample. Each sample is a numpy array.

    Example:
        x = np.random.uniform(-5, 5, (10,))
        problem_params = {
            "o": np.asarray([[0.2]] * 10)
        }
        problem = Problem(shifted_sphere, o)
        result = problem(x)
    """
    def __init__(self, problem, problem_params):
        self.problem = problem
        self.problem_params = problem_params

    def __call__(self, x):
        return self.problem(x, **self.problem_params)

def shifted_sphere(x, o):
    x = x - o
    return sum([xi**2 for xi in x])

def sphere(x):
    return sum(xi**2 for xi in x)

def onemax(genotype):
    """
    EDA algorithm implemented is a minimization algorithm. Maximization problems such as onemax can be
    converted to a minimization problem by multiplying the result with -1.
    """
    return -sum(genotype)
