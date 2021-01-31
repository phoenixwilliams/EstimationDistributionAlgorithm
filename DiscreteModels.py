import numpy as np

"""
This script hold models for discrete values. 
"""

class IndependentDiscreteModel:
    """
    This is a simple probability model, that determines the probability of selecting each discrete value at each
     entry from the given dataset. Sampling this model is done by generating random samples with probabilities of each
     value being selected at each entry,
    """
    def __init__(self, discrete_values):
        self.probability_vector = None
        self.discrete_values = discrete_values

    def __call__(self, dataset):
        self.probability_vector: float = np.zeros(shape=(dataset.shape[1], len(self.discrete_values)))
        for val in range(len(self.discrete_values)):
            count = np.count_nonzero(dataset == self.discrete_values[val], axis=0)
            for c in range(len(count)):
                self.probability_vector[c][val] = count[c] / dataset.shape[0]

        sums = np.asarray([sum(c) for c in self.probability_vector])
        assert sums.all() == 1


    def sample(self):
        sample_vec = np.zeros(shape=(self.probability_vector.shape[0],))

        for s in range(self.probability_vector.shape[0]):
            sample_vec[s] = np.random.choice(self.discrete_values, p=self.probability_vector[s])

        return sample_vec