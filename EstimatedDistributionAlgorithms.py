import numpy as np
from operator import attrgetter


"""
This script contains the Population EDA algorithm. Once an initial population if size N has been generated, 
population EDA algorithm iterate over 3 steps:
    - Evaluate Population and Select M best solutions as the population
    - Generate a probabilistic model of the population 
    - Fill the remaining solution of the population (N - M) by sampling the model
    
Different population EDA algorithms can be fully described by the model used, and the values of M and N. The algorithm
in this script follow this, therefore already made probability model or a user generated model can be used within this
implementation by simply following some basic syntax. View the GaussianModels script to see what is required.
"""

class Solution:
    """
    Class to hold the solutions and fitness.
    """
    def __init__(self, genotype):
        self.genotype = genotype
        self.fitness = None

    def set_fitness(self, f):
        self.fitness = f

def initial_continuous_population(size, dimension):
    population:Solution = [None] * size
    for i in range(size):
        population[i] = Solution(np.random.uniform(0, 1, size=(dimension,)))

    return population

def initial_discrete_population(size, dimension, discrete_values):
    population: Solution = [None] * size
    for i in range(size):
        population[i] = Solution(np.random.choice(discrete_values,dimension))

    return population


class ContinuousOptimizer:
    def __init__(self, design):
        self.design = design

    def optimize(self, return_process):
        """
        Function returns genotypes of the final population
        :param return_process: If true, returns the average fitness of the population at each generation
        :return:
        """
        population = initial_continuous_population(self.design["N"], self.design["dimension"])
        c_it = self.design["N"]
        avg_fitness = []

        # Step 1. Evaluate all candidate solutions
        for p in population:
            genotype = np.asarray([xi*(ub-lb)+lb for xi,ub,lb in zip(p.genotype, self.design["upper_bounds"],self.design["lower_bounds"])])
            p.set_fitness(self.design["problem"](genotype))

        # Step 2. Select Promising Solutions


        population = sorted(population, key=attrgetter("fitness"))[:self.design["M"]]
        model = self.design["model"]

        while c_it < self.design["function_evaluations"]:
            avg_fitness.append(sum(p.fitness for p in population) / len(population))

            # Step 3. Build Probabilistic Model of promising solutions
            model(np.asarray([p.genotype for p in population]))

            # Step 4. Sample N-M solutions to form new population
            offsprings: Solution = [None] * (self.design["N"] - self.design["M"])
            for o in range(self.design["N"] - self.design["M"]):
                samplegeno = np.clip(model.sample(), 0, 1)
                sample = Solution(samplegeno)
                genotype = np.asarray([xi * (ub - lb) + lb for xi, ub, lb in zip(sample.genotype, self.design["upper_bounds"], self.design["lower_bounds"])])
                sample.set_fitness(self.design["problem"](genotype))
                offsprings[o] = sample
                c_it += 1

            population = population + offsprings
            population = sorted(population, key=attrgetter("fitness"))[:self.design["M"]]

        avg_fitness.append(sum(p.fitness for p in population) / len(population))
        if return_process:
            return [i.genotype for i in population], avg_fitness
        else:
            return [i.genotype for i in population]


class BinaryOptimizer:

    """
    Class for optimizing problem with discrete values. Solutions of these problems can only take discrete values.
    """
    def __init__(self, design):
        self.design = design


    def optimize(self, return_process):
        population = initial_discrete_population(self.design["N"],self.design["dimension"], self.design["discrete_values"])
        c_it = self.design["N"]
        avg_fitness = []

        # Step 1. Evaluate all candidate solutions
        for p in population:
            p.set_fitness(self.design["problem"](p.genotype))

        # Step 2. Select Promising Solutions
        population = sorted(population, key=attrgetter("fitness"))[:self.design["M"]]

        model = self.design["model"]

        while c_it < self.design["function_evaluations"]:
            avg_fitness.append(sum(p.fitness for p in population)/len(population))

            # Step 3. Build Probabilistic Model of promising solutions
            model(np.asarray([p.genotype for p in population]))

            # Step 4. Sample N-M solutions to form new population
            offsprings:Solution = [None] * (self.design["N"] - self.design["M"])
            for o in range(self.design["N"] - self.design["M"]):
                sample = Solution(model.sample())

                sample.set_fitness(self.design["problem"](sample.genotype))
                offsprings[o] = sample
                c_it += 1

            population = population + offsprings
            population = sorted(population, key=attrgetter("fitness"))[:self.design["M"]]

        avg_fitness.append(sum(p.fitness for p in population) / len(population))
        if return_process:
            return [i.genotype for i in population], avg_fitness
        else:
            return [i.genotype for i in population]