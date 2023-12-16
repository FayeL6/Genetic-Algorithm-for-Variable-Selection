import numpy as np
import random
import statsmodels.api as sm
from pandas import DataFrame
from scipy.stats import rankdata

def calculate_aic(obj_fun_input):
    """
    Calculate the Akaike Information Criterion (AIC).

    Parameters:
    obj_fun_input (list): A list containing [y, prediction, num_of_variables].

    Returns:
    float: The calculated AIC value.
    """
    y, prediction, num_of_variables = obj_fun_input
    n = len(y)
    rss = np.sum((y - prediction) ** 2)
    likelihood_fun = (rss / n) + np.finfo(float).eps  # Add epsilon to avoid divide by zero
    k = num_of_variables + 1  # Number of variables + intercept
    return 2 * k + n * np.log(likelihood_fun)

def calculate_bic(obj_fun_input):
    """
    Calculate the Bayesian Information Criterion (BIC).

    Parameters:
    obj_fun_input (list): A list containing [y, prediction, num_of_variables].

    Returns:
    float: The calculated BIC value.
    """
    y, prediction, num_of_variables = obj_fun_input
    n = len(y)
    rss = np.sum((y - prediction) ** 2)
    likelihood_fun = (rss + np.finfo(float).eps) / n
    return np.log(n) * num_of_variables + n * np.log(likelihood_fun)

def calculate_adjusted_r2(obj_fun_input):
    """
    Calculate the Adjusted R-squared value.

    Parameters:
    obj_fun_input (list): A list containing [y, prediction, num_of_variables, X].

    Returns:
    float: The calculated Adjusted R-squared value.
    """
    y, prediction, num_of_variables, X = obj_fun_input
    n = len(y)
    if n <= num_of_variables + 1:
        return float('inf') 
    rss = np.sum((y - prediction) ** 2)
    tss = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (rss / tss)
    return 1 - ((1 - r_squared) * (n - 1) / (n - num_of_variables - 1))

class GeneticAlgorithm:
    """
    Genetic Algorithm for variable selection in regression models.
    """
    def __init__(self, num_generations, population_size, selection_method='roulette_wheel', mutation_rate=0.01, tournament_size=3, obj_fun='aic', opr_fun=['crossover', 'mutate'], opr_para=[2,1]):
        assert isinstance(num_generations, int) and num_generations > 0, "num_generations must be a positive integer"
        assert isinstance(population_size, int) and population_size > 0, "population_size must be a positive integer"
        assert selection_method in ['roulette_wheel', 'tournament'], "Invalid selection_method"
        assert 0 < mutation_rate < 1, "mutation_rate must be between 0 and 1"
        assert isinstance(tournament_size, int) and tournament_size > 0, "tournament_size must be a positive integer"

        self.num_generations = num_generations
        self.population_size = population_size
        self.selection_method = selection_method
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.obj_fun = self.set_obj_fun(obj_fun)
        self.opr_fun = self.set_opr_fun(opr_fun, opr_para)

    def set_obj_fun(self, obj_fun):
        if isinstance(obj_fun, str):
            obj_fun_map = {'aic': calculate_aic, 'bic': calculate_bic, 'adjusted_r2': calculate_adjusted_r2}
            chosen_fun = obj_fun_map.get(obj_fun.lower())
            assert chosen_fun is not None, f"Invalid objective function: {obj_fun}"
            return chosen_fun
        elif callable(obj_fun):
            # If obj_fun is a function, use it directly
            return obj_fun
        else:
            raise ValueError("obj_fun must be either a string key or a callable function")
        
    def set_opr_fun(self, opr_fun, opr_para):
        operator = []
        opr_fun_map = {'crossover': self.crossover, 'mutate': self.mutate}
        for fun, para in zip(opr_fun, opr_para):
            if isinstance(fun, str):
                chosen_fun = opr_fun_map.get(fun.lower())
                assert chosen_fun is not None, f"Invalid genetic operator function: {fun}"
                operator.append((chosen_fun, para))
            elif callable(fun):
                # If opr_fun is a function, use it directly
                operator.append((fun, para))
            else:
                raise ValueError("opr_fun must be either a string key or a callable function")
        return operator

    def fit(self, data, target_variable, FUN, family=None):
        """
        Prepares the genetic algorithm with the given dataset and regression function.

        Parameters:
        data (DataFrame): The dataset to be used. Must be a pandas DataFrame.
        target_variable (str): The name of the target variable in the dataset.
        FUN (function): The regression function to be used. This can be a function like sm.OLS 
                        for ordinary least squares regression, or any function from statsmodels 
                        or scikit-learn that conforms to their API.
        family (Optional[statsmodels.genmod.families.family]): The family of distributions to be 
            used in the model. This parameter is necessary for Generalized Linear Models (GLMs). 
            Examples include statsmodels.genmod.families.Gaussian, 
            statsmodels.genmod.families.Binomial, etc. Default is None, which is suitable for OLS.
        """
        # Validate the input data and target_variable
        assert isinstance(data, DataFrame), "data must be a pandas DataFrame"
        assert target_variable in data.columns, "target_variable not found in data"

        self.data = data
        self.target_variable = target_variable
        self.FUN = FUN
        self.variable_list = [col for col in self.data.columns if col != self.target_variable]
        self.num_predictor_variables = len(data.columns) - 1 if target_variable in data.columns else len(data.columns)

        # Initialize population with the correct number of predictor variables
        self.population = self.initialize_population()
        self.family = family

    def initialize_population(self):
        """
        Initializes the population for the genetic algorithm.

        Returns:
        np.array: An array representing the initial population.
        """
        population = []
        for _ in range(self.population_size):
            individual = np.random.choice([0, 1], size=self.num_predictor_variables)
            population.append(individual)
        return np.array(population)

    def calculate_fitness(self, individual):
        y = self.data[self.target_variable]
        selected_variables = [self.variable_list[i] for i in range(self.num_predictor_variables) if individual[i]]

        if not selected_variables:
            return float('inf')

        X = self.data[selected_variables]
        if 'statsmodels' in str(self.FUN):  # Check if FUN is a statsmodels function
            X = sm.add_constant(X)
            if self.family:  # Use the family parameter for GLM
                model = self.FUN(y, X, family=self.family).fit()
            else:
                model = self.FUN(y, X).fit()
        else:
            if callable(self.FUN):  # Check if FUN is a callable
                model = self.FUN()  # Call FUN to get a new model instance
                model = model.fit(X, y)
            else:
                model = self.FUN.fit(X, y)  # Use the model directly if it's not callable

        # Calculate fitness using the objective function...
        prediction = model.predict(X)
        obj_fun_input = [y, prediction, len(selected_variables)]
        return self.obj_fun(obj_fun_input)
        
    def selection(self):
        """
        Selects parents for the next generation based on the selection method.

        Returns:
        list: A list of selected parents.
        """
        selected_parents = []
        fitness_scores = [self.calculate_fitness(individual) for individual in self.population]
        ranks = rankdata(fitness_scores)

        if self.selection_method == 'roulette_wheel':
            probabilities = 1 / (ranks + 1)
            probabilities /= probabilities.sum()
            for _ in range(len(self.population) // 2):
                parent_indices = random.choices(range(len(self.population)), weights=probabilities, k=2)
                parents = [self.population[idx] for idx in parent_indices]
                selected_parents.extend(parents)
        elif self.selection_method == 'tournament':
            for _ in range(len(self.population) // 2):
                # Ensure tournament_size is not larger than the population
                actual_tournament_size = min(self.tournament_size, len(self.population))
                tournament_indices = np.random.choice(range(len(self.population)), actual_tournament_size, replace=False)
                tournament_fitness = [fitness_scores[i] for i in tournament_indices]
                winner_index = np.argmin(tournament_fitness)
                selected_parents.append(self.population[winner_index])

        return selected_parents

    def crossover(self, parent1, parent2):
        """
        Applies crossover operation to two parents to create offspring.

        Parameters:
        parent1 (np.array): The first parent.
        parent2 (np.array): The second parent.

        Returns:
        tuple: A tuple containing two offspring.
        """
        crossover_point = np.random.randint(1, len(parent1))
        offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return offspring1, offspring2

    def mutate(self, individual):
        """
        Applies mutation to an individual.

        Parameters:
        individual (np.array): An array representing an individual.

        Returns:
        np.array: The mutated individual.
        """
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def run(self):
        """
        Runs the genetic algorithm.

        Returns:
        tuple: A tuple containing the best solution and its fitness.
        """
        best_solution = None
        best_fitness = float('inf')

        for _ in range(self.num_generations):
            new_population = []
            selected_parents = self.selection()

            while len(selected_parents) >= 2:
                parent1, parent2 = selected_parents.pop(), selected_parents.pop()

                for operator, para in self.opr_fun:
                    if para == 2:
                        offspring1, offspring2 = operator(parent1, parent2)
                    elif para == 1:
                        offspring1 = self.mutate(offspring1)
                        offspring2 = self.mutate(offspring2)
                    else:
                        raise ValueError("The number of parents a genetic operator uses must be 1 or 2")
                    
                new_population.extend([offspring1, offspring2])

            self.population = new_population

            for individual in self.population:
                fitness = self.calculate_fitness(individual)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = individual

        selected_variables = np.array(self.variable_list)[best_solution.astype(bool)].tolist()
        return selected_variables, best_fitness
    
def select(data, target_variable, num_generations=100, population_size=50, 
           selection_method='roulette_wheel', mutation_rate=0.01, 
           obj_fun='aic', FUN=sm.OLS, family=None, opr_fun=['crossover','mutate'], opr_para=[2,1]):
    """
    High-level function to select variables using the genetic algorithm.

    Parameters:
    data (DataFrame): The dataset to be used. Must be a pandas DataFrame.
    target_variable (str): The name of the target variable in the dataset.
    num_generations (int): The number of generations for the GA to run. 
                           More generations may lead to a more optimized solution.
    population_size (int): The size of the population in each generation. Larger populations
                           may provide more diversity but increase computational cost.
    selection_method (str): The method used for selecting individuals. Options are 'roulette_wheel'
                            or 'tournament'. 'roulette_wheel' gives all individuals a chance to be
                            selected based on fitness, while 'tournament' selects the best out of a random subset.
    mutation_rate (float): The probability of mutation in an individual, between 0 and 1. Higher mutation
                           rates can increase diversity but may lead to less stability in solutions.
    obj_fun (str or callable): The objective function to be used. Defaults to 'aic'. Other options are 'bic' and 
                               'adjusted_r2', or a custom function can be provided.
    FUN (function): The regression function to use. Default is sm.OLS. This can be any function or class
                    that follows the scikit-learn estimator interface.
    family (Optional[statsmodels.genmod.families.family]): The family of distributions to be 
            used in the model. Necessary for Generalized Linear Models (GLMs) when using statsmodels functions.
            Default is None.
    opr_fun (list of str or callable): The genetic operators to be used. Defaults to 'crossover' and 'mutate'.
                               One or more custom functions can be provided.
    opr_para (list of int): The number of parent genes each genetic operator uses. Defaults to [1,2],
                            corresponding to 'crossover' and 'mutate'.

    Returns:
    tuple: A tuple containing the list of selected variable names and the best fitness score.

    Example:
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
    >>> data = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
    >>> data['y'] = y
    >>> selected_vars, score = select(data, 'y', num_generations=50, population_size=20, FUN=sm.OLS)

    Raises:
    ValueError: If an invalid selection method or objective function is specified.
    """
    
    # Assertions to ensure valid inputs
    assert isinstance(data, DataFrame), "data must be a pandas DataFrame"
    assert target_variable in data.columns, f"target_variable '{target_variable}' not found in data"
    assert isinstance(num_generations, int) and num_generations > 0, "num_generations must be a positive integer"
    assert isinstance(population_size, int) and population_size > 0, "population_size must be a positive integer"
    assert selection_method in ['roulette_wheel', 'tournament'], "selection_method must be either 'roulette_wheel' or 'tournament'"
    assert 0 < mutation_rate < 1, "mutation_rate must be between 0 and 1"
    assert isinstance(obj_fun, str) or callable(obj_fun), "obj_fun must be either a string or a callable function"

    ga = GeneticAlgorithm(num_generations, population_size, selection_method, mutation_rate, obj_fun=obj_fun, family=family, opr_fun=opr_fun, opr_para=opr_para)
    ga.fit(data, target_variable, FUN)
    return ga.run()
