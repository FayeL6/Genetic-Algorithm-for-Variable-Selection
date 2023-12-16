
# GA: Genetic Algorithm for Variable Selection

## Description
The GA package implements a Genetic Algorithm (GA) for variable selection, catering to linear regression and Generalized Linear Models (GLMs). It aids in identifying significant predictors in these models. The package offers flexibility in selecting objective functions, such as AIC, BIC, Adjusted R2, and supports custom user-defined functions.

## Features
- **Multiple Objective Functions**: Includes AIC, BIC, Adjusted R2, and supports custom functions.
- **Compatibility**: Works with linear regression and GLMs from statsmodels and scikit-learn.
- **Customizable Selection Methods**: Features Roulette Wheel and Tournament Selection methods.
- **User-Friendly API**: Easy to integrate into various data analysis workflows.

## Installation
This package can be set up locally:

1. **Clone or Download the Repository**:
   
   - git clone https://github.com/yourusername/GA-dev.git

   - Alternatively, download the repository as a ZIP file and extract it.

2. **Navigate to the Project Directory**:
   - Change into the project directory:
     ```
     cd GA-dev
     ```

3. **(Optional) Create and activate a Python virtual environment:**
    - # Windows
      python -m venv venv
        venv\Scripts\activate

    - # Unix or MacOS
    python3 -m venv venv
    source venv/bin/activate

4. **Install Required Dependencies**:
   pip install -r requirements.txt

## Getting Started

This example demonstrates the basic usage of the GA package, showing how to initialize and run the genetic algorithm for variable selection:

```python
import GA.GA as ga
import statsmodels.api as sm
import pandas as pd

# Sample Data Preparation
# data = pd.DataFrame(...)

# Initialize the Genetic Algorithm
ga_instance = ga.GeneticAlgorithm(
 num_generations=100,
 population_size=50,
 selection_method='tournament',
 mutation_rate=0.01,
 obj_fun='AIC'
)

# Fit the GA to your dataset
ga_instance.fit(data, 'target_variable', sm.OLS)

# Run the GA
selected_variables, best_fitness = ga_instance.run()

# Output the results
print("Selected Variables:", selected_variables)
print("Best Fitness:", best_fitness)
```
## Detailed Usage

### Objective Functions

The GA package supports multiple objective functions, each serving a specific purpose:

- **AIC (Akaike Information Criterion)**: Ideal for smaller datasets or models where the number of predictors is modest. AIC helps to find the model that best explains the data with the fewest predictors.

- **BIC (Bayesian Information Criterion)**: Best suited for larger datasets, as it includes a penalty term for the number of parameters, preventing overfitting. BIC is particularly useful in model selection scenarios where overfitting is a significant concern.

- **Adjusted R2**: Useful for comparing models with different numbers of predictors. Adjusted R2 provides a way to measure the proportion of the variance explained by the model, adjusted for the number of predictors used.

### Example: GA with BIC
```python
# Genetic Algorithm with BIC
print("Running GA with BIC as the objective function...")
ga_bic = GA.GeneticAlgorithm(num_generations=1000, population_size=500, selection_method='tournament', mutation_rate=0.02, obj_fun='bic')
ga_bic.fit(data, target_variable, FUN=sm.OLS)
selected_variables_bic, best_fitness_bic = ga_bic.run()
print("GA(BIC): Selected Variables:", selected_variables_bic)
print("Best Fitness (BIC):", best_fitness_bic)
```
### Custom Objective Functions
The GA package allows for the integration of custom objective functions. This feature provides users with the flexibility to tailor the GA to their specific needs.

#### Example: GA with Custom MSE
```python
# Custom Objective Function: Root Mean Square Error with Penalty for Number of Variables
def custom_rmse(obj_fun_input):
    y, prediction, num_of_variables = obj_fun_input
    mse = np.mean((y - prediction) ** 2)
    rmse = np.sqrt(mse)
    penalty = 0.1 * num_of_variables  # Penalty factor can be adjusted
    return rmse + penalty

# Genetic Algorithm with Custom Objective Function
print("Running GA with a custom RMSE objective function...")
ga_custom = GA.GeneticAlgorithm(num_generations=1000, population_size=500, selection_method='tournament', mutation_rate=0.02, obj_fun=custom_rmse)
ga_custom.fit(data, target_variable, FUN=sm.OLS)
selected_variables_custom, best_fitness_custom = ga_custom.run()
print("GA(Custom RMSE): Selected Variables:", selected_variables_custom)
print("Best Fitness (Custom RMSE):", best_fitness_custom)
```


### Selection, Crossover, and Mutation

These genetic operations are crucial for evolving the population towards an optimal solution:

- **Selection**: Chooses the fittest individuals as parents for the next generation. The GA package supports different selection methods like 'roulette_wheel' and 'tournament', each offering a unique way to select parent solutions.

- **Crossover**: Combines features from two parent solutions to create new offspring. Crossover aims to mix and match the best traits of parent solutions in the hope of creating even better offspring.

- **Mutation**: Introduces variability into the population, aiding in exploring new solutions and maintaining diversity. This operation prevents the algorithm from prematurely converging on sub-optimal solutions.

### `GeneticAlgorithm` Class - API Reference

The `GeneticAlgorithm` class is the centerpiece of the GA package, offering a flexible and robust implementation for variable selection.

#### Parameters:

- `num_generations` (int): The number of generations the algorithm will run. A higher number allows more evolution of the solution.
- `population_size` (int): The size of the population in each generation. A larger population size offers more diversity but increases computational cost.
- `selection_method` (str): The method used for selecting individuals. Options include 'roulette_wheel' and 'tournament'.
- `mutation_rate` (float): The probability of mutation in an individual, between 0 and 1. Mutation introduces randomness and diversity into the population.
- `_obj_fun` (str or callable): The objective function to be used. Options include 'AIC', 'BIC', 'Adjusted_R2', or a custom function provided by the user.

#### Methods:

##### `fit(data, target_variable, FUN)`

Prepares the algorithm with the dataset and regression function.

- `data` (DataFrame): The dataset to be used.
- `target_variable` (str): The name of the target variable in the dataset.
- `FUN` (function): The regression function (e.g., `sm.OLS` for ordinary least squares regression).

##### `run()`

Executes the genetic algorithm and returns the best set of variables and their fitness.

- Returns: A tuple (`selected_variables`, `best_fitness`) where `selected_variables` is a list of selected variable names and `best_fitness` is their fitness score.

## Examples

### Basic Example

This example demonstrates how to use the GA package for variable selection with a simple dataset.

```python
import pandas as pd
import statsmodels.api as sm
import GA.GA as ga

# Sample data preparation
data = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [5, 4, 3, 2, 1],
    'y': [2, 1, 3, 4, 5]
})
target_variable = 'y'

# Initialize the Genetic Algorithm
ga_instance = ga.GeneticAlgorithm(
    num_generations=10,
    population_size=5,
    selection_method='tournament',
    mutation_rate=0.01,
    obj_fun='AIC'
)

# Fit the GA to your dataset
ga_instance.fit(data, target_variable, FUN=sm.OLS)

# Run the GA
selected_variables, best_fitness = ga_instance.run()

# Output the results
print("Selected Variables:", selected_variables)
print("Best Fitness:", best_fitness)
```
## Example with Custom Objective Function

```python
import numpy as np

# Define a custom objective function
def custom_mse(obj_fun_input):
    y, prediction, _ = obj_fun_input
    mse = np.mean((y - prediction) ** 2)
    return mse

# Initialize the Genetic Algorithm with the custom objective function
ga_custom = ga.GeneticAlgorithm(
    num_generations=10,
    population_size=5,
    selection_method='roulette_wheel',
    mutation_rate=0.01,
    obj_fun=custom_mse  # Using the custom objective function
)

# Fit and run the GA as before
ga_custom.fit(data, target_variable, FUN=sm.OLS)
selected_variables_custom, best_fitness_custom = ga_custom.run()

# Output the results with the custom objective function
print("Selected Variables with Custom MSE:", selected_variables_custom)
print("Best Fitness (Custom MSE):", best_fitness_custom)
```

## Example with Custom Genetic Operator Function

```python
import numpy as np

# Define a custom crossover function
def unif_crossover(parent1, parent2, probability=0.5):
    mask = np.random.rand(len(parent1)) < probability
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)
    return offspring1, offspring2

# Initialize the Genetic Algorithm with the custom genetic operator function
ga_custom = ga.GeneticAlgorithm(
    num_generations=10,
    population_size=5,
    selection_method='roulette_wheel',
    mutation_rate=0.01,
    opr_fun=unif_crossover,  # Using the custom objective function
    opr_para=[2,1]  # note the number of parents the operator needs
)

# Fit and run the GA as before
ga_custom.fit(data, target_variable, FUN=sm.OLS)
selected_variables_custom, best_fitness_custom = ga_custom.run()

# Output the results with the custom objective function
print("Selected Variables with Custom MSE:", selected_variables_custom)
print("Best Fitness (Custom MSE):", best_fitness_custom)
```

## License
This project is for educational purposes only and is not licensed for other uses.

## Contact
For questions, suggestions, or feedback, please contact us:
jiarong_zhou2023@berkeley.edu
fangyuan_li@berkeley.edu
hannah_neumann@berkeley.edu

