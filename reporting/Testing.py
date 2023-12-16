from sklearn.linear_model import lasso_path, LinearRegression, BayesianRidge, Ridge
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GA.GA as GA
from sklearn import linear_model

# Load Data
file_path = 'data/baseball.dat'
try:
    data = pd.read_csv(file_path, delim_whitespace=True)
except Exception as e:
    print("Error loading file:", e)
    exit()

# Data Preparation
target_variable = 'salary'
variables = [col for col in data.columns if col != target_variable]
X = data[variables]
y = data[target_variable]

# Normalize data for LASSO
X_norm = X / X.std(axis=0)
y_norm = y / y.std(axis=0)

# LASSO Model Selection
print("Computing regularization path using LASSO...")
alphas_lasso, coefs_lasso, _ = lasso_path(X_norm, y_norm, eps=0.001)

lin_model_sel_var_AIC = np.zeros(coefs_lasso.shape[1])
for i in range(coefs_lasso.shape[1]):
    selected_variables = np.array(variables)[coefs_lasso[:, i] != 0]
    if len(selected_variables) > 0:
        model = LinearRegression().fit(X[selected_variables], y)
        prediction = model.predict(X[selected_variables])
        lin_model_sel_var_AIC[i] = GA.calculate_aic([y, prediction, len(selected_variables)])
    else:
        lin_model_sel_var_AIC[i] = np.inf

best_lasso_index = np.argmin(lin_model_sel_var_AIC)
selected_variables_best_AIC = np.array(variables)[coefs_lasso[:, best_lasso_index] != 0]
lin_model_sel_var_best_AIC = LinearRegression().fit(X[selected_variables_best_AIC], y)

print('Lasso: Selected Variables for the Model with the best AIC:', selected_variables_best_AIC)

# Greedy Algorithm Implementation
print("Running Greedy algorithm for variable selection...")
X_new = sm.add_constant(X)
greedy_aic = np.zeros(X_new.shape[1] - 1)
greedy_var_to_keep = []

for i in range(X_new.shape[1] - 1):
    fitted_model = sm.OLS(y, X_new).fit()
    var_max_pval = np.argmax(fitted_model.pvalues)
    greedy_var_to_keep.append(X_new.columns.tolist())
    
    prediction = fitted_model.predict(X_new)
    greedy_aic[i] = GA.calculate_aic([y, prediction, len(X_new.columns) - 1])
    
    X_new = X_new.drop(X_new.columns[var_max_pval], axis=1)

best_greedy_index = np.argmin(greedy_aic)
print('Greedy: Selected Variables for the Model with the best AIC:', greedy_var_to_keep[best_greedy_index])

# Genetic Algorithm with sm.OLS
print("Running GA with sm.OLS...")
ga_ols = GA.GeneticAlgorithm(num_generations=2000, population_size=2000, selection_method='tournament', mutation_rate=0.02)
ga_ols.fit(data, target_variable, FUN=sm.OLS)
selected_variables_ols, best_fitness_ols = ga_ols.run()
print("GA(sm.OLS): Selected Variables:", selected_variables_ols)
print("Best Fitness (AIC):", best_fitness_ols)

# Genetic Algorithm with Linear Regression
print("Running GA with Linear Regression...")
ga_lr = GA.GeneticAlgorithm(num_generations=2000, population_size=2000, selection_method='tournament', mutation_rate=0.03)
ga_lr.fit(data, target_variable, FUN=LinearRegression)
selected_variables_lr, best_fitness_lr = ga_lr.run()
print("GA(Linear Regression): Selected Variables:", selected_variables_lr)
print("Best Fitness (AIC):", best_fitness_lr)

# Plotting for Comparison
num_variables_selected_lasso = np.sum(coefs_lasso != 0, axis=0)
num_variables_selected_greedy = [len(vars) for vars in greedy_var_to_keep]
num_variables_selected_ga = len(selected_variables_ols)

plt.figure(figsize=(12, 8))
plt.plot(num_variables_selected_lasso, lin_model_sel_var_AIC, marker='o', color='blue', label='Lasso AIC')
plt.scatter(num_variables_selected_lasso[best_lasso_index], lin_model_sel_var_AIC[best_lasso_index], color='navy', label='Best Lasso AIC')
plt.plot(num_variables_selected_greedy, greedy_aic, marker='x', color='green', label='Greedy AIC')
plt.scatter(num_variables_selected_greedy[best_greedy_index], greedy_aic[best_greedy_index], color='darkgreen', label='Best Greedy AIC')
plt.scatter(num_variables_selected_ga, best_fitness_ols, marker='^', color='red', label='GA Best AIC')
plt.xlabel('Number of Variables Selected')
plt.ylabel('AIC')
plt.title('Comparison of Lasso, Greedy Algorithm, and GA: AIC vs. Number of Variables')
plt.legend()
plt.show()

# Genetic Algorithm with BIC
print("Running GA with BIC as the objective function...")
ga_bic = GA.GeneticAlgorithm(num_generations=1000, population_size=500, selection_method='tournament', mutation_rate=0.02, obj_fun='bic')
ga_bic.fit(data, target_variable, FUN=sm.OLS)
selected_variables_bic, best_fitness_bic = ga_bic.run()
print("GA(BIC): Selected Variables:", selected_variables_bic)
print("Best Fitness (BIC):", best_fitness_bic)

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

# Genetic algorithm with Bayesian Ridge Regression
print("Running GA with Bayesian Ridge Regression...")
ga_br = GA.GeneticAlgorithm(num_generations=1000, population_size=500, selection_method='tournament', mutation_rate=0.02)
ga_br.fit(data, target_variable, FUN=linear_model.BayesianRidge)
selected_variables_br, best_fitness_br = ga_br.run()
print("GA(Bayesian Ridge): Selected Variables:", selected_variables_br)
print("Best Fitness (AIC):", best_fitness_br)

# Genetic algorithm with Ridge Regression
ga_ridge = GA.GeneticAlgorithm(num_generations=1000, population_size=500, selection_method='tournament', mutation_rate=0.02)
ga_ridge.fit(data, target_variable, FUN=linear_model.Ridge(alpha=0.5))
selected_variables_ridge, best_fitness_ridge = ga_ridge.run()
print("GA(Ridge): Selected Variables:", selected_variables_ridge)
print("Best Fitness (AIC):", best_fitness_ridge)

# Genetic algorithm with Generalized Linear Models (GLM)
print("Running GA with Generalized Linear Models...")
ga_glm = GA.GeneticAlgorithm(num_generations=1000, population_size=500, selection_method='tournament', mutation_rate=0.02)
ga_glm.fit(data, target_variable, FUN=sm.GLM, family=sm.families.Gaussian())
selected_variables_glm, best_fitness_glm = ga_glm.run()
print("GA(GLM): Selected Variables:", selected_variables_glm)
print("Best Fitness (AIC):", best_fitness_glm)


# Custom Genetic Operator Function: Uniform Crossover and Cauchy Mutation
def uniform_crossover(parent1, parent2, probability=0.5):
    """
    Apply uniform crossover to two binary parents.

    Parameters:
    - parent1: The first parent.
    - parent2: The second parent.
    - probability: Probability of selecting a gene from the first parent.

    Returns:
    - Two offspring generated by uniform crossover.
    """
    # Ensure both parents have the same length
    assert len(parent1) == len(parent2), "Parents must have the same length"

    # Generate a mask of random values with the same length as the parents
    mask = np.random.rand(len(parent1)) < probability

    # Create offspring by selecting genes based on the mask
    offspring1 = np.where(mask, parent1, parent2)
    offspring2 = np.where(mask, parent2, parent1)

    return offspring1, offspring2
    
def cauchy_mutation(individual, scale=0.1, alpha=0.5):
    """
    Apply Cauchy mutation to an individual.

    Parameters:
    - individual: The individual to be mutated.
    - scale: The scale parameter controlling the spread of the Cauchy distribution.
    - alpha: The location parameter of the Cauchy distribution.

    Returns:
    - Mutated individual.
    """
    mutation = scale * np.tan(np.pi * (np.random.rand(len(individual)) - 0.5))
    mutated_individual = individual + alpha * mutation
    mutated_individual = abs(np.round(mutated_individual))
    return np.where(mutated_individual>1, 1, mutated_individual)

# Genetic Algorithm with Custom Genetic Operator Function
print("Running GA with two custom genetic operator functions...")
ga_opr_custom = GA.GeneticAlgorithm(num_generations=1000, population_size=500, selection_method='tournament', mutation_rate=0.02, opr_fun=[uniform_crossover, cauchy_mutation], opr_para=[2,1])
ga_opr_custom.fit(data, target_variable, FUN=sm.OLS)
selected_variables_opr_custom, best_fitness_opr_custom = ga_opr_custom.run()
print("GA(Custom operators): Selected Variables:", selected_variables_opr_custom)
print("Best Fitness (Custom operators):", best_fitness_opr_custom)