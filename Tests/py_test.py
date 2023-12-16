import sys
import os
# Add the parent directory to the sys.path to find the GA module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from GA.GA import calculate_aic, calculate_bic, calculate_adjusted_r2
from GA.GA import GeneticAlgorithm

# Test objective functions (assuming they are available globally)

def test_calculate_aic():
    y = np.array([2, 4, 6, 8])
    prediction = np.array([2, 4, 6, 8])
    num_of_variables = 1
    expected_result = -140.1746135564686
    result = calculate_aic([y, prediction, num_of_variables])
    assert np.isclose(result, expected_result, atol=1e-5)

def test_calculate_bic():
    y = np.array([2, 4, 6, 8])
    prediction = np.array([2, 4, 6, 8])
    num_of_variables = 1
    expected_result = -148.3334966398283
    result = calculate_bic([y, prediction, num_of_variables])
    assert np.isclose(result, expected_result, atol=1e-5)

def test_calculate_adjusted_r2():
    # Test calculate_adjusted_r2 function with sample data
    y = np.array([1, 2, 3, 4])
    prediction = np.array([1.1, 1.9, 3.1, 3.9])
    num_of_variables = 1
    X = np.array([[1, 2], [2, 3], [3, 4]])
    expected_result = 0.988  
    result = calculate_adjusted_r2([y, prediction, num_of_variables, X])
    assert np.isclose(result, expected_result)

# Test GeneticAlgorithm class methods

@pytest.fixture
def sample_data():
    # Generate synthetic sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.rand(100)
    data = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(5)])
    data['Target'] = y
    return data

def test_GeneticAlgorithm_initialization():
    ga = GeneticAlgorithm(100, 50, 'roulette_wheel', 0.01, 3, 'aic')  # Here, 3 is passed as tournament_size
    assert ga.tournament_size == 3
    assert ga.num_generations == 100
    assert ga.population_size == 50
    assert ga.selection_method == 'roulette_wheel'
    assert ga.mutation_rate == 0.01
    assert ga.obj_fun == calculate_aic  

def test_GeneticAlgorithm_fit(sample_data):
    data = sample_data
    ga = GeneticAlgorithm(100, 50, 'roulette_wheel', 0.01, 3, 'aic')
    ga.fit(data, 'Target', sm.OLS)
    assert ga.data.equals(data)
    assert ga.target_variable == 'Target'
    assert ga.FUN == sm.OLS
    assert len(ga.variable_list) == data.shape[1] - 1

def test_GeneticAlgorithm_run(sample_data):
    data = sample_data
    ga = GeneticAlgorithm(10, 5, 'roulette_wheel', 0.01, 3, 'aic')
    ga.fit(data, 'Target', sm.OLS)
    selected_variables, best_fitness = ga.run()
    assert isinstance(selected_variables, list)
    assert isinstance(best_fitness, float)

def test_GeneticAlgorithm_small_dataset():
    np.random.seed(0)
    small_data = pd.DataFrame({
        'Feature1': np.random.rand(10),
        'Feature2': np.random.rand(10),
        'Target': np.random.rand(10)
    })
    ga_small = GeneticAlgorithm(num_generations=10, population_size=4, selection_method='roulette_wheel', mutation_rate=0.01)
    ga_small.fit(small_data, 'Target', FUN=sm.OLS)
    selected_vars_small, fitness_small = ga_small.run()
    assert isinstance(selected_vars_small, list)
    assert isinstance(fitness_small, float)

def test_GeneticAlgorithm_correlated_features():
    correlated_data = pd.DataFrame({
        'Feature1': np.arange(100),
        'Feature2': np.arange(100) + np.random.normal(0, 5, 100),
        'Target': np.arange(100) * 2 + np.random.normal(0, 10, 100)
    })
    ga_corr = GeneticAlgorithm(num_generations=20, population_size=50, selection_method='tournament', mutation_rate=0.02)
    ga_corr.fit(correlated_data, 'Target', FUN=sm.OLS)
    selected_vars_corr, fitness_corr = ga_corr.run()
    assert isinstance(selected_vars_corr, list)
    assert isinstance(fitness_corr, float)

def test_GeneticAlgorithm_high_mutation_rate():
    normal_data = pd.DataFrame(np.random.rand(100, 5), columns=[f'Feature_{i}' for i in range(5)])
    normal_data['Target'] = np.random.rand(100)
    ga_extreme = GeneticAlgorithm(num_generations=50, population_size=100, selection_method='roulette_wheel', mutation_rate=0.5)
    ga_extreme.fit(normal_data, 'Target', FUN=sm.OLS)
    selected_vars_extreme, fitness_extreme = ga_extreme.run()
    assert isinstance(selected_vars_extreme, list)
    assert isinstance(fitness_extreme, float)

# Run the tests
if __name__ == "__main__":
    pytest.main()

