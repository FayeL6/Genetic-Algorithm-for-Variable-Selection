from GA.GA import select
import pandas as pd
import numpy as np
import statsmodels.api as sm
import GA.GA as GA
from sklearn.linear_model import LinearRegression, lasso_path

# Function to load dataset
def load_dataset(file_path, delimiter=','):
    try:
        data = pd.read_csv(file_path, delimiter=delimiter)
        data.columns = [col.strip().replace(' ', '_') for col in data.columns]
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Function to run Lasso and get selected variables and AIC
def run_lasso(X, y):
    X_norm = X / X.std(axis=0)
    y_norm = y / y.std(axis=0)
    alphas_lasso, coefs_lasso, _ = lasso_path(X_norm, y_norm, eps=0.001)
    aic_values = np.zeros(coefs_lasso.shape[1])
    for i in range(coefs_lasso.shape[1]):
        selected_variables = np.array(X.columns)[coefs_lasso[:, i] != 0]
        if len(selected_variables) > 0:
            model = LinearRegression().fit(X[selected_variables], y)
            prediction = model.predict(X[selected_variables])
            aic_values[i] = GA.calculate_aic([y, prediction, len(selected_variables)])
        else:
            aic_values[i] = np.inf
    best_index = np.argmin(aic_values)
    return X.columns[coefs_lasso[:, best_index] != 0], aic_values[best_index]

# Function to run Greedy Algorithm and get selected variables and AIC
def run_greedy(X, y):
    X_new = sm.add_constant(X)
    best_aic = np.inf
    best_vars = []
    fitted_model = sm.OLS(y, X_new).fit()  # Initial model fitting

    while len(X_new.columns) > 1:  # Ensure there's more than just the constant term
        var_max_pval = fitted_model.pvalues.idxmax()  # Get index of max p-value

        # Check if the highest p-value is significant
        if var_max_pval != 'const' and fitted_model.pvalues[var_max_pval] > 0.05:
            X_new = X_new.drop(var_max_pval, axis=1)  # Drop the variable
            fitted_model = sm.OLS(y, X_new).fit()  # Refit the model with updated X_new
        else:
            break

        prediction = fitted_model.predict(X_new)
        aic = GA.calculate_aic([y, prediction, len(X_new.columns) - 1])
        if aic < best_aic:
            best_aic = aic
            best_vars = list(X_new.columns[1:])  # Exclude the constant term

    return best_vars, best_aic

# Function to run GA and get selected variables and AIC
def run_ga(X, y, target_variable, num_generations=5000, population_size=2000, selection_method='tournament', mutation_rate=0.04, obj_fun='aic'):
    data = X.copy()
    data[target_variable] = y
    return GA.select(data, target_variable, num_generations=num_generations, population_size=population_size, selection_method=selection_method, mutation_rate=mutation_rate, obj_fun=obj_fun)


# Function to generate simulated data
def generate_simulated_data(num_points=500, num_features=20):
    X = pd.DataFrame(np.random.rand(num_points, num_features))
    true_coefficients = np.random.uniform(-5, 5, size=num_features)
    true_coefficients[10:] = np.zeros(10)
    noise = np.random.normal(0, 0.5, size=(num_points, ))
    y = X @ true_coefficients + noise
    return X, y

def prepare_dataset(file_path, delimiter, target_var):
    try:
        data = pd.read_csv(file_path, delimiter=delimiter)
        data.columns = [col.strip().replace(' ', '_') for col in data.columns]
        print("Columns after preprocessing:", data.columns.tolist())  # Add this line to check column names
        X = data.drop(target_var, axis=1)
        y = data[target_var]
        return X, y
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

# Use this function to load the Admission dataset
admission_X, admission_y = prepare_dataset('data/Admission_Predict.csv', ',', 'Chance_of_Admit')

# Datasets
datasets = {
    "Baseball": prepare_dataset('data/baseball.dat', '\s+', 'salary'),
    "Admission": prepare_dataset('data/Admission_Predict.csv', ',', 'Chance_of_Admit'),
    "Wine": prepare_dataset('data/winequality-red.csv', ';', 'quality'),
    "Simulated": generate_simulated_data()
}

# Mapping of dataset names to their target variable names
target_var_map = {
    "Baseball": "salary",
    "Admission": "Chance_of_Admit",
    "Wine": "quality",
    "Simulated": "y"
}

# Load the Admission dataset
file_path = 'data/Admission_Predict.csv'
try:
    admission_data = pd.read_csv(file_path)
except Exception as e:
    print("Error loading file:", e)
    exit()

# Check original column names
print("Original column names:", admission_data.columns.tolist())

# Rename columns if necessary
admission_data.columns = [col.strip().replace(' ', '_') for col in admission_data.columns]
print("Renamed column names:", admission_data.columns.tolist())

# Check if 'Chance_of_Admit' is in the columns after renaming
if 'Chance_of_Admit' not in admission_data.columns:
    print("Error: 'Chance_of_Admit' column not found after renaming.")
else:
    print("'Chance_of_Admit' column found.")

# Comparison Table
results = []
for name, data_tuple in datasets.items():
    if data_tuple is not None:
        X, y = data_tuple

        # Lasso
        lasso_vars, lasso_aic = run_lasso(X, y)

        # Greedy
        greedy_vars, greedy_aic = run_greedy(X, y)

        # GA
        ga_vars, ga_aic = run_ga(X, y, target_var_map[name])

        # Append results
        results.append({
            "Dataset": name,
            "Lasso Variables": len(lasso_vars),
            "Lasso AIC": lasso_aic,
            "Greedy Variables": len(greedy_vars),
            "Greedy AIC": greedy_aic,
            "GA Variables": len(ga_vars),
            "GA AIC": ga_aic
        })

results_df = pd.DataFrame(results)
print(results_df)
