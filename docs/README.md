
# GA: Genetic Algorithm for Variable Selection

## Description
This package implements a Genetic Algorithm (GA) for variable selection, specifically tailored for linear regression and Generalized Linear Models (GLMs). It assists the user in identifying significant predictors within these models. The package offers flexibility in choosing objective functions, including AIC, BIC, Adjusted R2, and supports custom user-defined functions for more tailored optimization.

## Features
- Multiple objective functions: AIC, BIC, Adjusted R2, and custom user-defined functions.
- Compatible with linear regression models and GLMs from statsmodels and scikit-learn.
- Customizable selection methods: Roulette Wheel and Tournament Selection.
- User-friendly API for straightforward integration into various data analysis workflows.

## Installation
This package is not available on PyPI, but can be set up locally by following these steps:

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

## Basic usage 

```python
import pandas as pd
import statsmodels.api as sm
from GA import select  # Import the select function from the GA package

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Use the select function for variable selection in linear regression or GLMs
selected_variables, best_fitness = select(
    data, 
    target_variable='your_target_column',
    num_generations=100, 
    population_size=50,
    selection_method='tournament',
    obj_fun='aic',
    FUN=sm.OLS  # or another regression function
)

print("Selected Variables:", selected_variables)
print("Best Fitness:", best_fitness)
```

## Documentation
For a detailed guide on all the functionalities and usage instructions, please refer to the documentation.md file.

## License
This project is for educational purposes only and is not licensed for other uses.

## Contact
For questions, suggestions, or feedback, please contact us:
jiarong_zhou2023@berkeley.edu
fangyuan_li@berkeley.edu
hannah_neumann@berkeley.edu

