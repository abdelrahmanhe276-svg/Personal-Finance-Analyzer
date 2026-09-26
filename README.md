# Personal Finance Analyzer

Personal Finance Analyzer is a Python package that analyze personal finance and models monthly net values using ARIMA time series models.
## Overview
The analyzer loads transactions data from a csv file, cleans and validates it, analyzes monthly income and expenses, and evaluates arima models using monthly net values.

The project produces summary tables, diagnostic plots, model evaluation metrics, residual diagnostics and comparisons between actual and predicted.

## Key features

The package:
- Validates and cleans data
- Calculates total monthly income, expenses and Net
- Analyzes income and expense by category
- Identifies potential influencial points using IQR method
- Split monthly data into train and test sets
- uses auto arima to suggest a model based on AIC and BIC
- Evaluates manually the specified ARIMA models
- Compares models using MAE, RMSE, mean directional accuracy, AIC and BIC
- Generates ACF and PACF for the suggested model
- Performs the Ljung-Box test on arima residuals
- Performs the Shapiro-wilk test on ARIMA residuals
- Generates a residual histogram and Q-Q Plot
- Compares actual and predicted Net values for AR(4) model
- Generates 95% prediction intervals for AR(4) predictions

## Installation

python 3.10 or newer is required

from the package directory install the package using :
```
uv pip install -e .
```
the required dependenices are automatically installed from pyproject.toml

The main dependencies are:

- Numpy
- pandas
- Matplotlib
- statsmodels
- pmdarima
- SciPy

## Input format

The csv file contains the following columns:
- Date
- Transaction Description
- Categort
- Amount 
- Type

The type column must contain either income or expense

The dataset is:
```
data/Personal_Finance_Dataset.csv
```
## Usage
Run the package using : 

```
uv run -m personal_finance_analyzer
```
Transaction file can also be supplied using :

```
uv run -m personal_finance_analyzer path/to/transactions.csv
```

## Output
The program creates an output directory containing analysis tables and figures

### tables
- `monthly sumary.csv`
- `expense by category.csv`
- `income by category.csv` 

### Figures
- `category expenses.png`
- `monthly summary.png`
- `acf.png`
- `pacf.png`
- `ar4 predictions.png`
- `ar4 QQ-plot`
- `ar4 residual histogram`

## Methods
**Financial analysis**:
Net is calculated Income - Expenses. 
Monthly income and expenses are aggregated from the transaction data. Income and expenses are also analyzed by category
Potential influencial points are identified by IQR method.

## Time-Series Evaluation
The monthly net series is divided into training and test sets, where the final 6 months are used as test set.
Auto-Arima is used to suggest candidate ARIMA model. Additional Arima models are evaluated manually.
The models are compared using:
- **MAE**: Mean absolute error
- **RMSE**: Root mean square error
- **MDA**: Mean directional accuracy
- **AIC**: Akaike Information Criterion
- **BIC**: Bayesian Information Criterion

Lower MAE, RMSE,AIC and Bic is preferred while higher MDA indicate better model performance.

### Residual Diagnostics
The residual of the selected AR(4) model are evaluated using Ljung box and Shapiro-Wilk test
The residual histogram and QQ plot are also generated to examine the residual distribution.

### AR(4) prediction
The residuals of the selected AR(4) model is evaluated on test period using the plot that compares the actual vs predicted net values and include 95% prediction interval to forecast uncertanity.