from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

import numpy as np
class FinanceAnalyzer:
    """analyze transactions, identify influencial points, and evaluate the ARIMA
        forecasts"""
    def __init__(self, transactions):
        """store copy of transaction dataset"""
        self.transactions = transactions.copy()

    def summary(self):
        """calculate total sum, expenses, net income and number of transactions"""
        income = self.transactions[self.transactions["Type"] == "Income"]
        income = income['Amount'].sum()
        
        expenses = self.transactions[self.transactions["Type"] == "Expense"]
        expenses = expenses["Amount"].sum()
        net_income = income - expenses
        return {
            "transactions": len(self.transactions),"total_income": float(income),"total_expenses": float(expenses),"net_income": float(net_income)}

    def monthly_summary(self):
        """calculate monthly income , expenses and net"""
        data = self.transactions.copy()
        data["Month"] = data["Date"].dt.to_period("M").astype(str)
        monthly = (
            data.groupby(["Month", "Type"])["Amount"].sum().unstack(fill_value=0))
        monthly = monthly[["Income", "Expense"]]
        monthly["Net"] = monthly["Income"] - monthly["Expense"]
        return monthly.reset_index()

    def expense_by_category(self):
        """calculate the expense and group by category"""
        expenses = self.transactions[self.transactions["Type"] == "Expense"]
        grouped = expenses.groupby("Category")["Amount"].agg(["count", "sum", "mean"])
    
        return grouped.reset_index()

    def income_by_category(self):
        """calculate the income by category"""
        income = self.transactions[self.transactions["Type"] == "Income"]
        grouped = income.groupby("Category")["Amount"].agg(["count", "sum", "mean"])
        grouped = grouped.rename(
            columns={"count": "Transactions", "sum": "Total", "mean": "Average"}
        )

        return grouped.sort_values("Total", ascending=False).reset_index()

    def large_expenses(self):
        """identify potential influencial points in expenses"""
        expenses = self.transactions[self.transactions["Type"] == "Expense"].copy()
        q1 = expenses["Amount"].quantile(0.25)
        q3 = expenses["Amount"].quantile(0.75)
        threshold = q3 + 1.5*(q3 - q1)
        result = expenses[expenses["Amount"] > threshold]
        return result

    def large_incomes(self):
        """identify potential influencial points in incomes"""
        incomes = self.transactions[
            self.transactions["Type"] == "Income"
        ].copy()
    
        q1 = incomes["Amount"].quantile(0.25)
        q3 = incomes["Amount"].quantile(0.75)
        threshold = q3 + 1.5 * (q3 - q1)
    
        result = incomes[incomes["Amount"] > threshold].copy()
    
        return result
    

    def split_train_test(self, test_size=6):
        """split the data into train and test with test set size of 6 months"""
        monthly = self.monthly_summary()
    
        train = monthly.iloc[:-test_size].copy()
        test = monthly.iloc[-test_size:].copy()
    
        return train, test
    
    def evaluate_auto_arima(self, information_criterion="aic", test_size=6):
        """selects and evaluate an arima model on test set"""
        train, test = self.split_train_test(test_size)
    
        series = train["Net"]
    
        model = auto_arima(
            series,
            seasonal=False,
            information_criterion=information_criterion
        )
        # seasonal = false since plots did not show any signs of seasonality
        # information criterion is used to asess different arima models
    
        predictions = model.predict(n_periods=len(test))
        actual = test["Net"].to_numpy()
        predicted = np.asarray(predictions)
        actual_direction = np.sign(np.diff(actual)) # direction for mda
        predicted_direction = np.sign(np.diff(predicted))
        
        mda = np.mean(actual_direction == predicted_direction) * 100
    
        mae = np.mean(np.abs(actual - predicted))
    
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
        results = test[["Month", "Net"]].copy()
        results["Predicted Net"] = predicted
    
        return {
            "order": model.order,
            "aic": float(model.aic()),
            "bic": float(model.bic()),
            "mae": float(mae),
            "rmse": float(rmse),
            "results": results,
            "mda": float(mda)
        }
    
    def evaluate_arima(self, p, d, q, test_size=6):
        """fit arima model and evaluate on test set"""
        train, test = self.split_train_test(test_size)
    
        model = ARIMA(train["Net"], order=(p, d, q))
        fitted_model = model.fit()
    
        predictions = fitted_model.forecast(steps=len(test))
    
        confidence_interval = fitted_model.get_forecast(
            steps=len(test)
        ).conf_int(alpha=0.05)
        ## getting the confidence intervals of the forecasts
        actual = test["Net"].to_numpy()
        predicted = np.asarray(predictions)
    
        mae = np.mean(np.abs(actual - predicted))
    
        rmse = np.sqrt(
            np.mean((actual - predicted) ** 2)
        )
    
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
    
        mda = np.mean(
            actual_direction == predicted_direction
        ) * 100
    
        results = test[["Month", "Net"]].copy()
        results["Predicted Net"] = predicted
    
        results["Lower CI"] = (
            confidence_interval.iloc[:, 0].to_numpy()
        )
    
        results["Upper CI"] = (
            confidence_interval.iloc[:, 1].to_numpy()
        )
    
        return {
            "order": (p, d, q),
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "mae": float(mae),
            "rmse": float(rmse),
            "results": results,
            "mda": float(mda)
        }
    
    def ljung_box_test(self, p, d, q, test_size=6, lag=10):
        """compute the ljung-box test"""
        train, test = self.split_train_test(test_size)
    
        model = ARIMA(
            train["Net"],
            order=(p, d, q)
        )
    
        fitted_model = model.fit()
    
        result = acorr_ljungbox(
            fitted_model.resid,
            lags=[lag],
            model_df=p + q,
            return_df=True
        )
    
        return {
            "lag": lag,
            "statistic": float(result["lb_stat"].iloc[0]),
            "p_value": float(result["lb_pvalue"].iloc[0])
        }
  
    def shapiro_wilk_test(self, p, d, q, test_size=6):
        """compute the Shapiro-Wilk test"""
        train, test = self.split_train_test(test_size)
    
        model = ARIMA(
            train["Net"],
            order=(p, d, q)
        )
    
        fitted_model = model.fit()
    
        statistic, p_value = shapiro(fitted_model.resid)
    
        return {
            "statistic": float(statistic),
            "p_value": float(p_value)
        }
    
    
    def arima_residuals(self, p, d, q, test_size=6):
        """computes residuals of the arima to be used in qq-plot and the histogram"""
        train, test = self.split_train_test(test_size)
    
        model = ARIMA(
            train["Net"],
            order=(p, d, q)
        )
    
        fitted_model = model.fit()
    
        return fitted_model.resid
    
