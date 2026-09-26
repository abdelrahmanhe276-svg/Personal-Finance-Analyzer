from pathlib import Path

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import probplot
class FinanceVisualizer:

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def _save_figure(self, fig, output_path):
        """save a figure to the output path"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return output_path

    def save_category_expenses(self, output_path):
        """plot the total expense vs the category and saves it"""
        categories = self.analyzer.expense_by_category().sort_values("sum")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(categories["Category"], categories["sum"])
        ax.set_xlabel("Total expense")
        ax.set_ylabel("Category")
        ax.set_title("Expenses by category")
        return self._save_figure(fig, output_path)

    def save_monthly_summary(self, output_path):
        """creates and save the monthly income,expense and net"""
        monthly = self.analyzer.monthly_summary()
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(monthly["Month"], monthly["Income"], label="Income")
        ax.plot(monthly["Month"], monthly["Expense"], label="Expense")
        ax.plot(monthly["Month"], monthly["Net"], label="Net")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount")
        ax.set_title("Monthly income, expense and net")
        tick_positions = range(0, len(monthly), max(1, len(monthly) // 10))
        ax.set_xticks(list(tick_positions))
        ax.set_xticklabels(
            monthly.loc[list(tick_positions), "Month"],
            rotation=45,
            ha="right"
        )
        ax.legend()
        return self._save_figure(fig, output_path)

    def save_ar4_predictions(self, output_path, test_size=6):
        """Plot the predictions vs actual net"""
        evaluation = self.analyzer.evaluate_arima(
            p=4,
            d=0,
            q=0,
            test_size=test_size
        )
    
        results = evaluation["results"]
    
        fig, ax = plt.subplots(figsize=(9, 5))
    
        ax.plot(
            results["Month"],
            results["Net"],
            label="Actual Net"
        )
    
        ax.plot(
            results["Month"],
            results["Predicted Net"],
            label="Predicted Net"
        )
    
        ax.fill_between(
            results["Month"],
            results["Lower CI"],
            results["Upper CI"],
            alpha=0.2,
            label="95% confidence interval"
        )
    
        ax.set_xlabel("month")
        ax.set_ylabel("Net")
        ax.set_title("AR(4): actual vs predicted Net")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

        return self._save_figure(fig, output_path)

    def save_acf(self, output_path, test_size=6, lags=24):
        """plot the acf to help determine the order of MA"""
        train, test = self.analyzer.split_train_test(test_size)
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_acf(train["Net"], lags=lags, ax=ax)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("ACF of Training Monthly Net")
        return self._save_figure(fig, output_path)

    def save_pacf(self, output_path, test_size=6, lags=24):
        """plot the Pacf to help determine the order of AR"""
        train, test = self.analyzer.split_train_test(test_size)
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_pacf(train["Net"], lags=lags, ax=ax)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Partial Autocorrelation")
        ax.set_title("PACF of Training Monthly Net")
        return self._save_figure(fig, output_path)
    
    def save_ar4_residual_histogram(self, output_path, test_size=6):
        """plot the histogram of the residuals"""
        residuals = self.analyzer.arima_residuals(p=4,d=0,q=0,test_size=test_size)
    
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(residuals)
    
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title("AR(4) residual distribution")
    
        return self._save_figure(fig, output_path)

    def save_ar4_qq_plot(self, output_path, test_size=6):
        """plot the QQ-plot of the residuals"""
        residuals = self.analyzer.arima_residuals(p=4,d=0,q=0,test_size=test_size)
    
        fig, ax = plt.subplots(figsize=(7, 6))
        probplot(residuals,dist="norm",plot=ax)
        ax.set_title("AR(4) Residual Q-Q plot")
        
        return self._save_figure(fig, output_path)

    def save_all(self, output_dir):
        """saves all plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self.save_category_expenses(output_dir / "category expenses.png"),
            self.save_monthly_summary(output_dir / "monthly summary.png"),
            self.save_acf(output_dir / "acf.png"),
            self.save_pacf(output_dir / "pacf.png"),
            self.save_ar4_predictions(output_dir / "ar4 predictions.png"),
            self.save_ar4_residual_histogram(output_dir / "ar4 residual histogram.png"),
            self.save_ar4_qq_plot(output_dir / "ar4 QQ-plot.png")
        ]