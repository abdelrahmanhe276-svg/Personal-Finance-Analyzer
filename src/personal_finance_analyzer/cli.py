import argparse
from pathlib import Path

from .analyzer import FinanceAnalyzer
from .data import load_transactions
from .visualization import FinanceVisualizer


def build_parser():
    """set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="analyze personal finance transactions from a CSV file"
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default="data/Personal_Finance_Dataset.csv",
        help="Path to the transaction CSV file",
    )

    return parser


def main():
    """Run the financial analysis"""
    parser = build_parser()
    options = parser.parse_args()

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    transactions = load_transactions(options.data_path)

    analyzer = FinanceAnalyzer(transactions)
    analyzer.monthly_summary().to_csv(
        output_dir / "monthly sumary.csv",index=False)
    analyzer.expense_by_category().to_csv(
        output_dir / "expense by category.csv",index=False)

    analyzer.income_by_category().to_csv(
        output_dir / "income by category.csv",index=False)
    FinanceVisualizer(analyzer).save_all(output_dir)
    print(f"generated files in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()