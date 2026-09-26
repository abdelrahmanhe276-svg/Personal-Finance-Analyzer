
import pandas as pd

req_columns = {
    "Date",
    "Transaction Description",
    "Category",
    "Amount",
    "Type",
}


def validate_transactions(data):
    """validate the dataset by checking required columns and transaction type"""
    missing_columns = req_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns: {missing}")
    if data.empty:
        raise ValueError("The dataset is empty")
    valid_types = {"Income", "Expense"} # only allowed column types
    observed_types = set(data["Type"].dropna().unique())
    invalid_types = observed_types.difference(valid_types) # it gets the elements in
    # the observed types tha are not in valid types 
    if invalid_types:
        invalid = ", ".join(sorted(invalid_types))
        raise ValueError(f"Invalid types: {invalid}")
    return True


def load_transactions(path):
    """load the transactions and validate the types of the columns and sort data by date"""
    data = pd.read_csv(path)
    # convert columns to  approperiate types
    data["Date"] = pd.to_datetime(data["Date"]) 
    data["Amount"] = pd.to_numeric(data["Amount"])
    text_columns = ["Transaction Description", "Category", "Type"]
    for column in text_columns:
        data[column] = data[column].astype(str).str.strip()
    validate_transactions(data)
    return data.sort_values("Date")
