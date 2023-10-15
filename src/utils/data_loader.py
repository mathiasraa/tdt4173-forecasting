import pandas as pd
import os

from src.utils.cleaning import find_repeated_rows

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_data():
    data = {"A": {}, "B": {}, "C": {}}

    for location in ("A", "B", "C"):
        y = pd.read_parquet(f"{dir_path}/../../data/{location}/train_targets.parquet")
        X_test_estimated = pd.read_parquet(
            f"{dir_path}/../../data/{location}/X_test_estimated.parquet"
        )
        X_train_estimated = pd.read_parquet(
            f"{dir_path}/../../data/{location}/X_train_estimated.parquet"
        )
        X_train_observed = pd.read_parquet(
            f"{dir_path}/../../data/{location}/X_train_observed.parquet"
        )
        data[location]["y"] = y
        data[location]["X_test_estimated"] = X_test_estimated
        data[location]["X_train_estimated"] = X_train_estimated
        data[location]["X_train_observed"] = X_train_observed

    return data


def load_train_test(location="A"):
    """
    Load training and test data from location dataset

    Params
    ------
    location: A | B | C

    Returns
    -------
    (X_train, y_train, X_test, y_test): Training and testing datasets
    """

    train_targets = pd.read_parquet(
        f"{dir_path}/../../data/{location}/train_targets.parquet"
    )
    X_train_estimated = pd.read_parquet(
        f"{dir_path}/../../data/{location}/X_train_estimated.parquet"
    )
    X_train_observed = pd.read_parquet(
        f"{dir_path}/../../data/{location}/X_train_observed.parquet"
    )

    # Clean up feature-set for training data
    X_train = X_train_observed.copy()
    X_train = (
        X_train.rename(columns={"date_forecast": "time"})
        .set_index("time")
        .drop(columns=["date_calc"], errors="ignore")
    )
    X_train = X_train.resample("1H").mean().sort_index()
    X_train.index = pd.to_datetime(X_train.index)

    # Clean up y for training data
    y_train = train_targets.copy()
    y_train = y_train.rename(columns={"pv_measurement": "y"}).set_index("time")
    y_train.index = pd.to_datetime(y_train.index)

    # Merge training data
    train = pd.merge(X_train, y_train, left_index=True, right_index=True, how="left")
    train = train[train.y.notna()]

    # Clean up feature-set for test data
    X_test = X_train_estimated.copy()
    X_test = (
        X_test.rename(columns={"date_forecast": "time"})
        .set_index("time")
        .drop(columns=["date_calc"], errors="ignore")
    )
    X_test = X_test.resample("1H").mean().sort_index()
    X_test.index = pd.to_datetime(X_test.index)

    # Clean up y for test data
    y_test = train_targets.copy()
    y_test = y_test.rename(columns={"pv_measurement": "y"}).set_index("time")
    y_test.index = pd.to_datetime(y_test.index)

    test = pd.merge(X_test, y_test, left_index=True, right_index=True, how="left")
    test = test[test.y.notna()]
    test = test[test["total_cloud_cover:p"].notna()]

    features = X_test.columns

    return train[features], train[["y"]], test[features], test[["y"]]


def load_all_locations():
    X_result = pd.DataFrame()
    y_result = pd.DataFrame()

    for location in ("A", "B", "C"):
        X_train, y_train, X_test, y_test = load_train_test(location)

        X, y = merge_train_test(X_train, X_test), merge_train_test(y_train, y_test)

        X = X.reset_index()
        y = y.reset_index()

        repeated_indices = find_repeated_rows(y)

        X["location"] = location
        y["location"] = location
        X = X.drop(index=repeated_indices)
        y = y.drop(index=repeated_indices)

        X_result = pd.concat([X_result, X])
        y_result = pd.concat([y_result, y])

    return X_result, y_result


def load_all_vals():
    X_result = pd.DataFrame()

    for location in ("A", "B", "C"):
        X = load_val(location)

        X = X.reset_index()

        X["location"] = location

        X_result = pd.concat([X_result, X])

    return X_result


def merge_train_test(train, test):
    train["set_type"] = "TRAIN"
    test["set_type"] = "TEST"

    # Concatenate the two dataframes vertically
    combined_df = pd.concat([train, test])

    return combined_df


def load_val(location="A"):
    """
    Load validation data from location dataset

    Params
    ------
    location: A | B | C

    Returns
    -------
    X_val: Training and testing datasets
    """

    X_val_estimated = pd.read_parquet(
        f"{dir_path}/../../data/{location}/X_test_estimated.parquet"
    )

    # Clean up feature-set for training data
    X_val = X_val_estimated.copy()
    X_val = (
        X_val.rename(columns={"date_forecast": "time"})
        .set_index("time")
        .drop(columns=["date_calc"], errors="ignore")
    )
    X_val = X_val.resample("1H").mean().sort_index()
    X_val.index = pd.to_datetime(X_val.index)

    return X_val


def load_val_dates():
    """
    Output a list of validation dataset datetimes

    Returns
    -------
    array: dates
    """

    test = pd.read_csv(f"{dir_path}/../../data/test.csv")

    return test["time"].unique().tolist()
