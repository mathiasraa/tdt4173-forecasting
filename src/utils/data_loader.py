import pandas as pd
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


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
