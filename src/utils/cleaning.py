import numpy as np


def find_repeated_rows(df, max_repetitions=1 * 24):
    """
    Outputs the indices of rows where prediction values are constant over max_repetitions hours.
    """

    df = df.reset_index()
    repeated_indices = []
    repeated_indices_temp = []

    for index, row in df.iterrows():
        if index == 0:
            continue

        if row.y == df.iloc[index - 1].y:
            repeated_indices_temp.append(index)
        else:
            if len(repeated_indices_temp) <= max_repetitions:
                repeated_indices_temp = []
            else:
                repeated_indices += repeated_indices_temp
                repeated_indices_temp = []

    return repeated_indices


def find_repeated_indices(df, column_name, repeat_count=12):
    """
    Find and return the indexes of rows with a specified number of repeated values in a given column.

    Parameters:
    - df: DataFrame to search for repeated rows.
    - column_name: Name of the column to check for repeated values.
    - repeat_count: Number of repeated values required to consider a row as a match.

    Returns:
    - List of indexes for rows with the specified number of repeated values in the given column.
    """
    df = df.reset_index()
    repeated_indexes = []
    temp_repeated_indexes = []
    current_value = None
    count = 0

    for index, row in df.iterrows():
        value = row[column_name]

        if value == current_value:
            count += 1
            temp_repeated_indexes.append(index)
        else:
            current_value = value
            if count <= repeat_count:
                temp_repeated_indexes = []
                count = 1
            else:
                for i in temp_repeated_indexes:
                    if i not in repeated_indexes:
                        repeated_indexes.append(i)
                temp_repeated_indexes = []
                count = 1

    return repeated_indexes


def remove_duplicates_in_coloumn(df, col):
    """
    Removes duplicate rows in a dataframe based on a column
    """
    duplicate_mask = df[col].duplicated(keep="first")
    if duplicate_mask.any():
        df = df[~duplicate_mask]
    return df


def convert_from_degree_to_ciruclar(df, feature):
    """
    Converts a feature from degree to circular
    """
    df[feature + "_sin"] = np.sin(np.radians(df[feature]))
    df[feature + "_cos"] = np.cos(np.radians(df[feature]))
    df = df.drop(feature, axis=1)
    return df


def resample_hourly(df, func="sum"):
    df.set_index("date_forecast", inplace=True)

    if func == "sum":
        df = df.resample("1H").sum()
    elif func == "mean":
        df = df.resample("1H").mean()
    elif func == "median":
        df = df.resample("1H").median()

    df.reset_index(inplace=True)
    return df


def create_time_features(df, col):
    df["hour"] = df[col].dt.hour
    df["dayofmonth"] = df[col].dt.day
    df["dayofweek"] = df[col].dt.dayofweek
    df["quarter"] = df[col].dt.quarter
    df["month"] = df[col].dt.month
    df["year"] = df[col].dt.year
    df["dayofyear"] = df[col].dt.dayofyear

    return df
