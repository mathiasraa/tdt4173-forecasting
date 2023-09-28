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
