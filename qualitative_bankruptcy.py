#for qualitative bankruptcy
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

classification_file = "raw_datasets/Qualitative_Bankruptcy.data.txt"
regression_file = "raw_datasets/ENB2012_data.xlsx"

clean(classification_file, "clean_data/Qualitative_Bankruptcy", 0.8)
clean(regression_file, "clean_data/ENB2012_data", 0.8)
