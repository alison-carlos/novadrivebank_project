import pandas as pd
import yaml
import psycopg2
from fuzzywuzzy import process
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
#import const


def fetch_data_from_db(sql_query):
    try:
        with open('config.yml', 'r') as open_file:
            config = yaml.safe_load(open_file)

        con = psycopg2.connect(
            dbname=config['database_config']['dbname'],
            user=config['database_config']['user'],
            password=config['database_config']['password'],
            host=config['database_config']['host']
        )

        cursor = con.cursor()

        cursor.execute(sql_query)

        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'con' in locals():
            con.close()

        return df


def fn_replace_nulls(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            mode = df[column].mode()[0]
            df.fillna({column : mode}, inplace=True)
        else:
            median = df[column].median()
            df.fillna({column : median}, inplace=True)


def fn_fix_typing_errors(df, column, valid_list):
    for i, value in enumerate(df[column]):
        value_as_str = str(value) if pd.notnull(value) else value

        if value_as_str not in valid_list and pd.notnull(value_as_str):
            corrected_value = process.extractOne(value_as_str, valid_list)[0]
            df.at[i, column] = corrected_value


def fn_treat_outliers(df, column, min, max):
    median = df[(df[column] >= min) & (df[column] <= max)][column].median()
    df[column] = df[column].apply(lambda x: median if x < min or x > max else x)
    return df


def fn_save_scalers(df, column_names):
    for column_name in column_names:
        scaler = StandardScaler()
        df[column_name] = scaler.fit_transform(df[[column_name]])
        joblib.dump(scaler, f'./objects/scaler_{column_name}.joblib')

    return df


def fn_save_encoders(df, column_names):
    for column_name in column_names:
        label_encoder = LabelEncoder()
        df[column_name] = label_encoder.fit_transform(df[column_name])
        joblib.dump(label_encoder, f'./objects/label_encoder_{column_name}.joblib')
    return df