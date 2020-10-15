import pandas as pd
import numpy as np


def filterCols(df: pd.DataFrame, imf: str, interpol: str):
    name_bool = (df['IMF Spot'] == imf).to_numpy()
    inter_bool = (df['Interpolation Method'] == interpol).to_numpy()
    return df.loc[name_bool & inter_bool], name_bool, inter_bool

def impute(df: pd.DataFrame, col: str):
    mean = df[col].mean()
    df[col] = df[col].fillna(value=mean)
    return df


def Main():
    df = pd.read_csv('interpolations.csv')
    cols_to_impute = df.columns[8:].to_numpy(dtype=str)

    for i in range(1, 13):
        name = 'IMF ' + str(i)
        relevent_cols = df.loc[df.isnull().any(axis=1)]
        interpolations = relevent_cols['Interpolation Method'].unique()

        for inter in interpolations:
            relevent_cols = None
            filtered, n, b = filterCols(df, name, inter)
            for col in cols_to_impute:
                df.loc[n & b] = impute(filtered, col)

    df.to_csv('imputed_interpolations.csv', index=False, index_label=False)



Main()
