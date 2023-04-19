import pandas as pd
import numpy as np
from .pool import Pooler, Loan

def create_pooler(in_df:pd.DataFrame)-> Pooler:
    """Create a pooler object"""
    temp = in_df.set_index('GP')
    temp = temp.to_dict()
    loans_dict = {}
    for gp in temp['NoteDt'].keys():
        loans_dict[str(gp)] = Loan(gp, pd.to_datetime(temp['NoteDt'][gp]))
        loans_dict[str(gp)].maturity_dt = temp['MaturityDt'][gp]
        loans_dict[str(gp)].maturity_mths_qty = temp['MaturityMthsQty'][gp]
        loans_dict[str(gp)].default_dt = temp['DefaultDt'][gp]
        loans_dict[str(gp)].default_mths_qty = temp['DefaultMthsQty'][gp]
        loans_dict[str(gp)].prepay_dt = temp['PrepayDt'][gp]
        loans_dict[str(gp)].prepay_mths_qty = temp['PrepayMthsQty'][gp]

    return Pooler(loans_dict)


def create_pool_dict(in_df: pd.DataFrame)-> dict:
    """convert pooler object into a python dictionary"""
    pooler = create_pooler(in_df)
    pooler.build_triangles_counts()
    pool_dict = {}
    for k, v in pooler.triangles.items():
        pool_dict[k] = dict(outstanding=v[0], prepayments=v[1], defaults=v[2])

    return pool_dict


def format_arr_lengths(in_df: pd.DataFrame)-> pd.DataFrame:
    """format array lengths in the 3D dataframe to match the number of months"""
    df = in_df.copy(deep=True)
    max_row_length = df.shape[0]
    count = -1
    for i, row in df.iterrows():
        count +=1
        for col in df.columns:
            arr = row[col][:(max_row_length)].astype(float)
            padded_arr = np.pad(arr, (0, max_row_length - (max_row_length-count) ), mode='constant', constant_values=np.nan)
            df.at[i,col] = padded_arr
    return df


def convert_dict_3d(in_df: pd.DataFrame)-> pd.DataFrame:
    """convert the dictionary into a dataframe"""
    pool_dict = create_pool_dict(in_df)
    df_pool = pd.DataFrame.from_dict(pool_dict, orient='index')
    df_pool.index = [float(e) for e in df_pool.index.to_list()]
    df_triangles = format_arr_lengths(df_pool)
    return df_triangles


