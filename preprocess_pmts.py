import pandas as pd
import numpy as np

def to_datetime(in_df: pd.DataFrame, c_list: list)->pd.DataFrame:
    for c in c_list:
        in_df[c] = pd.to_datetime(in_df[c])
    return in_df

def to_numeric(in_df: pd.DataFrame, c_list: list)->pd.DataFrame:
    for c in c_list:
        in_df[c] = pd.to_numeric(in_df[c])
    return in_df

def to_str(in_df: pd.DataFrame, c_list: list)->pd.DataFrame:
    for c in c_list:
        in_df[c] = in_df[c].astype(str)
    return in_df

def stratify_df_types(test:pd.DataFrame)->pd.DataFrame:
    """Set dtypes and column names like they will be in the database"""
    test = test.reset_index().drop(columns='index')
    test.columns = ['ObservationNmb', 
                    'LoanFundedDt', 'MaturityDt', 'EffectiveDt',
                    'TransactionCd', 'GeneralLedgerCd',
                    'TransactionAmt', 'TransactionBalanceAmt']
    dates_list = ['LoanFundedDt', 'MaturityDt', 'EffectiveDt']
    nums_list = ['TransactionAmt', 'TransactionBalanceAmt']
    str_list = ['TransactionCd', 'GeneralLedgerCd', 'ObservationNmb']
    test = to_datetime(test, dates_list)
    test = to_numeric(test, nums_list)
    test = to_str(test, str_list)
    return test
    
def set_top_bot(test: pd.DataFrame)-> pd.DataFrame:
    """Re-window a dataframe to the last occurring max TransactionBalanceAmt 
    and the first occurring min TransactionBalanceAmt"""
    ix_top = test[test.TransactionBalanceAmt.values == test.TransactionBalanceAmt.values.max()].index
    ix_bot = test[test.TransactionBalanceAmt.values == test.TransactionBalanceAmt.values.min()].index
    return test.iloc[ix_top[-1]:ix_bot[0]+1]

def remove_end_codes(test: pd.DataFrame) -> pd.DataFrame:
    """Remove increasing balances that occur with an end code"""
    e_codes = set(['195','218','305','410','416'])
    idx = test[(test.TransactionCd.isin(e_codes))&(test.TransactionAmt>0)].index.to_list()
    return test.drop(idx,axis=0)

def find_spikes(test: pd.DataFrame):
    """Find remaining balance spikes"""
    sample = set_top_bot(test)
    sample = remove_end_codes(sample)
    sample = sample.reset_index().drop(columns='index')
    arr = sample['TransactionBalanceAmt'].values
    mins = (np.diff(np.sign(np.diff(arr))) > 0).nonzero()[0] + 1
    maxs = (np.diff(np.sign(np.diff(arr))) < 0).nonzero()[0] + 1
    return sample, mins, maxs


def get_yrs_till_mat(test:pd.DataFrame)->float:
    try:
        return round((test['LoanFundedDt'][0] - test['MaturityDt'][0])/np.timedelta64(1, 'Y'),0)
    except:
        return 0

def hammer(test:pd.DataFrame)-> pd.DataFrame:
    """Force monotonic decreasing on balance and transactions"""
    test['resolved_Balance'] =   np.minimum.accumulate(test['resolved_Balance'])
    test['resolved_Amt'] = test['resolved_Balance'].diff()
    return test

def scalpel(test: pd.DataFrame, idx_min: list, idx_max: list)->pd.DataFrame:
    """Fix simplest form of accounting error"""
    if len(idx_min) == 0:
        return test
    x_max = test.loc[test.index.isin(idx_max)]
    x_min = test.loc[test.index.isin(idx_min)]
    
    min_row = x_min['resolved_Balance'].idxmin()
    max_row = x_max['resolved_Balance'].idxmax()
    test.at[min_row, 'resolved_Balance'] = -1*(test.at[min_row, 'resolved_Amt'])
    test.at[max_row, 'resolved_Amt'] = -1*(test.at[min_row, 'resolved_Balance'] - test.at[max_row, 'resolved_Balance'])
    test.at[min_row, 'resolved_Amt'] = -1*(test.at[min_row-1, 'resolved_Balance'] - test.at[min_row, 'resolved_Balance'])
    test['resolved_Amt'] = test['resolved_Balance'].diff()
    return test


def peak_handler(test: pd.DataFrame, idx_min: list, idx_max: list, yrs_till_mat: float):
    test['resolved_Amt'] = test['TransactionAmt']
    test['resolved_Balance'] = test['TransactionBalanceAmt']
    
    if test['resolved_Balance'].is_monotonic_decreasing:
        return test, 'N'
    elif len(idx_max)==0:
        return test, 'N'
    elif len(idx_max) > 3 and yrs_till_mat<=10:
        return test, 'Y'
    elif len(idx_max) <= 3:
        test = scalpel(test, idx_min, idx_max)
        if test['resolved_Balance'].is_monotonic_decreasing:
            return test, 'N'
        else:
            return hammer(test), 'N'
      
def preprocess_cohorts(c_map: dict):
    """Reassign GPs based on when their principal payments begin"""
    pass
    

def verify_vintage(test:pd.DataFrame, c_map: dict, curr_k: str, curr_idx: int) -> None:
    """Verify that a loan's payment history starts in the vintage it was originally assigned to before preprocessing"""
    v = str(test['EffectiveDt'].dt.year[0])
    if v != curr_k:
        to_change = c_map[curr_k].pop(curr_idx)
    if v in [str(k) for k in c_map.keys()]:
        c_map[v].append(to_change)
    return None

def remove_revolver(test: pd.DataFrame, c_map: dict, curr_k: str, curr_idx: int)-> False:
    """Remove loans that have now been labeled revolvers post-preprocessing"""
    if test.name == 'Y':
        c_map[curr_k].pop(curr_idx)
        return True
    else:
        return False

def preprocess_history(test: pd.DataFrame):
        test = stratify_df_types(test)
        test, mins, maxs = find_spikes(test)
        yrs = get_yrs_till_mat(test)

        test, is_revolver = peak_handler(test, mins, maxs, yrs)
        test['Revolving'] = is_revolver
        
        return test


        