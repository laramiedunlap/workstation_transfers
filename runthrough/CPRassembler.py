import pandas as pd
import numpy as np
from pooler import pool
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def subset_dataframe(df, conditions):
    """return a subset of a dataframe based on multiple columns and conditions"""
    mask = pd.Series(True, index=df.index)
    for col, cond in conditions.items():
        mask &= df[col].isin(cond)
    return df[mask]

 
def create_pooler(in_df:pd.DataFrame)-> pool.Pooler:
    """Create static pools of Loans from the DataFrame"""
    temp = in_df.set_index('GP')
    temp = temp.to_dict()
    loans_dict = {}
    for gp in temp['NoteDt'].keys():
        loans_dict[str(gp)] = pool.Loan(gp, pd.to_datetime(temp['NoteDt'][gp]))
        loans_dict[str(gp)].maturity_dt = temp['MaturityDt'][gp]
        loans_dict[str(gp)].maturity_mths_qty = temp['MaturityMthsQty'][gp]
        loans_dict[str(gp)].default_dt = temp['DefaultDt'][gp]
        loans_dict[str(gp)].default_mths_qty = temp['DefaultMthsQty'][gp]
        loans_dict[str(gp)].prepay_dt = temp['PrepayDt'][gp]
        loans_dict[str(gp)].prepay_mths_qty = temp['PrepayMthsQty'][gp]

    user_pool = pool.Pooler(loans_dict)
    user_pool.build_triangles_counts()
    pool_dict = {}
    for k, v in user_pool.triangles.items():
        pool_dict[k] = dict(outstanding=v[0], prepayments=v[1], defaults=v[2])

    return pool_dict

def create_pool_df(pool_dict:dict)-> pd.DataFrame:
    """Create formatted arrays inside a dataframe from the dictionary containing the data on static pools"""
    df_pool = pd.DataFrame.from_dict( pool_dict, orient='index')
    df_pool.index = [float(e) for e in df_pool.index.to_list()]
    df_pool = df_pool.sort_index()
    # Format array lengths
    max_row_length = df_pool.shape[0]
    count = -1
    for i, row in df_pool.iterrows():
        count+=1
        for col in df_pool.columns:
            # ----------------------------------------------------------------------------------------------
            #Method 1: This line will simply truncate the array to the right length:
            # ----------------------------------------------------------------------------------------------
            # df_pool.at[i,col] = row[col][:(max_row_length-count)]
            # ----------------------------------------------------------------------------------------------
            #Method 2: Pad arrays with NaNs --> this will truncate the array then fill it back in with NaNs
            # ----------------------------------------------------------------------------------------------
            arr = row[col][:(max_row_length-count)].astype(float)
            padded_arr = np.pad(arr, (0, max_row_length - (max_row_length-count) ), mode='constant', constant_values=np.nan)
            df_pool.at[i,col] = padded_arr
    return df_pool


def annualize_pool(df_pool)->pd.DataFrame:
    """Annualize the dataframe into vintages. Remove vintages without 12 months of data"""
    annual_pool = df_pool.copy(deep=True)
    annual_pool['Year'] = annual_pool.index.astype(int)
    # Filter out years without 12 months of history
    vals = annual_pool['Year'].value_counts().to_dict()
    yr_range = []
    for k, v in vals.items():
        if v == 12:
            yr_range.append(k)

    annual_pool = annual_pool[annual_pool['Year'].isin(yr_range)]
    # Switch to year group
    year_grouped = annual_pool.groupby('Year')
    year_grouped = year_grouped.agg(np.nansum)
    year_grouped['smm'] = (year_grouped['prepayments']+year_grouped['defaults'])/year_grouped['outstanding']
    year_grouped['cpr'] = (1-(1-year_grouped['smm'])**12)

    return year_grouped

def outstanding_annual_rundown(in_arr):
    return in_arr[::11]

def aggregate_annual_sums(in_arr):
    remainder = len(in_arr) % 12
    if remainder != 0:
        padding = np.zeros(12 - remainder)
        subsections = np.concatenate([in_arr, padding])
    else:
        subsections = in_arr
    subsections = np.split(subsections, len(subsections) // 12)
    return (np.sum(subsections, axis=1))

def aggregate_annual_averages(in_arr):
    n_years = len(in_arr) // 12
    arr_2d = in_arr[:n_years*12].reshape(n_years,12)
    return np.nanmean(arr_2d, axis=1)

def aggregate_annual_median(in_arr):
    n_years = len(in_arr) // 12
    arr_2d = in_arr[:n_years*12].reshape(n_years,12)
    return np.nanmedian(arr_2d, axis=1)

def generate_totals_data(year_grouped:pd.DataFrame, dir:Path)->pd.DataFrame:
    prepays = year_grouped[['prepayments', 'defaults']].applymap(aggregate_annual_sums).to_dict()
    totals = year_grouped[['outstanding']].applymap(outstanding_annual_rundown).to_dict()
    totals = totals['outstanding']
    totals = (pd.DataFrame.from_dict(totals, orient='index'))
    triangles = [totals]
    triangles += [pd.DataFrame.from_dict(prepays[k], orient='index') for k in prepays.keys()]
    triangles_df = pd.concat(triangles, axis=0)
    triangles_df.to_csv(os.path.join(dir,'totals.csv'))
    return triangles_df

def generate_cpr_heat(year_grouped:pd.DataFrame, dir:Path)->pd.DataFrame:
    arr = year_grouped[['cpr']]
    cpr_heat = arr.applymap(aggregate_annual_median).to_dict()
    cpr_heat = cpr_heat['cpr']
    cpr_heat = pd.DataFrame.from_dict(cpr_heat, orient='index')
    cpr_heat.to_csv(os.path.join(dir,'cpr_heat.csv'))
    return cpr_heat

def generate_min_max_mid(cpr_heat:pd.DataFrame, dir:Path)->pd.DataFrame:
    min_max_mid_df = pd.DataFrame.from_dict({'max': cpr_heat.max(axis=0), 'median': cpr_heat.median(axis=0), 'avg': cpr_heat.mean(axis=0), 'min': cpr_heat.min(axis=0)}).transpose()
    min_max_mid_df.to_csv(os.path.join(dir,'min_max_mids.csv'))
    return min_max_mid_df
    
def generate_lifetime(cpr_heat:pd.DataFrame, dir:Path)->pd.DataFrame:
    # get the cumulative sum of each row WHILE ignoring NaN values (otherwise the denominator is off)
    cumulative_sum = np.nancumsum(cpr_heat.values, axis=1)
    # compute the number of non-NaN values in each row
    num_non_nan = (~np.isnan(cpr_heat.values)).cumsum(axis=1)
    # get ROW-WISE average up until the first NaN value is encountered
    row_avg = np.where(np.isnan(cpr_heat), np.nan, cumulative_sum / num_non_nan)
    # create new dataframe with row-wise averages
    lifetime_df = pd.DataFrame(row_avg, columns=cpr_heat.columns, index=cpr_heat.index)
    lifetime_df.to_csv(os.path.join(dir,'lifetime.csv'))
    return lifetime_df

def create_data_directory(directory:str)->Path:
    if os.path.exists(directory):
        # if it does exist, remove the files from it
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(e)
    else:
        # if it doesn't exist, then create it
        os.makedirs(directory)
    return(Path(directory))

def run_baseline_assembly(maturity_slice:pd.DataFrame, baseline_directory:Path)->None:
    print('running baseline assembly...')
    pool_dict = create_pooler(maturity_slice)
    pool_df = create_pool_df(pool_dict)
     # From year grouped, you can get all the statistics you need for each bucket
    year_grouped = annualize_pool(pool_df)
    totals = generate_totals_data(year_grouped=year_grouped, dir=baseline_directory)
    cpr_heat = generate_cpr_heat(year_grouped=year_grouped, dir=baseline_directory)
    min_max_mid_df = generate_min_max_mid(cpr_heat=cpr_heat, dir=baseline_directory)
    lifetime_df = generate_lifetime(cpr_heat=cpr_heat, dir=baseline_directory)
    return None

def run_notional_assembly(maturity_slice:pd.DataFrame, notional_directory:Path):
    print('running notional assembly')
    pass

def main():
    """The script inside main needs to get split out into other functions"""
    print('booting up...\n')
    loan_data = pd.read_csv("raw_data/master_loan_tape.csv")
    # format date columns to datetime data types
    date_cols = [c for c in loan_data.columns if str(c)[-2:]=='Dt']
    for col in date_cols:
        loan_data[col] = pd.to_datetime(loan_data[col])
    # Get user Maturity option
    maturity_choices = loan_data['MatBucket'].value_counts().index.to_list()
    print("Select a Maturity:\n")
    for idx, m in enumerate(maturity_choices):
        print(idx, m)
    maturity_selection = input('Type either the number or name of the maturity bucket\n')
    if maturity_selection.find('-')>0 or maturity_selection.find('+')>0:
        # This means the user input typed in the maturity bucket
        pass
    else:
        # This means the user typed the index
        maturity_selection = maturity_choices[int(maturity_selection)]
    
    # Slice to the correct maturity loans for the user's selection
    maturity_condition = {'MatBucket':[maturity_selection]}
    maturity_slice = subset_dataframe(loan_data, maturity_condition)
    # Create data directory
    DATA_DIRECTORY = create_data_directory('CPR_assembly_outputs/')
    MAT_DIRECTORY = create_data_directory(os.path.join(DATA_DIRECTORY,f'{maturity_selection}/'))
    # After Maturity Selection, figure out subset:
    subset_choice = input("What should we assemble?:\n 1. Baseline\n 2. Margin Buckets\n 3. Notional Buckets\n")


    if subset_choice == '1' or subset_choice.lower() == 'baseline':
        BASELINE_DIRECTORY = create_data_directory(os.path.join(MAT_DIRECTORY,'baseline/'))
        run_baseline_assembly(maturity_slice=maturity_slice, baseline_directory=BASELINE_DIRECTORY)
        return print('completed')
    
    elif subset_choice == '3' or subset_choice.lower() == 'baseline':
        NOTIONAL_DIRECTORY = create_data_directory(os.path.join(MAT_DIRECTORY,'notional/'))
        custom_choice = input("Would you like custom buckets? (y/n) ")
        if custom_choice.lower()=='y':
            bins = input("enter bin edges seperated by a comma\n")
            bins = [float(e) for e in bins.split(',')]
            bins.append(np.inf)
            maturity_slice['CustomNotionalBuckets'] = pd.cut(maturity_slice['LoanAmt'], bins=bins)
            notional_buckets = maturity_slice['CustomNotionalBuckets'].value_counts().index.to_list()
            notional_reference = 'CustomNotionalBuckets'
        for b in notional_buckets:
            print(f"running assembly for {b}...")
            bucket_directory = create_data_directory(os.path.join(NOTIONAL_DIRECTORY,f'{b}/'))
            bucket_condition = {notional_reference:[b]}
            bucket_slice = subset_dataframe(maturity_slice, bucket_condition)
            pool_dict = create_pooler(bucket_slice)
            pool_df = create_pool_df(pool_dict)
            # From year grouped, you can get all the statistics you need for each bucket
            try:
                year_grouped = annualize_pool(pool_df)
            except KeyError:
                print('KeyError encountered')
                print(pool_df)
                return None
            totals = generate_totals_data(year_grouped=year_grouped, dir=bucket_directory)
            cpr_heat = generate_cpr_heat(year_grouped=year_grouped, dir=bucket_directory)
            min_max_mid_df = generate_min_max_mid(cpr_heat=cpr_heat, dir=bucket_directory)
            lifetime_df = generate_lifetime(cpr_heat=cpr_heat, dir=bucket_directory)  
    else:
    # get all possible margin buckets
    # Ask user for custom margin buckets
        custom_choice = input("Would you like custom buckets? (y/n) ")
        if custom_choice.lower()=='y':
            bins = input("enter bin edges seperated by a comma\n")
            bins = [float(e) for e in bins.split(',')]
            bins.append(np.inf)
            maturity_slice['CustomMarginBuckets'] = pd.cut(maturity_slice['Margin'], bins=bins)
            margin_buckets = maturity_slice['CustomMarginBuckets'].value_counts().index.to_list()
            margin_reference = 'CustomMarginBuckets'
        else:
            margin_buckets = maturity_slice['MarginBucket'].value_counts().index.to_list()
            margin_reference = 'MarginBucket'
    
        for b in margin_buckets:
            print(f"running assembly for {b}...")
            bucket_directory = create_data_directory(os.path.join(MAT_DIRECTORY,f'{b}/'))
            bucket_condition = {margin_reference:[b]}
            bucket_slice = subset_dataframe(maturity_slice, bucket_condition)
            pool_dict = create_pooler(bucket_slice)
            pool_df = create_pool_df(pool_dict)
            # From year grouped, you can get all the statistics you need for each bucket
            year_grouped = annualize_pool(pool_df)
            totals = generate_totals_data(year_grouped=year_grouped, dir=bucket_directory)
            cpr_heat = generate_cpr_heat(year_grouped=year_grouped, dir=bucket_directory)
            min_max_mid_df = generate_min_max_mid(cpr_heat=cpr_heat, dir=bucket_directory)
            lifetime_df = generate_lifetime(cpr_heat=cpr_heat, dir=bucket_directory)        
    
    



    

    


if __name__ == "__main__":
    main()
    


    

