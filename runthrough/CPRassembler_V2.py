import pandas as pd
import numpy as np
from pooler import pool
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def subset_dataframe(df, conditions, inverse=False):
    """return a subset of a dataframe based on multiple columns and conditions"""
    if not inverse:
        mask = pd.Series(True, index=df.index)
        for col, cond in conditions.items():
            mask &= df[col].isin(cond)
    else:
        mask = pd.Series(True, index=df.index)
        for col, cond in conditions.items():
            mask &= ~df[col].isin(cond)
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
            arr = row[col][:(max_row_length-count)].astype(float)
            padded_arr = np.pad(arr, (0, max_row_length - (max_row_length-count) ), mode='constant', constant_values=np.nan)
            df_pool.at[i,col] = padded_arr
    return df_pool

def outstanding_annual_rundown(in_arr):
    return in_arr[::11]

def aggregate_annual_sums(in_arr):
    remainder = len(in_arr) % 11
    if remainder != 0:
        padding = np.zeros(11 - remainder)
        subsections = np.concatenate([in_arr, padding])
    else:
        subsections = in_arr
    subsections = np.split(subsections, len(subsections) // 11)
    return (np.nansum(subsections, axis=1))

def aggregate_annual_averages(in_arr):
    n_years = len(in_arr) // 11
    arr_2d = in_arr[:n_years*11].reshape(n_years,11)
    return np.nanmean(arr_2d, axis=1)

def aggregate_annual_median(in_arr):
    n_years = len(in_arr) // 11
    arr_2d = in_arr[:n_years*11].reshape(n_years,11)
    return np.nanmedian(arr_2d, axis=1)

def annualize_pool(df_pool:pd.DataFrame)->pd.DataFrame:
    """Annualize the dataframe into vintages and years from origination."""
    annual_pool = df_pool.copy(deep=True)
    annual_pool['Year'] = annual_pool.index.astype(int)
    vals = annual_pool['Year'].value_counts().to_dict()
    yr_range = []
    for k, v in vals.items():
        if v == 12:
            yr_range.append(k)
    annual_pool = annual_pool[annual_pool['Year'].isin(yr_range)]
    year_grouped = annual_pool.groupby('Year')
    year_grouped = year_grouped.agg(np.nansum)
    year_grouped['outstanding'] = year_grouped['outstanding'].apply(outstanding_annual_rundown)
    year_grouped[['prepayments','defaults']] = year_grouped[['prepayments','defaults']].applymap(aggregate_annual_sums)
    year_grouped['cpr'] = (year_grouped['prepayments']+year_grouped['defaults'])/year_grouped['outstanding']
    return year_grouped

def enforce_shape(in_df:pd.DataFrame)->pd.DataFrame:
    data = in_df.copy(deep=True)
    last_year = data.index.max()+1
    for i in range(len(data)):
        max_col = last_year - data.index[i]
        data.iloc[i,(max_col):] = np.NaN
    return data

def generate_totals_data(df_pool:pd.DataFrame, dir:Path)->pd.DataFrame:
    """This method is just getting the raw data for totals into it's own csv file"""
    annual_pool = df_pool.copy(deep=True)
    annual_pool['Year'] = annual_pool.index.astype(int)
    vals = annual_pool['Year'].value_counts().to_dict()
    yr_range = []
    for k, v in vals.items():
        if v == 12:
            yr_range.append(k)
    annual_pool = annual_pool[annual_pool['Year'].isin(yr_range)]
    year_grouped = annual_pool.groupby('Year')
    year_grouped = year_grouped.agg(np.nansum)
    totals = year_grouped[['outstanding']].applymap(outstanding_annual_rundown).to_dict()
    prepays = year_grouped[['prepayments']].applymap(aggregate_annual_sums).to_dict()
    defaults = year_grouped[['defaults']].applymap(aggregate_annual_sums).to_dict()
    totals = totals['outstanding']
    prepays = prepays['prepayments']
    defaults = defaults['defaults']
    prepays = (pd.DataFrame.from_dict(prepays, orient='index'))
    totals = (pd.DataFrame.from_dict(totals, orient='index'))
    defaults = (pd.DataFrame.from_dict(defaults, orient='index'))
    totals = enforce_shape(totals)
    prepays = enforce_shape(prepays)
    defaults = enforce_shape(defaults)
    triangles = [totals, prepays, defaults]
    triangles_df = pd.concat(triangles, axis=0)
    triangles_df.to_csv(os.path.join(dir,'totals.csv'))
    return triangles_df

def generate_cpr_heat(year_grouped:pd.DataFrame, dir:Path)->pd.DataFrame:
    cpr_heat = pd.DataFrame.from_dict(year_grouped['cpr'].to_dict(), orient='index')
    cpr_heat = enforce_shape(cpr_heat)
    cpr_heat.to_csv(os.path.join(dir,'cpr_heat.csv'))
    return cpr_heat

def generate_min_max_mid(cpr_heat:pd.DataFrame, dir:Path)->pd.DataFrame:
    min_max_mid_df = pd.DataFrame.from_dict({'max': cpr_heat.max(axis=0,skipna=True), 'median': cpr_heat.median(axis=0,skipna=True), 'avg': cpr_heat.mean(axis=0,skipna=True), 'min': cpr_heat.min(axis=0,skipna=True)}).transpose()
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
    """Assemble CPRs for the baseline maturity bucket"""
    print('running baseline assembly...')
    pool_dict = create_pooler(maturity_slice)
    pool_df = create_pool_df(pool_dict)
     # From year grouped, you can get all the statistics you need for each bucket
    year_grouped = annualize_pool(pool_df)
    totals = generate_totals_data(df_pool=pool_df, dir=baseline_directory)
    cpr_heat = generate_cpr_heat(year_grouped=year_grouped, dir=baseline_directory)
    min_max_mid_df = generate_min_max_mid(cpr_heat=cpr_heat, dir=baseline_directory)
    lifetime_df = generate_lifetime(cpr_heat=cpr_heat, dir=baseline_directory)
    return None


def run_bucket_assembly(maturity_slice:pd.DataFrame, buckets:list, col_ref:str, parent_dir:Path, args=0)->str:
    """Assemble CPRs for the all notional buckets passed to the function"""
    for b in buckets:
        bucket_directory = create_data_directory(os.path.join(parent_dir,f'{b}/'))
        if args == 'naics':
            if b== 722110:
                bucket_condition = {col_ref:[722110,722511]}
            elif b== 722513:
                bucket_condition = {col_ref:[722513,722211]}
            else:
                bucket_condition = {col_ref:[b]}
        else:
            bucket_condition = {col_ref:[b]}
        bucket_slice = subset_dataframe(maturity_slice, bucket_condition)
        print(f'running assembly {col_ref} -- {bucket_condition}...')
        pool_dict = create_pooler(bucket_slice)
        pool_df = create_pool_df(pool_dict)
        year_grouped = annualize_pool(pool_df)
        totals = generate_totals_data(df_pool=pool_df, dir=bucket_directory)
        # This is where I need to tie out the counts for each year to the counts from the baseline. 
        # So the first column [0] from each totals data frame needs to get stripped out.
        # Alternatively, we could write code that receives directions about what directories to tie out, then perform the tie out after this. The data is all parked into csvs.
        cpr_heat = generate_cpr_heat(year_grouped=year_grouped, dir=bucket_directory)
        min_max_mid_df = generate_min_max_mid(cpr_heat=cpr_heat, dir=bucket_directory)
        lifetime_df = generate_lifetime(cpr_heat=cpr_heat, dir=bucket_directory)
    return ('complete')

def is_binnable(in_df, col_name):
    col_dtype = in_df[col_name].dtype
    return col_dtype == 'float64' or col_dtype == 'int64'

def main():
    """The script inside main needs to get split out into other functions"""
    print('booting up...\n')
    loan_data = pd.read_csv("raw_data/loans_v2.csv")
    # loan_data = pd.read_csv("raw_data/master_loan_tape.csv")
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
    if maturity_selection != '21+':
        maturity_slice.loc[maturity_slice['PP_qty']==maturity_slice['MaturityMthsQty'],'PrepayMthsQty']= np.nan
        maturity_slice.loc[maturity_slice['PP_qty']==maturity_slice['MaturityMthsQty'],'DefaultMthsQty']= np.nan
    # Create data directory
    DATA_DIRECTORY = create_data_directory('CPR_assembly_outputs/')
    MAT_DIRECTORY = create_data_directory(os.path.join(DATA_DIRECTORY,f'{maturity_selection}/'))
    # After Maturity Selection, figure out subset:
    
    subset_choice = input("What should we assemble?:\n 1. Baseline\n 2. Margin Buckets\n 3. Notional Buckets\n 4. NAICS\n 5. State\n 6. Combo\n")
    if subset_choice == '1' or subset_choice.lower() == 'baseline':
        BASELINE_DIRECTORY = create_data_directory(os.path.join(MAT_DIRECTORY,'baseline/'))
        run_baseline_assembly(maturity_slice=maturity_slice, baseline_directory=BASELINE_DIRECTORY)
        return print('completed')
    
    elif subset_choice=='2' or subset_choice.lower() == 'margin buckets':
    # get all possible margin buckets
    # Ask user for custom margin buckets
        MARGIN_DIRECTORY = create_data_directory(os.path.join(MAT_DIRECTORY,'margin/'))
        custom_choice = input("Would you like custom buckets or 25bps? \n(Initial bucket 0-1%, 1-1.25%, etc.) (y/n) ")
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
        status = run_bucket_assembly(maturity_slice=maturity_slice, buckets=margin_buckets, col_ref=margin_reference, parent_dir=MARGIN_DIRECTORY)
        print(status)
        return None

    elif subset_choice == '3' or subset_choice.lower() == 'notional buckets':
        NOTIONAL_DIRECTORY = create_data_directory(os.path.join(MAT_DIRECTORY,'notional/'))
        custom_choice = input("Currently there are no preset notional buckets, will you create custom buckets? (y/n) ")
        if custom_choice.lower()=='y':
            bins = input("Enter bin edges seperated by a comma\n")
            print('Example: 0,2500000 --> (0, 2,500,000], (2,500,000, inf]')
            bins = [float(e) for e in bins.split(',')]
            bins.append(np.inf)
            maturity_slice['CustomNotionalBuckets'] = pd.cut(maturity_slice['LoanAmt'], bins=bins)
            notional_buckets = maturity_slice['CustomNotionalBuckets'].value_counts().index.to_list()
            notional_reference = 'CustomNotionalBuckets'
            status = run_bucket_assembly(maturity_slice=maturity_slice, buckets=notional_buckets,col_ref=notional_reference,parent_dir=NOTIONAL_DIRECTORY)
            print(status)
            return None
    
    elif subset_choice == '4' or subset_choice.lower() == 'NACIS':
        NAICS_DIRECTORY = create_data_directory(os.path.join(MAT_DIRECTORY, 'NAICS/'))
        bins = input("Enter NAICS codes seperated by a comma\n")
        bins = [int(e) for e in bins.split(',')]
        naics_reference = 'Code'
        status = run_bucket_assembly(maturity_slice=maturity_slice, buckets=bins, col_ref=naics_reference,parent_dir=NAICS_DIRECTORY,args='naics')
        print(status)
        return None
    
    elif subset_choice == '6' or subset_choice.lower() == 'combo':
        COMBO_DIRECTORY = create_data_directory(os.path.join(MAT_DIRECTORY,'Combinations/'))
        options = [x for x in enumerate(maturity_slice.columns.to_list())]
        options = {key: value for key, value in options}
        combo_slice = maturity_slice.copy(deep=True)
        user_initiate = 0
        while user_initiate == 0:    
            print('Enter the column number reference for the column you would like to filter on:')
            choice = int(input(f"{options}\n"))
            col = options[choice]
            if col == 'Code':
                bins = input("Enter NAICS codes seperated by a comma\n")
                bins = [int(e) for e in bins.split(',')]
                inverse_map = {'y': False, 'n': True}
                inverse_opt = input('Inclusive or exclusive? (y= include, n= exclude) ')
                inverse_opt = inverse_map[inverse_opt]
                combo_condition = {col:[bins]}
                combo_slice = subset_dataframe(combo_slice, combo_condition, inverse=inverse_opt)
                user_initiate = int(input('run? 1= yes, 0= no\n'))

            elif is_binnable(combo_slice,col):
                bins = input("Enter bin edges seperated by a comma\n")
                bins = [float(e) for e in bins.split(',')]
                bins.append(np.inf)
                combo_slice[f'CustomBuckets_{choice}'] = pd.cut(combo_slice[col], bins=bins)
                combo_buckets= combo_slice[f'CustomBuckets_{choice}'].value_counts().index.to_list()
                user_initiate = int(input('run? 1= yes, 0= no\n'))
    try:
        status = run_bucket_assembly(combo_slice,combo_buckets,f'CustomBuckets_{choice}',COMBO_DIRECTORY)
    except KeyError:
        print(combo_slice.head(10))
        print('\n' )
        print(bins)

    
if __name__ == "__main__":
    main()
    


    

