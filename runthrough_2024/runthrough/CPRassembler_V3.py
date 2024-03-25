import pandas as pd
import numpy as np
from pooler import pool
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional, Union

class DataFrameOutput(BaseModel):
    """
    
    Pydantic model for the script's data outputs
    
    Re-define the data attr however you need to-- just make sure: 
   
     1) validator return typing matches the data attribute type definition
     2) the dataframe property return works (gotta use it in a couple spots. Although I could refactor if that causes a headache.)
    
    """

    data: List[Dict[Union[str,Any],Any]]
    description: Optional[str]=None # Not sure if this is helpful, descriptions of the data are added in the assembly functions.

    @validator('data',pre=True)
    def convert_df_to_data(cls, value):
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        return value
    
    @property
    def dataframe(self):
        return pd.DataFrame(self.data) # NOTE: Now if I want this data in JSON format, I just say that.dataframe.to_json()
    

    
# ----- NOTE: ----> Here begins a horde of helper functions that can all stay the same, skip to the next big dashed comment :) ----->

def subset_dataframe(df, conditions, inverse=False):
    """Return a subset of a dataframe based on multiple columns and conditions"""
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

def is_binnable(in_df, col_name):
    col_dtype = in_df[col_name].dtype
    return col_dtype == 'float64' or col_dtype == 'int64'



# ----- NOTE: ----> Here begins stuff that gets your data. Don't need to change anything, just notice the output type hints on the `generate_this` functions -----

def generate_totals_data(df_pool:pd.DataFrame, dir:Optional[Path]=None )->DataFrameOutput:
    """This method generates totals data and returns a DataFrameOutput. They're called triangles because that's what they look like."""
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
    
    if dir:
        triangles_df.to_csv(os.path.join(dir,'totals.csv'))

    return DataFrameOutput(data=triangles_df)



def generate_cpr_heat(year_grouped:pd.DataFrame, dir:Optional[Path]=None)->DataFrameOutput:
    cpr_heat = pd.DataFrame.from_dict(year_grouped['cpr'].to_dict(), orient='index')
    cpr_heat = enforce_shape(cpr_heat)
    
    if dir:
        cpr_heat.to_csv(os.path.join(dir,'cpr_heat.csv'))

    return DataFrameOutput(data=cpr_heat)


def generate_min_max_mid(cpr_heat:pd.DataFrame, dir:Optional[Path]=None)->DataFrameOutput:
    min_max_mid_df = pd.DataFrame.from_dict({'max': cpr_heat.max(axis=0,skipna=True), 'median': cpr_heat.median(axis=0,skipna=True), 'avg': cpr_heat.mean(axis=0,skipna=True), 'min': cpr_heat.min(axis=0,skipna=True)}).transpose()
    
    if dir:
        min_max_mid_df.to_csv(os.path.join(dir,'min_max_mids.csv'))

    return DataFrameOutput(data=min_max_mid_df)


def generate_lifetime(cpr_heat:pd.DataFrame, dir:Optional[Path]=None)->DataFrameOutput:
    # get the cumulative sum of each row WHILE ignoring NaN values (otherwise the denominator is off)
    cumulative_sum = np.nancumsum(cpr_heat.values, axis=1)
    # compute the number of non-NaN values in each row
    num_non_nan = (~np.isnan(cpr_heat.values)).cumsum(axis=1)
    # get ROW-WISE average up until the first NaN value is encountered
    row_avg = np.where(np.isnan(cpr_heat), np.nan, cumulative_sum / num_non_nan)
    # create new dataframe with row-wise averages
    lifetime_df = pd.DataFrame(row_avg, columns=cpr_heat.columns, index=cpr_heat.index)
    
    if dir:
        lifetime_df.to_csv(os.path.join(dir,'lifetime.csv'))
    
    return DataFrameOutput(data=lifetime_df)



# ----- NOTE: ----> 
# I could change the descriptions to be the exact chart titles if that would be helpful. Or I could make keys the chart titles, you tell me. ----->

def run_baseline_assembly(maturity_slice:pd.DataFrame, baseline_directory:Optional[Path]=None, maturity_descr:Optional[str]=None)-> Dict[str,DataFrameOutput]:
    """Assemble CPRs for the baseline maturity bucket"""
    print('running baseline assembly...')
    # Bunch of complicated functions
    pool_dict = create_pooler(maturity_slice)
    pool_df = create_pool_df(pool_dict)
    year_grouped = annualize_pool(pool_df)
    
    totals = generate_totals_data(df_pool=pool_df, dir=None)
    cpr_heat = generate_cpr_heat(year_grouped=year_grouped, dir=None)
   
    cpr_heat_df = cpr_heat.dataframe  # NOTE added this line, because the above object is now a DATAFRAMEOUTPUT type

    min_max_mid = generate_min_max_mid(cpr_heat=cpr_heat_df, dir=None)
    lifetime = generate_lifetime(cpr_heat=cpr_heat_df, dir=None)

    # You could change these however you want, it's just describing what the data is if that's useful.
    if maturity_descr:
        totals.description = f"{maturity_descr} baseline_totals"
        cpr_heat.description = f"{maturity_descr} cohort_CPRs"
        min_max_mid.description = f"{maturity_descr} min_max_mids"
        lifetime.description = f"{maturity_descr} lifetime"

    # Ideally, you get this package back, and you can just add var.data or var.description to get the data/description for that pointer
    return {'totals':totals, 'cpr_heat':cpr_heat, 'mmm':min_max_mid, 'lifetime':lifetime}


def run_bucket_assembly(maturity_slice:pd.DataFrame, buckets:list, col_ref:str, parent_dir:Optional[Path]=None, args:Optional[Any]=None, maturity_descr:Optional[str]=None)-> Dict[str, Dict[str, Dict[str, DataFrameOutput] ] ]:
    """Assemble CPRs for the buckets passed to the function"""
    bin_data_package = { maturity_descr:{} }
    for b in buckets:
        # bucket_directory = create_data_directory(os.path.join(parent_dir,f'{b}/')) NOTE: this is commented out because we aren't printing csv files.
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
        totals = generate_totals_data(df_pool=pool_df, dir=None)
        cpr_heat = generate_cpr_heat(year_grouped=year_grouped, dir=None)
        cpr_heat_df = cpr_heat.dataframe  # NOTE added this line, because the above object is now a DATAFRAMEOUTPUT type
        min_max_mid = generate_min_max_mid(cpr_heat=cpr_heat_df, dir=None)
        lifetime = generate_lifetime(cpr_heat=cpr_heat_df, dir=None)
        if maturity_descr:
            totals.description = f"{maturity_descr}_{b} totals"
            cpr_heat.description = f"{maturity_descr}_{b} CPR"
            min_max_mid.description = f"{maturity_descr}_{b} min_max_mids"
            lifetime.description = f"{maturity_descr}_{b} lifetime"

        #  example bin_data_package:

        #       {"21+"":
        #           "<=250_000":{
        #                   "totals":totals,
        #                   "cpr_heat":cpr_heat,
        #                   "mmm":min_max_mid,
        #                   "lifetime":lifetime
        #                       }
        # 
        #            "Next {b}":{
        #                   ...
        #                       }
        #            }

        bin_data_package[maturity_descr][b] = {'totals':totals, 'cpr_heat':cpr_heat, 'mmm':min_max_mid, 'lifetime':lifetime}

    return bin_data_package


# We could define functions to set the available options for each available filter 
def get_maturity_options(loan_data:pd.DataFrame)->List[str]:
    return loan_data['MatBucket'].value_counts().index.to_list()    

def get_margin_options(maturity_slice:pd.DataFrame):
    """Returns the basic Margin Buckets"""
    margin_buckets = maturity_slice['MarginBucket'].value_counts().index.to_list()
    return margin_buckets

def get_naics_options(maturity_slice:pd.DataFrame)->List[int]:
    """Returns the Top 5 Industries for a given Maturity"""
    top_5_ind = maturity_slice.groupby('Code').sum()[['LoanAmt']].sort_values('LoanAmt',ascending=False).head(5).index.to_list()
    # It would be nice to allow users to select industries themselves, but this is okay for now.
    return [int(e) for e in top_5_ind]

def get_notional_options(maturity_slice:pd.DataFrame)->List[str]:
    starting_bin_edges = {"0-8":10_000, "8-11":100_000, "11-16":150_000, "16-21":250_000, "21+":300_000}

    
def get_state_options(maturity_slice:str):
    """Returns top ten states based on Original Balance"""
    pass

# Then we'll just return them whenever they're asked for.

def get_terminal_maturity_input(maturity_choices:List[str])->str:
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
    return maturity_selection

def test_main():
    print('booting up...\n')
    loan_data = pd.read_csv("raw_data/loans_v2.csv")
    date_cols = [c for c in loan_data.columns if str(c)[-2:]=='Dt']
    for col in date_cols:
        loan_data[col] = pd.to_datetime(loan_data[col], format='mixed')
    maturity_choices = get_maturity_options(loan_data=loan_data)
    maturity_selection = get_terminal_maturity_input(maturity_choices=maturity_choices)

    # This is just passing some arguments to a function that will slice the dataframe to that maturity. Nothing crazy
    maturity_condition = {'MatBucket':[maturity_selection]}
    maturity_slice = subset_dataframe(loan_data, maturity_condition)
    # Because maturity ranges (i.e. 8-11 year loans) could have different maturities, we don't want to label something a prepayment if it went to term. 
    # Basically all 21+ loans are 25yr so it's fine.
    if maturity_selection != '21+':
        maturity_slice.loc[maturity_slice['PP_qty']==maturity_slice['MaturityMthsQty'],'PrepayMthsQty']= np.nan
        maturity_slice.loc[maturity_slice['PP_qty']==maturity_slice['MaturityMthsQty'],'DefaultMthsQty']= np.nan

    # This is going to run the baseline assembly
    test_return = run_baseline_assembly(maturity_slice=maturity_slice, maturity_descr=maturity_selection)

    look_at_it = input("what do you want to see? (enter number)\n (1) Description\n (2) Dataframe\n (3) JSON\n (4) All the above\n")
    for k, v in test_return.items():
        if str(look_at_it).strip()=="1":
            print(f"{k}:\n")
            print(f"{v.description}")
            print("----------------")
        if str(look_at_it).strip()=="2":
            print(f"{k}:\n")
            print(f"{v.dataframe}")
            print("----------------")
        if str(look_at_it).strip()=="3":
            print(f"{k}:\n")
            print(f"{v.dataframe.to_json()}")
            print("----------------")
        if str(look_at_it).strip()=="4":
            print(f"{k}:\n")
            print(f"{v.description}")
            print("----------------")
            print(f"{v.dataframe}")
            print("----------------")
            print(f"{v.dataframe.to_json()}")
            print("----------------")
        
    
if __name__ == "__main__":
    test_main()
    


# Ignore main()
def main():
    """The script inside main needs to get split out into other functions"""
    print('booting up...\n')
    loan_data = pd.read_csv("raw_data/loans_v2.csv")
    # loan_data = pd.read_csv("raw_data/master_loan_tape.csv")
    # format date columns to datetime data types
    date_cols = [c for c in loan_data.columns if str(c)[-2:]=='Dt']
    for col in date_cols:
        loan_data[col] = pd.to_datetime(loan_data[col], format='mixed')
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
    
    elif subset_choice == '5' or subset_choice.lower() == 'state':
        STATE_DIRECTORY = create_data_directory(os.path.join(MAT_DIRECTORY,'state/'))
        print("Getting top 5 states...")
        top_5_states = maturity_slice[['GP','state_abbreviation','LoanAmt']].groupby('state_abbreviation').sum()[['LoanAmt']].sort_values('LoanAmt',ascending=False).head(5).index.to_list()
        state_reference = 'state_abbreviation'
        # state_slice = maturity_slice[maturity_slice['state_abbreviation'].isin(top_5_states)]
        # status = run_baseline_assembly(state_slice, STATE_DIRECTORY)
        status = run_bucket_assembly(maturity_slice=maturity_slice, buckets=top_5_states, col_ref=state_reference,parent_dir=STATE_DIRECTORY)
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
                print(f"I made this NAICS bucket for you: {bins}")
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

            else:
                print('column not currently supported')
            
            
        try:
            status = run_bucket_assembly(combo_slice,combo_buckets,f'CustomBuckets_{choice}',COMBO_DIRECTORY)
        except KeyError:
            print(combo_slice.head(10))
            print('\n' )
            print(bins)



    

