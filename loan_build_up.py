import numpy as np
import pandas as pd
import sqlalchemy as sa
import datetime
import pandas as pd
from dotenv import load_dotenv
import os
import pickle
from tqdm import tqdm

load_dotenv()


class Loan:
    ObservationNmb: int
    maturity_dt: datetime.date
    origination_dt: datetime.date
    lender_status: int
    all_pmts: np.ndarray
    
    def __init__(self, nmb, m_dt, o_dt, l_st,pmts):
        self.ObservationNmb = nmb
        self.maturity_dt = m_dt
        self.origination_dt = o_dt
        self.lender_status = l_st
        self.all_pmts = pmts

engine = sa.create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
			.format(host=os.getenv('host'), db=os.getenv('db'), user=os.getenv('uname'), pw=os.getenv('password')))

def get_ids()-> list:
    with engine.connect() as con:
            con.execute('SET GLOBAL innodb_buffer_pool_size=2147483648;')
            # Get ObservationNmb's for all non-revolver loans
            query = """ SELECT elipsmisc7afoia.ObservationNmb, repymnttbl7afoia.MaturityDt, elipsmisc7afoia.LoanFundedDt, loantbl7afoia.LenderStatusCd FROM elipsmisc7afoia
                    JOIN repymnttbl7afoia
                    ON repymnttbl7afoia.ObservationNmb = elipsmisc7afoia.ObservationNmb
                    JOIN loantbl7afoia
                    ON loantbl7afoia.ObservationNmb = elipsmisc7afoia.ObservationNmb
                    WHERE RevolvingLineofCreditCd = 'N' AND YEAR(LoanFundedDt) >= 2000;"""
                    
            print("aggregating loan ids...\n")
            rows = con.execute(query)
            return [list(ele) for ele in rows.fetchall()]

def get_loans(id_list:list)->dict:
    loans = {}
    for id in tqdm(id_list):
        with engine.connect() as con:    
            con.execute('SET GLOBAL innodb_buffer_pool_size=2147483648;')
            query = f"""SELECT EffectiveDt, TransacionCd, GeneralLedgerCd, TransactionAmt, TransactionBalanceAmt FROM master_fin WHERE ObservationNmb = \"{id[0]}\" """
            result = con.execute(query)
            rows = np.array([list(ele) for ele in result])
            loans[id[0]] = Loan(id[0], id[1], id[2], id[3], rows)
    return loans
        # DataStructure of Loan.pmt_history: 
        #   [
        #   [EffectiveDt, TransacionCd, GeneralLedgerCd, TransactionAmt, TransactionBalanceAmt],
        #   [EffectiveDt, TransacionCd, GeneralLedgerCd, TransactionAmt, TransactionBalanceAmt],
        #   [EffectiveDt, TransacionCd, GeneralLedgerCd, TransactionAmt, TransactionBalanceAmt], ...
        #   ]
           
def build_cohort(loan_dict, year):
    cohort = []
    for key in loan_dict:
        if loan_dict[key].origination_dt.year == year:
            cohort.append(loan_dict[key])
    return cohort

def do_principal_pmts(cohrt_ls: list):
    """WHENEVER YOU TEST A CHANGE --- CHANGE cohrt_ls to cohrt_ls[0:2] and see how it looks"""
    master_principal = []
    print('parsing all payment strings...')
    for x in (cohrt_ls):
        try:
            hist = x.all_pmts
            hist = np.insert(hist, 0, x.maturity_dt, axis=1)
            hist = np.insert(hist, 0, x.origination_dt, axis=1)      
            hist = np.insert(hist, 0, x.ObservationNmb, axis=1)
            hist_1510 = hist[ hist[:, 5]== '1510', :]
            hist_6031 = hist[ hist[:, 5]== '6031', :]
            principal_pmts = np.append(hist_1510, hist_6031, axis=0)
            order = principal_pmts[:, 3].argsort()
            principal_pmts = principal_pmts[order]
            master_principal.append(principal_pmts)
        except:
            continue
    print('returning master principal pmt structure')
    return master_principal

def build_df_list(prepayment_arr, cols):
    df_list = []
    for d in prepayment_arr:
        df = pd.DataFrame(data=d, columns= cols).drop_duplicates()
        df = df.reset_index().drop(columns='index')
        df_list.append(df)
    return df_list

def main():
    print('Getting IDs...\n')
    ids = get_ids()
    
    print('Building Loan Objects...\n')
    loans = get_loans(ids)
    with open('pickle_files/build-outs/loans.pickle', 'wb') as f:
        pickle.dump(loans, f)
        f.close()
        
    start_year = input("What is the beginning year?")
    end_year = input("What is the end year?")
    to_do_list = list(range(int(start_year), int(end_year)+1, 1))
    cohort_map = {}
    for yr in (to_do_list):
        print(f"collecting principal payments for vintage: {yr}")
        l = build_cohort(loans, yr)
        cohort_map[str(yr)] = do_principal_pmts(l)

    print('Converting to Pandas Dataframes...\n')
    columns = ['ObservationNmb', 'LoanFundedDt', 'MaturityDt', 'EffectiveDt', 'TransactionCd' ,'GeneralLedgerCd', 'TransactionAmt', 'TransactionBalanceAmt']
    for key in (cohort_map.keys()):
        print(f"converting vintage {key} to dataframes...")
        cohort_map[key] = build_df_list(cohort_map[key], columns)

    print('Sending data to build-outs folder...\n')
    with open('pickle_files/build-outs/build.pickle', 'wb') as f:
        pickle.dump(cohort_map, f)
        f.close()
    
    print('Complete')


if __name__ == "__main__":
    main()