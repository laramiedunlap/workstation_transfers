from collections import OrderedDict
import sqlalchemy as sa
import os
import numpy_financial as npf
import pandas as pd



def build_df_list(prepayment_arr, cols):
    df_list = []
    for d in prepayment_arr:
        df_list.append(pd.DataFrame(data=d, columns= cols))
    return df_list

def show_slice(df: pd.DataFrame, obs: str)-> pd.DataFrame:
    """show the payment string for a particular observation number. Use within notebooks to verify data and spot check GPs."""
    return df[df['ObservationNmb']==obs]

def create_amortization_lookup(engine_obj: sa.engine)->pd.DataFrame:
    query = f"""SELECT loantbl7afoia.ObservationNmb, loantbl7afoia.BankInterestPct, elipsamt7afoia.GrossGtyDisbursementAmt, loantbl7afoia.MaturityMthsQty from loantbl7afoia
    LEFT JOIN elipsamt7afoia ON loantbl7afoia.ObservationNmb = elipsamt7afoia.ObservationNmb;"""
    
    return pd.read_sql(query, engine_obj)

def amortize(interest_rate: float, begLoanBalance: float, begMonths: int)-> OrderedDict:
    """Create an OrderedDict with a loan's expected amortization schedule"""
    """NOTE: This will work for most loans, however there are loans that have quarterly payments. So there 
    should be another method that checks periodicity of loan payments."""
    cnt = 0 
    beg_period_balance = begLoanBalance
    remaining_months = begMonths    
    schedLoanPmt = round(npf.pmt((interest_rate/12), begMonths, begLoanBalance*-1), 0)
    
    while schedLoanPmt > 0:
        try: 
            # Calculate payment
            schedLoanPmt = round(npf.pmt((interest_rate/12), remaining_months, beg_period_balance*-1), 0)
            # Calculate scheduled interest
            schedInterestPmt = round(beg_period_balance*(interest_rate/12), 0)
            # Check to see if this payment will pay off the loan
            pmt = min(schedLoanPmt, beg_period_balance + schedInterestPmt)
            # Calculate scheduled principal
            schedPrincipal = round(pmt - schedInterestPmt, 2)
            # Calculate Ending Loan Balance
            endingLoanBalance = round(beg_period_balance - schedPrincipal,0)
            cnt += 1
            
            yield OrderedDict([('Month', cnt), 
                            ('Begging Loan Balance', beg_period_balance),
                            ('Scheduled Loan Payment', pmt),
                            ('Scheduled Interest Payment', schedInterestPmt),
                            ('Scheduled Principal', schedPrincipal),
                            ('Ending Loan Balance', endingLoanBalance)])
            beg_period_balance = endingLoanBalance
            remaining_months -= 1
            
        except IndexError:
            print(f"error at index: {cnt}")
            
