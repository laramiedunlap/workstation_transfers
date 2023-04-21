import pandas as pd
from pandas._libs.tslibs.nattype import NaTType
import numpy as np
import math
import datetime

# Define pooler class
class Pooler: 
    loans: dict
    triangles: dict
    prepays_errs: int
    defaults_errs: int

    def __init__(self, l_dict) -> None:
        self.loans = l_dict
        self.triangles = {}
        self.prepays_errs = 0
        self.defaults_errs = 0
    #  This method builds the data we're used to seeing
    def build_triangles_counts(self):
        """Stratify along year and month originations"""
        # Loop through each individual loan
        for l in self.loans.values():
            # Get the values for when the loan originated, prepaid, and defaulted
            origination = l.short_orig_str
            pp_mths_qty = l.prepay_mths_qty
            default_mths_qty = l.default_mths_qty
            # If the origination --> read mini-cohort --> already exists, increase the starting loan count for that pool
            if origination in self.triangles.keys():
                self.triangles[origination][0][0] += 1
                # Both of these only work if a prepayment date or default date exists
                # Count default first --> add one to the default triangles for this subcohort at the position X months from origination
                if not np.isnan(default_mths_qty):
                    self.triangles[origination][2][int(default_mths_qty)] += 1
                # If the loan didn't default, count prepay, same way as above
                elif not np.isnan(pp_mths_qty):
                    self.triangles[origination][1][int(pp_mths_qty)] += 1
            # This block runs when the Pooler hasn't seen this mini-vintage before
            else:
                # If this is the first time seeing this loan origination date and month, create arrays to store the data
                self.triangles[origination] =  np.array([np.repeat([0],300), np.repeat([0],300), np.repeat([0],300)], dtype=np.int64)
                self.triangles[origination][0][0] += 1
                # Count default first
                if not np.isnan(default_mths_qty):
                    self.triangles[origination][2][int(default_mths_qty)] += 1
                # Then count prepay
                elif not np.isnan(pp_mths_qty):
                    self.triangles[origination][1][int(pp_mths_qty)] += 1
                    
        # After all Prepays and Defaults have been counted, calculate survior array 
        for t in self.triangles.keys():
            a = self.triangles[t][0]
            b = self.triangles[t][1]
            c = self.triangles[t][2]
            # This is saying --> starting with the SECOND column (1 month after origination), 
            # subtract the prepayments and defaults from the previous month, so that the loan count at the start of each month rolls down
            for i in range(1, len(a)):
                a[i] = a[i-1] - (b[i-1] + c[i-1])
            self.triangles[t][0] = a
    # This code gives the arrays we just built the Triangle look, as opposed to seeing a loan count remain constant 
    def get_max_tbl_cols(self)-> int:
        start_date = None
        end_date = None
        for l in self.loans.values():
            if not start_date:
                start_date = l.origination_dt
                end_date = l.origination_dt
            elif l.origination_dt < start_date:
                start_date = l.origination_dt
            if l.origination_dt > end_date:
                end_date = l.origination_dt        
        time_delta = end_date-start_date
        return (math.ceil((time_delta.days)/31))

    # This method is how I originally formatted the data to be printed out like this to a csv file.
    def format_raw_output(self)-> dict:
        orig_list = self.triangles.keys()
        orig_list = sorted([float(e) for e in orig_list])
        output = {
            'counts': [],
            'prepay': [],
            'default': []
            } 
        for e in orig_list:
                line_ref = "{:.2f}".format(e)
                output['counts'].append(list(self.triangles[line_ref][0]))
                output['prepay'].append(list(self.triangles[line_ref][1]))
                output['default'].append(list(self.triangles[line_ref][2]))
                if line_ref[-2:] == '.1':
                    line_ref = line_ref[:-2]+'.10'
                output['counts'][-1].insert(0,line_ref)
                output['prepay'][-1].insert(0,line_ref)
                output['default'][-1].insert(0,line_ref)

        return output

# This code stores loan information. It's just a way to store all relevant prepayment/default data on a loan in one place
class Loan:
    ObservationNmb: int
    origination_dt: datetime.date
    maturity_dt: datetime.date
    maturity_mths_qty: int
    prepay_dt: datetime.date
    prepay_mths_qty: int
    default_dt: datetime.date
    default_mths_qty: int
    origination_yr: datetime.date.year
    origination_mth: datetime.date.month
    short_orig_str: str

    def __init__(self, obs, o_dt) -> None:
        self.ObservationNmb = obs
        self.origination_dt = o_dt
        self.origination_yr = o_dt.year
        self.origination_mth = o_dt.month
        if int(self.origination_mth) < 10:
            string = str(self.origination_yr)+'.0'+str(self.origination_mth)
            x = float(string)
            self.short_orig_str = "{:.2f}".format(x)
        else:
            string = str(self.origination_yr)+'.'+str(self.origination_mth)
            x = float(string)
            self.short_orig_str = "{:.2f}".format(x)
        return None

    def calc_mat_mths(self) -> None:
        """Number of months from origination to maturity"""
        temp_diff = (self.maturity_dt - self.origination_dt)
        temp_diff = math.ceil(temp_diff.days/31)
        self.maturity_mths_qty = int(temp_diff)
        return None
    
    def calc_default_mths(self)-> None:
        """Number of months from origination to default"""
        if type(self.default_dt) == NaTType:
            self.default_mths_qty = NaTType()
        else:
            temp_diff = (self.default_dt - self.origination_dt)
            temp_diff = math.ceil(temp_diff.days/31)
            self.default_mths_qty = int(temp_diff)
        return None

    def calc_prepay_mths(self) -> None:
        """Number of months from origination to full prepay"""
        if type(self.prepay_dt) == NaTType:
            self.prepay_mths_qty = NaTType()
        elif type(self.default_dt) != NaTType:
            self.prepay_dt = NaTType()
            self.prepay_mths_qty = NaTType()
        else:
            temp_diff = (self.prepay_dt - self.origination_dt)
            temp_diff = math.ceil(temp_diff.days/31)
            self.prepay_mths_qty = int(temp_diff)
        return None
    
    def __str__(self) -> str:
        """Return strings of all attributes"""
        return str(self.ObservationNmb), str(self.origination_dt), \
        str(self.origination_yr), str(self.origination_mth), \
        str(self.maturity_dt), str(self.maturity_mths_qty), \
        str(self.default_dt),str(self.default_mths_qty), \
        str(self.prepay_dt), str(self.prepay_mths_qty)
    



        