"""Port of the Fannie Mae LPPUB R stat-file workflow to pandas.

This script reads raw pipe-delimited Loan Performance files and produces the
one-row-per-loan statistical file created by LPPUB_StatFile.R and
LPPUB_StatFile_Production.R.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "2024Q1_30.csv"
OUTPUT_FILE = SCRIPT_DIR.parent / "data_prepare" / "2024Q1_30_1.csv"
NROWS = None


LPPUB_COLUMN_NAMES = [
'Reference Pool ID',
'Loan Identifier',
'Monthly Reporting Period',
'Channel',
'Seller Name',
'Servicer Name',
'Master Servicer',
'Original Interest Rate',
'Current Interest Rate',
'Original UPB',
'UPB at Issuance',
'Current Actual UPB',
'Original Loan Term',
'Origination Date',
'First Payment Date',
'Loan Age',
'Remaining Months to Legal Maturity',
'Remaining Months To Maturity',
'Maturity Date',
'Original Loan to Value Ratio (LTV)',
'Original Combined Loan to Value Ratio (CLTV)',
'Number of Borrowers',
'Debt-To-Income (DTI)',
'Borrower Credit Score at Origination',
'Co-Borrower Credit Score at Origination',
'First Time Home Buyer Indicator',
'Loan Purpose ',
'Property Type',
'Number of Units',
'Occupancy Status',
'Property State',
'Metropolitan Statistical Area (MSA) or Metropolitan Statistical Division Area (MSDA)',
'Zip Code Short',
'Mortgage Insurance Percentage',
'Amortization Type',
'Prepayment Penalty Indicator',
'Interest Only Loan Indicator',
'Interest Only First Principal And Interest Payment Date',
'Months to Amortization',
'Current Loan Delinquency Status',
'Loan Payment History',
'Modification Flag',
'Mortgage Insurance Cancellation Indicator',
'Zero Balance Code',
'Zero Balance Effective Date',
'UPB at the Time of Removal',
'Repurchase Date',
'Scheduled Principal Current',
'Total Principal Current',
'Unscheduled Principal Current',
'Last Paid Installment Date',
'Foreclosure Date',
'Disposition Date',
'Foreclosure Costs',
'Property Preservation and Repair Costs',
'Asset Recovery Costs',
'Miscellaneous Holding Expenses and Credits',
'Associated Taxes for Holding Property',
'Net Sales Proceeds',
'Credit Enhancement Proceeds',
'Repurchase Make Whole Proceeds',
'Other Foreclosure Proceeds',
'Modification-Related Non-Interest Bearing UPB',
'Principal Forgiveness Amount',
'Original List Start Date',
'Original List Price',
'Current List Start Date',
'Current List Price',
'Borrower Credit Score At Issuance',
'Co-Borrower Credit Score At Issuance',
'Borrower Credit Score Current ',
'Co-Borrower Credit Score Current',
'Mortgage Insurance Type',
'Servicing Activity Indicator',
'Current Period Modification Loss Amount',
'Cumulative Modification Loss Amount',
'Current Period Credit Event Net Gain or Loss',
'Cumulative Credit Event Net Gain or Loss',
'Special Eligibility Program',
'Foreclosure Principal Write-off Amount',
'Relocation Mortgage Indicator',
'Zero Balance Code Change Date',
'Loan Holdback Indicator',
'Loan Holdback Effective Date',
'Delinquent Accrued Interest',
'Property Valuation Method ',
'High Balance Loan Indicator ',
'ARM Initial Fixed-Rate Period  ≤ 5 YR Indicator',
'ARM Product Type',
'Initial Fixed-Rate Period ',
'Interest Rate Adjustment Frequency',
'Next Interest Rate Adjustment Date',
'Next Payment Change Date',
'Index',
'ARM Cap Structure',
'Initial Interest Rate Cap Up Percent',
'Periodic Interest Rate Cap Up Percent',
'Lifetime Interest Rate Cap Up Percent',
'Mortgage Margin',
'ARM Balloon Indicator',
'ARM Plan Number',
'Borrower Assistance Plan',
'High Loan to Value (HLTV) Refinance Option Indicator',
'Deal Name',
'Repurchase Make Whole Proceeds Flag',
'Alternative Delinquency Resolution',
'Alternative Delinquency  Resolution Count',
'Total Deferral Amount ',
'Payment Deferral Modification Event Indicator',
'Interest Bearing UPB',
]

NUMERIC_COLUMNS = [
"character", "character", "character", "character", "character", "character",
                          "character", "numeric", "numeric", "numeric", "numeric",
                          "numeric", "numeric", "character", "character", "numeric", "numeric",
                          "numeric", "character", "numeric", "numeric", "character", "numeric",
                          "numeric", "numeric", "character", "character", "character",
                          "numeric", "character", "character", "character", "character",
                          "numeric", "character", "character", "character", "character",
                          "numeric", "character", "character", "character", "character",
                          "character", "character", "numeric", "character", "numeric",
                          "numeric", "numeric", "character", "character", "character",
                          "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                          "numeric", "numeric", "numeric", "numeric", "numeric", "character",
                          "numeric", "character", "numeric", "numeric", "numeric", "numeric",
                          "numeric", "numeric", "character", "numeric", "numeric", "numeric",
                          "numeric", "character", "numeric", "character", "numeric", "character",
                          "numeric", "numeric", "character", "character", "numeric", "numeric",
                          "numeric", "numeric", "numeric", "numeric", "numeric", "numeric",
                          "numeric", "numeric", "numeric", "numeric", "numeric", "character",
                          "character", "character", "character", "character",
						  "character", "numeric", "numeric", "character", "numeric"
]


LPPUB_COLUMN_NAMES_SELECT=[
'Reference Pool ID',
'Loan Identifier',
'Monthly Reporting Period',
'Channel',
'Original Interest Rate',
'Current Interest Rate',
'Original UPB',
'UPB at Issuance',
'Current Actual UPB',
'Original Loan Term',
'First Payment Date',
'Loan Age',
'Remaining Months to Legal Maturity',
'Remaining Months To Maturity',
'Original Loan to Value Ratio (LTV)',
'Original Combined Loan to Value Ratio (CLTV)',
'Number of Borrowers',
'Debt-To-Income (DTI)',
'Borrower Credit Score at Origination',
'Co-Borrower Credit Score at Origination',
'First Time Home Buyer Indicator',
'Loan Purpose ',
'Property Type',
'Number of Units',
'Occupancy Status',
'Property State',
'Metropolitan Statistical Area (MSA) or Metropolitan Statistical Division Area (MSDA)',
'Mortgage Insurance Percentage',
'Amortization Type',
'Prepayment Penalty Indicator',
'Interest Only Loan Indicator',
'Interest Only First Principal And Interest Payment Date',
'Months to Amortization',
'Loan Payment History',
'Modification Flag',
'Mortgage Insurance Cancellation Indicator',
'Zero Balance Code',
'Zero Balance Effective Date',
'UPB at the Time of Removal',
'Repurchase Date',
'Scheduled Principal Current',
'Total Principal Current',
'Unscheduled Principal Current',
'Last Paid Installment Date',
'Property Preservation and Repair Costs',
'Asset Recovery Costs',
'Miscellaneous Holding Expenses and Credits',
'Associated Taxes for Holding Property',
'Modification-Related Non-Interest Bearing UPB',
'Principal Forgiveness Amount',
'Original List Start Date',
'Original List Price',
'Current List Price',
'Borrower Credit Score At Issuance',
'Co-Borrower Credit Score At Issuance',
'Borrower Credit Score Current ',
'Co-Borrower Credit Score Current',
'Mortgage Insurance Type',
'Servicing Activity Indicator',
'Special Eligibility Program',
'Relocation Mortgage Indicator',
'Zero Balance Code Change Date',
'Loan Holdback Effective Date',
'Property Valuation Method ',
'High Balance Loan Indicator ',
'ARM Initial Fixed-Rate Period  ≤ 5 YR Indicator',
'ARM Product Type',
'Initial Fixed-Rate Period ',
'Interest Rate Adjustment Frequency',
'Next Interest Rate Adjustment Date',
'Next Payment Change Date',
'ARM Cap Structure',
'Initial Interest Rate Cap Up Percent',
'Periodic Interest Rate Cap Up Percent',
'Lifetime Interest Rate Cap Up Percent',
'Mortgage Margin',
'ARM Balloon Indicator',
'ARM Plan Number',
'Borrower Assistance Plan',
'High Loan to Value (HLTV) Refinance Option Indicator',
'Repurchase Make Whole Proceeds Flag',
'Total Deferral Amount ',
'Payment Deferral Modification Event Indicator',
'Interest Bearing UPB',
]





def load_lppub_file(filename: str | Path,months_select: list[str] = None) -> pd.DataFrame:
    df = pd.read_csv(
        filename,
        sep="|",
        header=None,
        names=LPPUB_COLUMN_NAMES,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="python",
    )
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].mask(df[col].eq(""), np.nan), errors="coerce")


    df=df[df['Monthly Reporting Period'].isin(months_select)]

    ##collect Loan ID which has prepay in the acquisition month, then exlucde such LOAN ID
    acq_m = months_select[0]
    acq_m_prepay_ID = df[
                            (df["Monthly Reporting Period"] == acq_m)
                            & (df["Zero Balance Code"] == "01")
                        ]["Loan Identifier"]
    df=df[~df['Loan Identifier'].isin(acq_m_prepay_ID)]

    ##reshape the df by selecting columns
    df = df[LPPUB_COLUMN_NAMES_SELECT]

    ##replace empty with nan
    df = df.replace("", np.nan)

    return df

def mmYYYY_to_date(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    mask = s.str.len() >= 6
    return pd.Series(
        np.where(mask, s.str.slice(2, 6) + "-" + s.str.slice(0, 2) + "-01", s),
        index=series.index,
    )


def acqm_next3m_from_filename(filename: str | Path) -> str:
    """
    given file name,
    return the list of acquisition month and next 3 month with year
    the monthyear format algins with the format of Monthly Reporting Period
    """
    match = re.search(r"(\d{4})Q([1-4])", Path(filename).name)
    if not match:
        raise ValueError(f"Cannot derive acquisition quarter from filename: {filename}")
    year, quarter = match.groups()
    months = {"1": ["03","04","05","06"], 
              "2":  ["06","07","08","09"], 
              "3": ["09","10","11","12"], 
              "4": ["12","01","02","03"]
                    }[quarter] # aqusition month and next 3 month
    if quarter =="4":
        newyear = str(int(year)+1)
        return [f"{months[0]}{year}"] + [f"{month}{newyear}" for month in months[1:]]
    else:
        return [f"{month}{year}" for month in months] 


def latest_rows(df: pd.DataFrame, group_col: str, period_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values([group_col, period_col]).drop_duplicates(group_col, keep="last")


def build_stat_file(input_file: str | Path) -> pd.DataFrame:
    ##collect rows whose Monthly Reporting Period is between aquisition month and aquisition + 3 month
    effective_months = acqm_next3m_from_filename(input_file)
    lppub_file = load_lppub_file(input_file,effective_months)
    aqu_month =effective_months[0]
    #next3months = effective_months[1:]

    lppub_file["Original Interest Rate"] = pd.to_numeric(lppub_file["Original Interest Rate"], errors="coerce")
    lppub_file["Current Interest Rate"] = pd.to_numeric(lppub_file["Current Interest Rate"], errors="coerce")


    lppub_file = lppub_file.sort_values(["Loan Identifier", "Monthly Reporting Period"])
    LoanIdentifiers = lppub_file['Loan Identifier'].unique()
    lst_singleLoanInfo=[]
    lst_singleloanPrepayInfo=[]
    for LoanID in LoanIdentifiers:
        lppub_file_sample1  =lppub_file[lppub_file["Loan Identifier"]==LoanID]
        loanInfoatAcquisition = lppub_file_sample1[lppub_file_sample1["Monthly Reporting Period"]==aqu_month]
        prepayNext3m = 1 if lppub_file_sample1['Zero Balance Code'].eq('01').any() else 0
    
        lst_singleLoanInfo.append(loanInfoatAcquisition)
        lst_singleloanPrepayInfo.append(prepayNext3m)

    if len(lst_singleLoanInfo) != len(lst_singleloanPrepayInfo):
        raise ValueError("length of 2 lists(single loans and prepay info) are not the same")
    loanPerformance_df = pd.concat(lst_singleLoanInfo)
    loanPerformance_df['Prepaied_3m'] = lst_singleloanPrepayInfo

    return loanPerformance_df


def format_for_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(format_number)
        else:
            out[col] = out[col].mask(out[col].eq(""), np.nan)
    return out


def format_number(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.10f}".rstrip("0").rstrip(".")



def main() -> None:
    stat_file = build_stat_file(INPUT_FILE)
    formatted = format_for_output(stat_file)
    formatted.to_csv(OUTPUT_FILE, index=False, na_rep="NULL", quoting=1)
    print(f"Wrote {len(formatted)} loan rows and {len(formatted.columns)} columns to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
