"""Port of the Fannie Mae LPPUB R stat-file workflow to pandas.

This script reads raw pipe-delimited Loan Performance files and produces the
one-row-per-loan statistical file created by LPPUB_StatFile.R and
LPPUB_StatFile_Production.R.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


LPPUB_COLUMN_NAMES = [
    "POOL_ID", "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "SERVICER",
    "MASTER_SERVICER", "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "ISSUANCE_UPB",
    "CURRENT_UPB", "ORIG_TERM", "ORIG_DATE", "FIRST_PAY", "LOAN_AGE",
    "REM_MONTHS", "ADJ_REM_MONTHS", "MATR_DT", "OLTV", "OCLTV",
    "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE",
    "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP", "MI_PCT",
    "PRODUCT", "PPMT_FLG", "IO", "FIRST_PAY_IO", "MNTHS_TO_AMTZ_IO",
    "DLQ_STATUS", "PMT_HISTORY", "MOD_FLAG", "MI_CANCEL_FLAG", "Zero_Bal_Code",
    "ZB_DTE", "LAST_UPB", "RPRCH_DTE", "CURR_SCHD_PRNCPL", "TOT_SCHD_PRNCPL",
    "UNSCHD_PRNCPL_CURR", "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE",
    "DISPOSITION_DATE", "FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
    "ASSET_RECOVERY_COSTS", "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
    "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
    "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
    "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB",
    "PRINCIPAL_FORGIVENESS_AMOUNT", "ORIGINAL_LIST_START_DATE",
    "ORIGINAL_LIST_PRICE", "CURRENT_LIST_START_DATE", "CURRENT_LIST_PRICE",
    "ISSUE_SCOREB", "ISSUE_SCOREC", "CURR_SCOREB", "CURR_SCOREC", "MI_TYPE",
    "SERV_IND", "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
    "CUMULATIVE_MODIFICATION_LOSS_AMOUNT",
    "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
    "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS", "HOMEREADY_PROGRAM_INDICATOR",
    "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", "RELOCATION_MORTGAGE_INDICATOR",
    "ZERO_BALANCE_CODE_CHANGE_DATE", "LOAN_HOLDBACK_INDICATOR",
    "LOAN_HOLDBACK_EFFECTIVE_DATE", "DELINQUENT_ACCRUED_INTEREST",
    "PROPERTY_INSPECTION_WAIVER_INDICATOR", "HIGH_BALANCE_LOAN_INDICATOR",
    "ARM_5_YR_INDICATOR", "ARM_PRODUCT_TYPE", "MONTHS_UNTIL_FIRST_PAYMENT_RESET",
    "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET", "INTEREST_RATE_CHANGE_DATE",
    "PAYMENT_CHANGE_DATE", "ARM_INDEX", "ARM_CAP_STRUCTURE",
    "INITIAL_INTEREST_RATE_CAP", "PERIODIC_INTEREST_RATE_CAP",
    "LIFETIME_INTEREST_RATE_CAP", "MARGIN", "BALLOON_INDICATOR",
    "PLAN_NUMBER", "FORBEARANCE_INDICATOR",
    "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR", "DEAL_NAME",
    "RE_PROCS_FLAG", "ADR_TYPE", "ADR_COUNT", "ADR_UPB",
    "PAYMENT_DEFERRAL_MOD_EVENT_FLAG", "INTEREST_BEARING_UPB",
]

NUMERIC_COLUMNS = {
    "ORIG_RATE", "CURR_RATE", "ORIG_UPB", "ISSUANCE_UPB", "CURRENT_UPB",
    "ORIG_TERM", "LOAN_AGE", "REM_MONTHS", "ADJ_REM_MONTHS", "OLTV", "OCLTV",
    "DTI", "CSCORE_B", "CSCORE_C", "NO_UNITS", "MI_PCT", "LAST_UPB",
    "CURR_SCHD_PRNCPL", "TOT_SCHD_PRNCPL", "UNSCHD_PRNCPL_CURR",
    "FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
    "ASSET_RECOVERY_COSTS", "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
    "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
    "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
    "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB",
    "PRINCIPAL_FORGIVENESS_AMOUNT", "ORIGINAL_LIST_PRICE",
    "CURRENT_LIST_PRICE", "ISSUE_SCOREB", "ISSUE_SCOREC", "CURR_SCOREB",
    "CURR_SCOREC", "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
    "CUMULATIVE_MODIFICATION_LOSS_AMOUNT",
    "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
    "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS",
    "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", "DELINQUENT_ACCRUED_INTEREST",
    "MONTHS_UNTIL_FIRST_PAYMENT_RESET", "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET",
    "INITIAL_INTEREST_RATE_CAP", "PERIODIC_INTEREST_RATE_CAP",
    "LIFETIME_INTEREST_RATE_CAP", "MARGIN", "ADR_COUNT", "ADR_UPB",
    "INTEREST_BEARING_UPB",
}

FINAL_COLUMNS = [
    "LOAN_ID", "ORIG_CHN", "SELLER", "orig_rt", "orig_amt", "orig_trm", "oltv",
    "ocltv", "num_bo", "dti", "CSCORE_B", "FTHB_FLG", "purpose", "PROP_TYP",
    "NUM_UNIT", "occ_stat", "state", "zip_3", "mi_pct", "CSCORE_C", "relo_flg",
    "MI_TYPE", "AQSN_DTE", "ORIG_DTE", "FRST_DTE", "LAST_RT", "LAST_UPB", "msa",
    "FCC_COST", "PP_COST", "AR_COST", "IE_COST", "TAX_COST", "NS_PROCS",
    "CE_PROCS", "RMW_PROCS", "O_PROCS", "repch_flag", "LAST_ACTIVITY_DATE",
    "LPI_DTE", "FCC_DTE", "DISP_DTE", "SERVICER", "F30_DTE", "F60_DTE",
    "F90_DTE", "F120_DTE", "F180_DTE", "FCE_DTE", "F180_UPB", "FCE_UPB",
    "F30_UPB", "F60_UPB", "F90_UPB", "MOD_FLAG", "FMOD_DTE", "FMOD_UPB",
    "MODIR_COST", "MODFB_COST", "MODFG_COST", "MODTRM_CHNG", "MODUPB_CHNG",
    "z_num_periods_120", "F120_UPB", "CSCORE_MN", "ORIG_VAL", "LAST_DTE",
    "LAST_STAT", "COMPLT_FLG", "INT_COST", "PFG_COST", "NET_LOSS", "NET_SEV",
    "MODTOT_COST",
]


def load_lppub_file(filename: str | Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        filename,
        sep="|",
        header=None,
        names=LPPUB_COLUMN_NAMES,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        nrows=nrows,
        engine="python",
    )
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].mask(df[col].eq(""), np.nan), errors="coerce")
    return df


def mmYYYY_to_date(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    mask = s.str.len() >= 6
    return pd.Series(
        np.where(mask, s.str.slice(2, 6) + "-" + s.str.slice(0, 2) + "-01", s),
        index=series.index,
    )


def month_number(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    return pd.to_numeric(s.str.slice(0, 4), errors="coerce") * 12 + pd.to_numeric(
        s.str.slice(5, 7), errors="coerce"
    )


def acquisition_date_from_filename(filename: str | Path) -> str:
    match = re.search(r"(\d{4})Q([1-4])", Path(filename).name)
    if not match:
        raise ValueError(f"Cannot derive acquisition quarter from filename: {filename}")
    year, quarter = match.groups()
    month = {"1": "03", "2": "06", "3": "09", "4": "12"}[quarter]
    return f"{year}-{month}-01"


def latest_rows(df: pd.DataFrame, group_col: str, period_col: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df.sort_values([group_col, period_col]).drop_duplicates(group_col, keep="last")


def first_event_table(
    slim: pd.DataFrame,
    mask: pd.Series,
    date_col: str,
    upb_col: str,
) -> pd.DataFrame:
    event = slim.loc[mask].copy()
    if event.empty:
        return pd.DataFrame(columns=["LOAN_ID", date_col, upb_col])
    first_dates = event.groupby("LOAN_ID", as_index=False)["period"].min().rename(
        columns={"period": date_col}
    )
    out = first_dates.merge(
        slim[["LOAN_ID", "period", "act_upb"]],
        left_on=["LOAN_ID", date_col],
        right_on=["LOAN_ID", "period"],
        how="left",
    )
    return out[["LOAN_ID", date_col, "act_upb"]].rename(columns={"act_upb": upb_col})


def add_missing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_stat_file(input_file: str | Path, nrows: int | None = None) -> pd.DataFrame:
    lppub_file = load_lppub_file(input_file, nrows=nrows)
    lppub_file["ORIG_RATE"] = pd.to_numeric(lppub_file["ORIG_RATE"], errors="coerce")
    lppub_file["CURR_RATE"] = pd.to_numeric(lppub_file["CURR_RATE"], errors="coerce")

    base_cols = [
        "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "SERVICER", "ORIG_RATE",
        "CURR_RATE", "ORIG_UPB", "CURRENT_UPB", "ORIG_TERM", "ORIG_DATE",
        "FIRST_PAY", "LOAN_AGE", "REM_MONTHS", "ADJ_REM_MONTHS", "MATR_DT",
        "OLTV", "OCLTV", "NUM_BO", "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG",
        "PURPOSE", "PROP", "NO_UNITS", "OCC_STAT", "STATE", "MSA", "ZIP",
        "MI_PCT", "PRODUCT", "DLQ_STATUS", "MOD_FLAG", "Zero_Bal_Code", "ZB_DTE",
        "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE", "DISPOSITION_DATE",
        "FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
        "ASSET_RECOVERY_COSTS", "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
        "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
        "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
        "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB",
        "PRINCIPAL_FORGIVENESS_AMOUNT", "RELOCATION_MORTGAGE_INDICATOR",
        "MI_TYPE", "SERV_IND", "RPRCH_DTE", "LAST_UPB",
    ]
    lppub_base = lppub_file[base_cols].copy()
    lppub_base["repch_flag"] = np.where(lppub_base["RPRCH_DTE"].fillna("") != "", 1, 0)
    lppub_base["ACT_PERIOD"] = mmYYYY_to_date(lppub_base["ACT_PERIOD"])
    lppub_base["FIRST_PAY"] = mmYYYY_to_date(lppub_base["FIRST_PAY"])
    lppub_base["ORIG_DATE"] = mmYYYY_to_date(lppub_base["ORIG_DATE"])
    lppub_base = lppub_base.sort_values(["LOAN_ID", "ACT_PERIOD"])

    acquisition = lppub_base[
        [
            "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "ORIG_RATE", "ORIG_UPB",
            "ORIG_TERM", "ORIG_DATE", "FIRST_PAY", "OLTV", "OCLTV", "NUM_BO",
            "DTI", "CSCORE_B", "CSCORE_C", "FIRST_FLAG", "PURPOSE", "PROP",
            "NO_UNITS", "OCC_STAT", "STATE", "ZIP", "MI_PCT", "PRODUCT",
            "MI_TYPE", "RELOCATION_MORTGAGE_INDICATOR",
        ]
    ].rename(
        columns={
            "CHANNEL": "ORIG_CHN",
            "ORIG_RATE": "orig_rt",
            "ORIG_UPB": "orig_amt",
            "ORIG_TERM": "orig_trm",
            "ORIG_DATE": "orig_date",
            "FIRST_PAY": "first_pay",
            "OLTV": "oltv",
            "OCLTV": "ocltv",
            "NUM_BO": "num_bo",
            "DTI": "dti",
            "FIRST_FLAG": "FTHB_FLG",
            "PURPOSE": "purpose",
            "PROP": "PROP_TYP",
            "NO_UNITS": "NUM_UNIT",
            "OCC_STAT": "occ_stat",
            "STATE": "state",
            "ZIP": "zip_3",
            "MI_PCT": "mi_pct",
            "PRODUCT": "prod_type",
            "RELOCATION_MORTGAGE_INDICATOR": "relo_flg",
        }
    )
    first_period = acquisition.groupby("LOAN_ID", as_index=False)["ACT_PERIOD"].max().rename(
        columns={"ACT_PERIOD": "first_period"}
    )
    acquisition_file = first_period.merge(
        acquisition,
        left_on=["LOAN_ID", "first_period"],
        right_on=["LOAN_ID", "ACT_PERIOD"],
        how="left",
    )[
        [
            "LOAN_ID", "ORIG_CHN", "SELLER", "orig_rt", "orig_amt", "orig_trm",
            "orig_date", "first_pay", "oltv", "ocltv", "num_bo", "dti",
            "CSCORE_B", "CSCORE_C", "FTHB_FLG", "purpose", "PROP_TYP",
            "NUM_UNIT", "occ_stat", "state", "zip_3", "mi_pct", "prod_type",
            "MI_TYPE", "relo_flg",
        ]
    ]

    performance_file = lppub_base[
        [
            "LOAN_ID", "ACT_PERIOD", "SERVICER", "CURR_RATE", "CURRENT_UPB",
            "LOAN_AGE", "REM_MONTHS", "ADJ_REM_MONTHS", "MATR_DT", "MSA",
            "DLQ_STATUS", "MOD_FLAG", "Zero_Bal_Code", "ZB_DTE",
            "LAST_PAID_INSTALLMENT_DATE", "FORECLOSURE_DATE", "DISPOSITION_DATE",
            "FORECLOSURE_COSTS", "PROPERTY_PRESERVATION_AND_REPAIR_COSTS",
            "ASSET_RECOVERY_COSTS", "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS",
            "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY", "NET_SALES_PROCEEDS",
            "CREDIT_ENHANCEMENT_PROCEEDS", "REPURCHASES_MAKE_WHOLE_PROCEEDS",
            "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB",
            "PRINCIPAL_FORGIVENESS_AMOUNT", "repch_flag", "LAST_UPB",
        ]
    ].rename(
        columns={
            "ACT_PERIOD": "period",
            "SERVICER": "servicer",
            "CURR_RATE": "curr_rte",
            "CURRENT_UPB": "act_upb",
            "LOAN_AGE": "loan_age",
            "REM_MONTHS": "rem_mths",
            "ADJ_REM_MONTHS": "adj_rem_months",
            "MATR_DT": "maturity_date",
            "MSA": "msa",
            "DLQ_STATUS": "dlq_status",
            "MOD_FLAG": "mod_ind",
            "Zero_Bal_Code": "z_zb_code",
            "ZB_DTE": "zb_date",
            "LAST_PAID_INSTALLMENT_DATE": "lpi_dte",
            "FORECLOSURE_DATE": "fcc_dte",
            "DISPOSITION_DATE": "disp_dte",
            "FORECLOSURE_COSTS": "FCC_COST",
            "PROPERTY_PRESERVATION_AND_REPAIR_COSTS": "PP_COST",
            "ASSET_RECOVERY_COSTS": "AR_COST",
            "MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS": "IE_COST",
            "ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY": "TAX_COST",
            "NET_SALES_PROCEEDS": "NS_PROCS",
            "CREDIT_ENHANCEMENT_PROCEEDS": "CE_PROCS",
            "REPURCHASES_MAKE_WHOLE_PROCEEDS": "RMW_PROCS",
            "OTHER_FORECLOSURE_PROCEEDS": "O_PROCS",
            "NON_INTEREST_BEARING_UPB": "non_int_upb",
            "PRINCIPAL_FORGIVENESS_AMOUNT": "prin_forg_upb",
            "LAST_UPB": "zb_upb",
        }
    )

    acquisition_date = acquisition_date_from_filename(input_file)
    acquisition_file = acquisition_file.rename(
        columns={"orig_date": "ORIG_DTE", "first_pay": "FRST_DTE"}
    )
    for date_col in ["maturity_date", "zb_date", "lpi_dte", "fcc_dte", "disp_dte"]:
        performance_file[date_col] = mmYYYY_to_date(performance_file[date_col])

    base1 = acquisition_file.copy()
    base1["AQSN_DTE"] = acquisition_date
    base1["MI_TYPE"] = np.select(
        [base1["MI_TYPE"].eq("1"), base1["MI_TYPE"].eq("2"), base1["MI_TYPE"].eq("3")],
        ["BPMI", "LPMI", "IPMI"],
        default="None",
    )
    base1["ocltv"] = base1["ocltv"].fillna(base1["oltv"])

    last_activity = performance_file.groupby("LOAN_ID", as_index=False)["period"].max().rename(
        columns={"period": "LAST_ACTIVITY_DATE"}
    )
    last_upb = latest_rows(performance_file, "LOAN_ID", "period").copy()
    last_upb["LAST_UPB"] = np.where(last_upb["zb_upb"].notna(), last_upb["zb_upb"], last_upb["act_upb"])
    last_upb = last_upb[["LOAN_ID", "LAST_UPB"]]

    last_rt_source = performance_file[performance_file["curr_rte"].notna()]
    if last_rt_source.empty:
        last_rt = pd.DataFrame(columns=["LOAN_ID", "LAST_RT"])
    else:
        last_rt = latest_rows(last_rt_source, "LOAN_ID", "period")[["LOAN_ID", "curr_rte"]].rename(
            columns={"curr_rte": "LAST_RT"}
        )
        last_rt["LAST_RT"] = last_rt["LAST_RT"].round(3)

    zb_source = performance_file[performance_file["z_zb_code"].fillna("") != ""]
    if zb_source.empty:
        zb_code = pd.DataFrame(columns=["LOAN_ID", "zb_code"])
    else:
        zb_code = latest_rows(zb_source, "LOAN_ID", "period")[["LOAN_ID", "z_zb_code"]].rename(
            columns={"z_zb_code": "zb_code"}
        )

    max_table = (
        last_activity.merge(
            performance_file,
            left_on=["LOAN_ID", "LAST_ACTIVITY_DATE"],
            right_on=["LOAN_ID", "period"],
            how="left",
        )
        .merge(last_upb, on="LOAN_ID", how="left")
        .merge(last_rt, on="LOAN_ID", how="left")
        .merge(zb_code, on="LOAN_ID", how="left")
    )

    serv_source = performance_file[performance_file["servicer"].fillna("") != ""]
    if serv_source.empty:
        servicer = pd.DataFrame(columns=["LOAN_ID", "SERVICER"])
    else:
        servicer = latest_rows(serv_source, "LOAN_ID", "period")[["LOAN_ID", "servicer"]].rename(
            columns={"servicer": "SERVICER"}
        )

    non_int_rows = []
    for loan_id, grp in performance_file.sort_values(["LOAN_ID", "period"]).groupby("LOAN_ID"):
        if len(grp) > 1:
            non_int_rows.append({"LOAN_ID": loan_id, "NON_INT_UPB": grp.iloc[-2]["non_int_upb"]})
    non_int = pd.DataFrame(non_int_rows, columns=["LOAN_ID", "NON_INT_UPB"])
    if not non_int.empty:
        non_int["NON_INT_UPB"] = non_int["NON_INT_UPB"].fillna(0)

    base2 = base1.merge(max_table, on="LOAN_ID", how="left").merge(
        servicer, on="LOAN_ID", how="left"
    ).merge(non_int, on="LOAN_ID", how="left")

    fcc_source = performance_file[
        performance_file["lpi_dte"].notna()
        & performance_file["fcc_dte"].notna()
        & performance_file["disp_dte"].notna()
    ]
    fcc = fcc_source.groupby("LOAN_ID", as_index=False).agg(
        LPI_DTE=("lpi_dte", "max"), FCC_DTE=("fcc_dte", "max"), DISP_DTE=("disp_dte", "max")
    )
    base3 = base2.merge(fcc, on="LOAN_ID", how="left")

    slim = performance_file[
        ["LOAN_ID", "period", "dlq_status", "z_zb_code", "act_upb", "zb_upb", "mod_ind", "maturity_date", "rem_mths"]
    ].copy()
    slim["dlq_status"] = slim["dlq_status"].replace("XX", "999")
    slim["dlq_num"] = pd.to_numeric(slim["dlq_status"], errors="coerce")
    active = slim["z_zb_code"].fillna("").eq("")
    f30 = first_event_table(slim, (slim["dlq_num"] >= 1) & (slim["dlq_num"] < 999) & active, "F30_DTE", "F30_UPB")
    f60 = first_event_table(slim, (slim["dlq_num"] >= 2) & (slim["dlq_num"] < 999) & active, "F60_DTE", "F60_UPB")
    f90 = first_event_table(slim, (slim["dlq_num"] >= 3) & (slim["dlq_num"] < 999) & active, "F90_DTE", "F90_UPB")
    f120 = first_event_table(slim, (slim["dlq_num"] >= 4) & (slim["dlq_num"] < 999) & active, "F120_DTE", "F120_UPB")
    f180 = first_event_table(slim, (slim["dlq_num"] >= 6) & (slim["dlq_num"] < 999) & active, "F180_DTE", "F180_UPB")

    fce_source = slim[
        slim["z_zb_code"].isin(["02", "03", "09", "15"])
        | ((slim["dlq_num"] >= 6) & (slim["dlq_num"] < 999))
    ]
    if fce_source.empty:
        fce = pd.DataFrame(columns=["LOAN_ID", "FCE_DTE", "FCE_UPB"])
    else:
        fce_dates = fce_source.groupby("LOAN_ID", as_index=False)["period"].min().rename(
            columns={"period": "FCE_DTE"}
        )
        fce = fce_dates.merge(
            slim[["LOAN_ID", "period", "act_upb", "zb_upb"]],
            left_on=["LOAN_ID", "FCE_DTE"],
            right_on=["LOAN_ID", "period"],
            how="left",
        )
        fce["FCE_UPB"] = fce["zb_upb"] + fce["act_upb"]
        fce = fce[["LOAN_ID", "FCE_DTE", "FCE_UPB"]]

    fmod_source = slim[(slim["mod_ind"].eq("Y")) & active]
    if fmod_source.empty:
        fmod = pd.DataFrame(columns=["LOAN_ID", "FMOD_DTE", "FMOD_UPB", "maturity_date"])
    else:
        fmod_dates = fmod_source.groupby("LOAN_ID", as_index=False)["period"].min().rename(
            columns={"period": "FMOD_DTE"}
        )
        fmod_window = fmod_source.merge(fmod_dates, on="LOAN_ID", how="left")
        fmod_window = fmod_window[
            month_number(fmod_window["period"]) <= month_number(fmod_window["FMOD_DTE"]) + 3
        ]
        fmod_upb = fmod_window.groupby("LOAN_ID", as_index=False)["act_upb"].max().rename(
            columns={"act_upb": "FMOD_UPB"}
        )
        fmod = (
            fmod_upb.merge(fmod_dates, on="LOAN_ID", how="left")
            .merge(
                slim[["LOAN_ID", "period", "maturity_date"]],
                left_on=["LOAN_ID", "FMOD_DTE"],
                right_on=["LOAN_ID", "period"],
                how="left",
            )[["LOAN_ID", "FMOD_DTE", "FMOD_UPB", "maturity_date"]]
        )

    num120 = f120.merge(acquisition_file[["LOAN_ID", "FRST_DTE"]], on="LOAN_ID", how="left")
    if num120.empty:
        num120 = pd.DataFrame(columns=["LOAN_ID", "z_num_periods_120"])
    else:
        num120["z_num_periods_120"] = month_number(num120["F120_DTE"]) - month_number(num120["FRST_DTE"]) + 1
        num120 = num120[["LOAN_ID", "z_num_periods_120"]]

    maturity_source = slim[slim["maturity_date"].notna()]
    if maturity_source.empty:
        orig_maturity = pd.DataFrame(columns=["LOAN_ID", "orig_maturity_date"])
    else:
        orig_maturity = maturity_source.sort_values(["LOAN_ID", "period"]).drop_duplicates(
            "LOAN_ID", keep="first"
        )[["LOAN_ID", "maturity_date"]].rename(columns={"maturity_date": "orig_maturity_date"})

    trm = slim.sort_values(["LOAN_ID", "period"]).copy()
    trm["prev_rem_mths"] = trm.groupby("LOAN_ID")["rem_mths"].shift(1)
    trm["did_trm_chng"] = np.where((trm["rem_mths"] - trm["prev_rem_mths"]) >= 0, 1, 0)
    trm = trm[trm["did_trm_chng"].eq(1)].groupby("LOAN_ID", as_index=False)["period"].min().rename(
        columns={"period": "trm_chng_dt"}
    )
    modtrm = fmod.merge(orig_maturity, on="LOAN_ID", how="left").merge(trm, on="LOAN_ID", how="left")
    if modtrm.empty:
        modtrm = pd.DataFrame(columns=["LOAN_ID", "MODTRM_CHNG"])
    else:
        modtrm["MODTRM_CHNG"] = np.where(
            modtrm["maturity_date"].ne(modtrm["orig_maturity_date"]) | modtrm["trm_chng_dt"].notna(),
            1,
            0,
        )
        modtrm = modtrm[["LOAN_ID", "MODTRM_CHNG"]]

    pre_mod = slim.merge(fmod[["LOAN_ID", "FMOD_DTE"]], on="LOAN_ID", how="inner")
    pre_mod = pre_mod[pre_mod["period"] < pre_mod["FMOD_DTE"]]
    if pre_mod.empty:
        pre_mod_upb = pd.DataFrame(columns=["LOAN_ID", "pre_mod_upb"])
    else:
        pre_mod_upb = latest_rows(pre_mod, "LOAN_ID", "period")[["LOAN_ID", "act_upb"]].rename(
            columns={"act_upb": "pre_mod_upb"}
        )
    modupb = fmod.merge(pre_mod_upb, on="LOAN_ID", how="left")
    if modupb.empty:
        modupb = pd.DataFrame(columns=["LOAN_ID", "MODUPB_CHNG"])
    else:
        modupb["MODUPB_CHNG"] = np.where(modupb["FMOD_UPB"] >= modupb["pre_mod_upb"], 1, 0)
        modupb = modupb[["LOAN_ID", "MODUPB_CHNG"]]

    base4 = (
        base3.merge(f30, on="LOAN_ID", how="left")
        .merge(f60, on="LOAN_ID", how="left")
        .merge(f90, on="LOAN_ID", how="left")
        .merge(f120, on="LOAN_ID", how="left")
        .merge(f180, on="LOAN_ID", how="left")
        .merge(fce, on="LOAN_ID", how="left")
        .merge(fmod, on="LOAN_ID", how="left")
        .merge(num120, on="LOAN_ID", how="left")
        .merge(modtrm, on="LOAN_ID", how="left")
        .merge(modupb, on="LOAN_ID", how="left")
    )
    for date_col, upb_col in [
        ("F30_DTE", "F30_UPB"),
        ("F60_DTE", "F60_UPB"),
        ("F90_DTE", "F90_UPB"),
        ("F120_DTE", "F120_UPB"),
        ("F180_DTE", "F180_UPB"),
        ("FCE_DTE", "FCE_UPB"),
    ]:
        base4[upb_col] = np.where(base4[upb_col].isna() & base4[date_col].notna(), base4["orig_amt"], base4[upb_col])

    base5 = base4.copy()
    base5["LAST_DTE"] = np.where(base5["disp_dte"].fillna("") != "", base5["disp_dte"], base5["LAST_ACTIVITY_DATE"])
    base5["repch_flag"] = np.where(base5["repch_flag"].eq("Y"), 1, 0)
    base5["PFG_COST"] = base5["prin_forg_upb"]
    base5["MOD_FLAG"] = np.where(base5["FMOD_DTE"].notna(), 1, 0)
    base5["MODFG_COST"] = np.where(base5["mod_ind"].eq("Y") & (base5["PFG_COST"] > 0), base5["PFG_COST"], 0)
    base5["MODTRM_CHNG"] = pd.to_numeric(base5["MODTRM_CHNG"], errors="coerce").fillna(0)
    base5["MODUPB_CHNG"] = pd.to_numeric(base5["MODUPB_CHNG"], errors="coerce").fillna(0)
    base5["CSCORE_MN"] = np.where(
        base5["CSCORE_C"].notna() & (base5["CSCORE_C"] < base5["CSCORE_B"]),
        base5["CSCORE_C"],
        base5["CSCORE_B"],
    )
    base5["CSCORE_MN"] = pd.Series(base5["CSCORE_MN"], index=base5.index).fillna(base5["CSCORE_B"]).fillna(base5["CSCORE_C"])
    base5["ORIG_VAL"] = (base5["orig_amt"] / (base5["oltv"] / 100)).round(2)
    last_dlq = base5["dlq_status"].replace({"X": "999", "XX": "999"})
    base5["z_last_status"] = pd.to_numeric(last_dlq, errors="coerce")

    conditions = [
        base5["zb_code"].eq("09"), base5["zb_code"].eq("03"), base5["zb_code"].eq("02"),
        base5["zb_code"].eq("06"), base5["zb_code"].eq("15"), base5["zb_code"].eq("16"),
        base5["zb_code"].eq("01"), (base5["z_last_status"] < 999) & (base5["z_last_status"] >= 9),
        base5["z_last_status"].eq(8), base5["z_last_status"].eq(7), base5["z_last_status"].eq(6),
        base5["z_last_status"].eq(5), base5["z_last_status"].eq(4), base5["z_last_status"].eq(3),
        base5["z_last_status"].eq(2), base5["z_last_status"].eq(1), base5["z_last_status"].eq(0),
    ]
    choices = ["F", "S", "T", "R", "N", "L", "P", "9", "8", "7", "6", "5", "4", "3", "2", "1", "C"]
    base5["LAST_STAT"] = np.select(conditions, choices, default="X")
    foreclosure_status = base5["LAST_STAT"].isin(["F", "S", "N", "T"])
    base5["FCC_DTE"] = np.where(base5["FCC_DTE"].fillna("").eq("") & foreclosure_status, base5["zb_date"], base5["FCC_DTE"])
    base5["COMPLT_FLG"] = np.where(base5["DISP_DTE"].fillna("") != "", 1.0, 0.0)
    base5["COMPLT_FLG"] = np.where(foreclosure_status, base5["COMPLT_FLG"], np.nan)

    months_last_lpi = month_number(base5["LAST_DTE"]) - month_number(base5["LPI_DTE"])
    base5["INT_COST"] = np.where(
        (base5["COMPLT_FLG"] == 1) & (base5["LPI_DTE"].fillna("") != ""),
        months_last_lpi * (((base5["LAST_RT"] / 100) - 0.0035) / 12) * (base5["LAST_UPB"] + (-1 * base5["NON_INT_UPB"])),
        np.nan,
    )
    base5["INT_COST"] = base5["INT_COST"].round(2)
    base5.loc[(base5["COMPLT_FLG"] == 1) & base5["INT_COST"].isna(), "INT_COST"] = 0
    for col in ["FCC_COST", "PP_COST", "AR_COST", "IE_COST", "TAX_COST", "PFG_COST", "CE_PROCS", "NS_PROCS", "RMW_PROCS", "O_PROCS"]:
        base5.loc[(base5["COMPLT_FLG"] == 1) & base5[col].isna(), col] = 0
    base5["NET_LOSS"] = np.where(
        base5["COMPLT_FLG"] == 1,
        base5["LAST_UPB"] + base5["FCC_COST"] + base5["PP_COST"] + base5["AR_COST"] + base5["IE_COST"]
        + base5["TAX_COST"] + base5["PFG_COST"] + base5["INT_COST"] - base5["NS_PROCS"]
        - base5["CE_PROCS"] - base5["RMW_PROCS"] - base5["O_PROCS"],
        np.nan,
    )
    base5["NET_LOSS"] = base5["NET_LOSS"].round(2)
    base5["NET_SEV"] = np.where(base5["COMPLT_FLG"] == 1, base5["NET_LOSS"] / base5["LAST_UPB"], np.nan)
    base5["NET_SEV"] = base5["NET_SEV"].round(6)

    modir_source = base1.merge(performance_file, on="LOAN_ID", how="left")
    modir_source = modir_source[modir_source["mod_ind"].eq("Y")].copy()
    if modir_source.empty:
        modir = pd.DataFrame(columns=["LOAN_ID", "MODIR_COST", "MODFB_COST", "MODTOT_COST"])
    else:
        modir_source["non_int_upb"] = modir_source["non_int_upb"].fillna(0)
        modir_source["modir_cost"] = (((modir_source["orig_rt"] - modir_source["curr_rte"]) / 1200) * modir_source["act_upb"]).round(2)
        modir_source["modfb_cost"] = np.where(
            modir_source["non_int_upb"] > 0,
            (modir_source["curr_rte"] / 1200) * modir_source["non_int_upb"],
            0,
        )
        modir_source["modfb_cost"] = modir_source["modfb_cost"].round(2)
        modir = modir_source.groupby("LOAN_ID", as_index=False).agg(
            MODIR_COST=("modir_cost", "sum"), MODFB_COST=("modfb_cost", "sum")
        )
        modir["MODTOT_COST"] = (modir["MODFB_COST"] + modir["MODIR_COST"]).round(2)

    base6 = base5.merge(modir, on="LOAN_ID", how="left")
    for col in ["MODIR_COST", "MODFB_COST", "MODTOT_COST"]:
        base6[col] = pd.to_numeric(base6[col], errors="coerce")
    completed = base6["COMPLT_FLG"].eq(1)
    base6.loc[completed & base6["non_int_upb"].isna(), "non_int_upb"] = 0
    duration = month_number(base6["LAST_DTE"]) - month_number(base6["zb_date"])
    base6["MODIR_COST"] = np.where(
        completed,
        base6["MODIR_COST"] + duration * ((base6["orig_rt"] - base6["LAST_RT"]) / 1200) * base6["LAST_UPB"],
        base6["MODIR_COST"],
    )
    base6["MODIR_COST"] = base6["MODIR_COST"].round(2)
    base6["MODFB_COST"] = np.where(
        completed,
        base6["MODFB_COST"] + duration * (base6["LAST_RT"] / 1200) * base6["non_int_upb"],
        base6["MODFB_COST"],
    )
    base6["MODFB_COST"] = base6["MODFB_COST"].round(2)
    base6["orig_rt"] = pd.to_numeric(base6["orig_rt"], errors="coerce").round(3)

    base7 = add_missing_columns(base6, FINAL_COLUMNS)[FINAL_COLUMNS].copy()
    return base7


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


def default_output_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path.with_name(f"{path.stem}_stat.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LPPUB one-row-per-loan stat file.")
    parser.add_argument("--input", default="2024Q1_200lines.csv", help="Raw pipe-delimited LPPUB input file.")
    parser.add_argument("--output", default=None, help="Output CSV path. Defaults to <input_stem>_stat.csv.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional number of raw rows to read.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output) if args.output else default_output_path(args.input)
    stat_file = build_stat_file(args.input, nrows=args.nrows)
    formatted = format_for_output(stat_file)
    formatted.to_csv(output, index=False, na_rep="NULL", quoting=1)
    print(f"Wrote {len(formatted)} loan rows and {len(formatted.columns)} columns to {output}")


if __name__ == "__main__":
    main()
