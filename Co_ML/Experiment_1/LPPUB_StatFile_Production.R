####################################################################
# This code is meant to be called from LPPUB_StatFile.R
# Here the Loan Performance data is modified into a one-loan-per-row dataset including key analytic data fields.
# We encourage exploration of this code to understand how certain fields in the statistical summary are derived.
####################################################################

### Define LPPUB table headers
lppub_column_names <- c("POOL_ID", "LOAN_ID", "ACT_PERIOD", "CHANNEL", "SELLER", "SERVICER",
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
                        "OTHER_FORECLOSURE_PROCEEDS", "NON_INTEREST_BEARING_UPB", "PRINCIPAL_FORGIVENESS_AMOUNT",
                        "ORIGINAL_LIST_START_DATE", "ORIGINAL_LIST_PRICE", "CURRENT_LIST_START_DATE",
                        "CURRENT_LIST_PRICE", "ISSUE_SCOREB", "ISSUE_SCOREC", "CURR_SCOREB",
                        "CURR_SCOREC", "MI_TYPE", "SERV_IND", "CURRENT_PERIOD_MODIFICATION_LOSS_AMOUNT",
                        "CUMULATIVE_MODIFICATION_LOSS_AMOUNT", "CURRENT_PERIOD_CREDIT_EVENT_NET_GAIN_OR_LOSS",
                        "CUMULATIVE_CREDIT_EVENT_NET_GAIN_OR_LOSS", "HOMEREADY_PROGRAM_INDICATOR",
                        "FORECLOSURE_PRINCIPAL_WRITE_OFF_AMOUNT", "RELOCATION_MORTGAGE_INDICATOR",
                        "ZERO_BALANCE_CODE_CHANGE_DATE", "LOAN_HOLDBACK_INDICATOR", "LOAN_HOLDBACK_EFFECTIVE_DATE",
                        "DELINQUENT_ACCRUED_INTEREST", "PROPERTY_INSPECTION_WAIVER_INDICATOR",
                        "HIGH_BALANCE_LOAN_INDICATOR", "ARM_5_YR_INDICATOR", "ARM_PRODUCT_TYPE",
                        "MONTHS_UNTIL_FIRST_PAYMENT_RESET", "MONTHS_BETWEEN_SUBSEQUENT_PAYMENT_RESET",
                        "INTEREST_RATE_CHANGE_DATE", "PAYMENT_CHANGE_DATE", "ARM_INDEX",
                        "ARM_CAP_STRUCTURE", "INITIAL_INTEREST_RATE_CAP", "PERIODIC_INTEREST_RATE_CAP",
                        "LIFETIME_INTEREST_RATE_CAP", "MARGIN", "BALLOON_INDICATOR",
                        "PLAN_NUMBER", "FORBEARANCE_INDICATOR", "HIGH_LOAN_TO_VALUE_HLTV_REFINANCE_OPTION_INDICATOR",
                        "DEAL_NAME", "RE_PROCS_FLAG", "ADR_TYPE", "ADR_COUNT", "ADR_UPB", "PAYMENT_DEFERRAL_MOD_EVENT_FLAG", "INTEREST_BEARING_UPB")
lppub_column_classes <- c("character", "character", "character", "character", "character", "character",
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
                          "character", "numeric", "numeric", "character", "numeric")

### Load Loan Performance file
lppub_file <- load_lppub_file(FileName, lppub_column_names, lppub_column_classes)

### Ensure interest rate columns are treated as numeric
lppub_file$ORIG_RATE <- as.numeric(lppub_file$ORIG_RATE)
lppub_file$CURR_RATE <- as.numeric(lppub_file$CURR_RATE)

### Select and rename key columns for statistical summary analysis
lppub_base <- lppub_file %>%
  select(
    LOAN_ID,
    ACT_PERIOD,
    CHANNEL,
    SELLER,
    SERVICER,
    ORIG_RATE,
    CURR_RATE,
    ORIG_UPB,
    CURRENT_UPB,
    ORIG_TERM,
    ORIG_DATE,
    FIRST_PAY,
    LOAN_AGE,
    REM_MONTHS,
    ADJ_REM_MONTHS,
    MATR_DT,
    OLTV,
    OCLTV,
    NUM_BO,
    DTI,
    CSCORE_B,
    CSCORE_C,
    FIRST_FLAG,
    PURPOSE,
    PROP,
    NO_UNITS,
    OCC_STAT,
    STATE,
    MSA,
    ZIP,
    MI_PCT,
    PRODUCT,
    DLQ_STATUS,
    MOD_FLAG,
    Zero_Bal_Code,
    ZB_DTE,
    LAST_PAID_INSTALLMENT_DATE,
    FORECLOSURE_DATE,
    DISPOSITION_DATE,
    FORECLOSURE_COSTS,
    PROPERTY_PRESERVATION_AND_REPAIR_COSTS,
    ASSET_RECOVERY_COSTS,
    MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS,
    ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY,
    NET_SALES_PROCEEDS,
    CREDIT_ENHANCEMENT_PROCEEDS,
    REPURCHASES_MAKE_WHOLE_PROCEEDS,
    OTHER_FORECLOSURE_PROCEEDS,
    NON_INTEREST_BEARING_UPB,
    PRINCIPAL_FORGIVENESS_AMOUNT,
    RELOCATION_MORTGAGE_INDICATOR,
    MI_TYPE,
    SERV_IND,
    RPRCH_DTE,
	LAST_UPB
  ) %>%
  mutate(
    repch_flag = if_else(is.na(RPRCH_DTE) == FALSE, 1, 0),
    ACT_PERIOD = paste(substr(ACT_PERIOD, 3, 6), substr(ACT_PERIOD, 1, 2), '01', sep = "-"),
    FIRST_PAY = paste(substr(FIRST_PAY, 3, 6), substr(FIRST_PAY, 1, 2), '01', sep = "-"),
    ORIG_DATE = paste(substr(ORIG_DATE, 3, 6), substr(ORIG_DATE, 1, 2), '01', sep = "-")
  ) %>%
  arrange(LOAN_ID, ACT_PERIOD)

### We remove dataframes after they are no longer needed in order to preserve memory during the process
rm(lppub_file)

### Split the data into static "Acquisition" variables and dynamic "Performance" variables
acquisitionFile <- lppub_base %>%
	select(LOAN_ID, ACT_PERIOD, CHANNEL, SELLER, ORIG_RATE, ORIG_UPB,
		   ORIG_TERM, ORIG_DATE, FIRST_PAY, OLTV,
		   OCLTV, NUM_BO, DTI, CSCORE_B, CSCORE_C,
		   FIRST_FLAG, PURPOSE, PROP, NO_UNITS, OCC_STAT,
		   STATE, ZIP, MI_PCT, PRODUCT, MI_TYPE,
		   RELOCATION_MORTGAGE_INDICATOR) %>%
	rename(
		ORIG_CHN = CHANNEL,
		orig_rt = ORIG_RATE,
		orig_amt = ORIG_UPB,
		orig_trm = ORIG_TERM,
		orig_date = ORIG_DATE,
		first_pay = FIRST_PAY,
		oltv = OLTV,
		ocltv = OCLTV,
		num_bo = NUM_BO,
		dti = DTI,
		FTHB_FLG = FIRST_FLAG,
		purpose = PURPOSE,
		PROP_TYP = PROP,
		NUM_UNIT = NO_UNITS,
		occ_stat = OCC_STAT,
		state = STATE,
		zip_3 = ZIP,
		mi_pct = MI_PCT,
		prod_type = PRODUCT,
		relo_flg = RELOCATION_MORTGAGE_INDICATOR
	)

acqFirstPeriod <- acquisitionFile %>%
  group_by(LOAN_ID) %>%
  summarize(first_period = max(ACT_PERIOD)) %>%
  left_join(acquisitionFile, by = c("LOAN_ID" = "LOAN_ID", "first_period" = "ACT_PERIOD")) %>%
  select(
    LOAN_ID, ORIG_CHN, SELLER, orig_rt, orig_amt,
    orig_trm, orig_date, first_pay, oltv,
    ocltv, num_bo, dti, CSCORE_B, CSCORE_C,
    FTHB_FLG, purpose, PROP_TYP, NUM_UNIT, occ_stat,
    state, zip_3, mi_pct, prod_type, MI_TYPE,
    relo_flg
  )

acquisitionFile <- acqFirstPeriod
rm(acqFirstPeriod)

### Prepare the Performance variables
performanceFile <- lppub_base %>%
	select(LOAN_ID, ACT_PERIOD, SERVICER, CURR_RATE, CURRENT_UPB,
		   LOAN_AGE, REM_MONTHS, ADJ_REM_MONTHS, MATR_DT, MSA,
		   DLQ_STATUS, MOD_FLAG, Zero_Bal_Code, ZB_DTE, LAST_PAID_INSTALLMENT_DATE,
		   FORECLOSURE_DATE, DISPOSITION_DATE, FORECLOSURE_COSTS, PROPERTY_PRESERVATION_AND_REPAIR_COSTS, ASSET_RECOVERY_COSTS,
		   MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS, ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY, NET_SALES_PROCEEDS, CREDIT_ENHANCEMENT_PROCEEDS, REPURCHASES_MAKE_WHOLE_PROCEEDS,
		   OTHER_FORECLOSURE_PROCEEDS, NON_INTEREST_BEARING_UPB, PRINCIPAL_FORGIVENESS_AMOUNT, repch_flag, LAST_UPB) %>%
	rename(
	period = ACT_PERIOD,
	servicer = SERVICER,
	curr_rte = CURR_RATE,
	act_upb = CURRENT_UPB,
	loan_age = LOAN_AGE,
	rem_mths = REM_MONTHS,
	adj_rem_months = ADJ_REM_MONTHS,
	maturity_date = MATR_DT,
	msa = MSA,
	dlq_status =DLQ_STATUS,
	mod_ind = MOD_FLAG,
	z_zb_code = Zero_Bal_Code,
	zb_date = ZB_DTE,
	lpi_dte = LAST_PAID_INSTALLMENT_DATE,
	fcc_dte = FORECLOSURE_DATE,
	disp_dte = DISPOSITION_DATE,
	FCC_COST = FORECLOSURE_COSTS,
	PP_COST = PROPERTY_PRESERVATION_AND_REPAIR_COSTS,
	AR_COST = ASSET_RECOVERY_COSTS,
	IE_COST = MISCELLANEOUS_HOLDING_EXPENSES_AND_CREDITS,
	TAX_COST = ASSOCIATED_TAXES_FOR_HOLDING_PROPERTY,
	NS_PROCS = NET_SALES_PROCEEDS,
	CE_PROCS = CREDIT_ENHANCEMENT_PROCEEDS,
	RMW_PROCS = REPURCHASES_MAKE_WHOLE_PROCEEDS,
	O_PROCS = OTHER_FORECLOSURE_PROCEEDS,
	non_int_upb = NON_INTEREST_BEARING_UPB,
	prin_forg_upb = PRINCIPAL_FORGIVENESS_AMOUNT,
	zb_upb = LAST_UPB
	)

rm(lppub_base)

### Create AQSN_DTE field from filename
acquisition_year <- substr(FileName, 1, 4)
acquisition_qtr <- substr(FileName, 5, 7)
if(acquisition_qtr == 'Q1'){
  acquisition_month <- '03'
} else if(acquisition_qtr == 'Q2'){
  acquisition_month <- '06'
} else if(acquisition_qtr == 'Q3'){
  acquisition_month <- '09'
} else {
  acquisition_month <- '12'
}
acquisition_date <- paste(acquisition_year, acquisition_month, '01', sep = "-")

### Convert all date fields to YYYY-MM-DD format
acquisitionFile <- acquisitionFile %>%
  rename(
    ORIG_DTE = orig_date,
    FRST_DTE = first_pay
  )

performanceFile <- performanceFile %>%
  mutate(
    maturity_date = if_else(maturity_date != '', paste(substr(maturity_date, 3, 6), substr(maturity_date, 1, 2), '01', sep = '-'), maturity_date),
    zb_date = if_else(zb_date != '', paste(substr(zb_date, 3, 6), substr(zb_date, 1, 2), '01', sep = '-'), zb_date),
    lpi_dte = if_else(lpi_dte != '', paste(substr(lpi_dte, 3, 6), substr(lpi_dte, 1, 2), '01', sep = '-'), lpi_dte),
    fcc_dte = if_else(fcc_dte != '', paste(substr(fcc_dte, 3, 6), substr(fcc_dte, 1, 2), '01', sep = '-'), fcc_dte),
    disp_dte = if_else(disp_dte != '', paste(substr(disp_dte, 3, 6), substr(disp_dte, 1, 2), '01', sep = '-'), disp_dte)
  )

### We will sequentially create several "base" tables that form the building blocks of our final analytic dataset
### Examining the different base tables can be helpful for modifying and debugging this code
### Create first base table with a copy of acquisition fields plus AQSN_DTE field and recodes of MI_TYPE and OCLTV
baseTable1 <- acquisitionFile %>%
	mutate(
		AQSN_DTE = acquisition_date,
		MI_TYPE = case_when(
		  MI_TYPE == '1' ~ 'BPMI', #MI_TYPE is recoded to be more descriptive
		  MI_TYPE == '2' ~ 'LPMI',
		  MI_TYPE == '3' ~ 'IPMI',
		  TRUE ~ 'None'
		),
		ocltv = if_else(is.na(ocltv), oltv, ocltv) #If OCLTV is missing, we replace it with OLTV
	)

### Create the second base table with the latest-available or aggregated data from the Performance fields
last_act_dte_table <- performanceFile %>%
	group_by(LOAN_ID) %>%
	summarize(LAST_ACTIVITY_DATE = max(period))

last_upb_table <- performanceFile %>% 
  group_by(LOAN_ID) %>% 
  filter(period == max(period)) %>% 
  summarize(LAST_UPB = ifelse(!is.na(zb_upb),zb_upb,act_upb)) 

last_rt_table <- performanceFile %>%
  filter(!is.na(curr_rte)) %>%
  group_by(LOAN_ID) %>%
  summarize(
    LAST_RT_DATE = max(period)
  ) %>%
  left_join(performanceFile, by = c("LOAN_ID" = "LOAN_ID", "LAST_RT_DATE" = "period")) %>%
  select(LOAN_ID, curr_rte) %>%
  rename(LAST_RT = curr_rte) %>%
  mutate(LAST_RT = round(LAST_RT, 3))

zb_code_table <- performanceFile %>%
  filter(z_zb_code != '') %>%
  group_by(LOAN_ID) %>%
  summarize(zb_code_dt = max(period)) %>%
  left_join(performanceFile, by = c("LOAN_ID" = "LOAN_ID", "zb_code_dt" = "period")) %>%
  select(LOAN_ID, z_zb_code) %>%
  rename(zb_code = z_zb_code)

max_table <- last_act_dte_table %>%
	left_join(performanceFile, by = c("LOAN_ID" = "LOAN_ID", "LAST_ACTIVITY_DATE" = "period")) %>%
  left_join(last_upb_table, by = c("LOAN_ID" = "LOAN_ID")) %>%
  left_join(last_rt_table, by = c("LOAN_ID" = "LOAN_ID")) %>%
  left_join(zb_code_table, by = c("LOAN_ID" = "LOAN_ID"))

rm(last_act_dte_table)
rm(last_rt_table)
rm(last_upb_table)
rm(zb_code_table)

servicer_table <- performanceFile %>%
  filter(servicer != '') %>%
  group_by(LOAN_ID) %>%
  summarize(servicer_period = max(period)) %>%
  left_join(performanceFile, by = c("LOAN_ID" = "LOAN_ID", "servicer_period" = "period")) %>%
  mutate(SERVICER = servicer) %>%
  select(LOAN_ID, SERVICER)

non_int_upb_table <- performanceFile %>%
  group_by(LOAN_ID) %>%
  slice(n()-1) %>% 
  select(LOAN_ID, NON_INT_UPB = non_int_upb) %>% 
  mutate(NON_INT_UPB = if_else(is.na(NON_INT_UPB), 0, NON_INT_UPB))


baseTable2 <- baseTable1 %>%
	left_join(max_table, by = "LOAN_ID") %>%
  left_join(servicer_table, by = 'LOAN_ID') %>%
  left_join(non_int_upb_table, by = 'LOAN_ID')

rm(max_table)
rm(servicer_table)
rm(non_int_upb_table)

### Create the third base table with the latest-available forclosure/disposition data
fcc_table <- performanceFile %>%
	filter(!is.na(lpi_dte) & !is.na(fcc_dte) & !is.na(disp_dte))

fcc_table <- fcc_table %>%
	group_by(LOAN_ID) %>%
	summarize(
		LPI_DTE = max(lpi_dte),
		FCC_DTE = max(fcc_dte),
		DISP_DTE = max(disp_dte)
	)

baseTable3 <- baseTable2 %>%
	left_join(fcc_table, by = "LOAN_ID")

rm(baseTable2)
rm(fcc_table)

### Create the series of "first DQ occurence" tables and loan modification tables
slimPerformanceFile <- performanceFile %>%
	select(LOAN_ID, period, dlq_status, z_zb_code, act_upb, zb_upb, mod_ind, maturity_date, rem_mths) %>%
	mutate(
		dlq_status = if_else(dlq_status == 'XX', '999', dlq_status),
		dlq_status = as.numeric(dlq_status)
	)

f30_table <- slimPerformanceFile %>%
	filter(dlq_status >= 1 & dlq_status < 999, z_zb_code == '') %>%
	group_by(LOAN_ID) %>%
	summarize(F30_DTE = min(period)) %>%
	left_join(slimPerformanceFile, by = c("LOAN_ID" = "LOAN_ID", "F30_DTE" = "period")) %>%
	select(LOAN_ID, F30_DTE, act_upb) %>%
	rename(F30_UPB = act_upb)

f60_table <- slimPerformanceFile %>%
	filter(dlq_status >= 2 & dlq_status < 999, z_zb_code == '') %>%
	group_by(LOAN_ID) %>%
	summarize(F60_DTE = min(period)) %>%
	left_join(slimPerformanceFile,  by = c("LOAN_ID" = "LOAN_ID", "F60_DTE" = "period")) %>%
	select(LOAN_ID, F60_DTE, act_upb) %>%
	rename(F60_UPB = act_upb)

f90_table <- slimPerformanceFile %>%
	filter(dlq_status >= 3 & dlq_status < 999, z_zb_code == '') %>%
	group_by(LOAN_ID) %>%
	summarize(F90_DTE = min(period)) %>%
	left_join(slimPerformanceFile, by = c("LOAN_ID" = "LOAN_ID", "F90_DTE" = "period")) %>%
	select(LOAN_ID, F90_DTE, act_upb) %>%
	rename(F90_UPB = act_upb)

f120_table <- slimPerformanceFile %>%
	filter(dlq_status >= 4 & dlq_status < 999, z_zb_code == '') %>%
	group_by(LOAN_ID) %>%
	summarize(F120_DTE = min(period)) %>%
	left_join(slimPerformanceFile, by = c("LOAN_ID" = "LOAN_ID", "F120_DTE" = "period")) %>%
	select(LOAN_ID, F120_DTE, act_upb) %>%
	rename(F120_UPB = act_upb)

f180_table <- slimPerformanceFile %>%
  filter(dlq_status >= 6 & dlq_status < 999, z_zb_code == '') %>%
  group_by(LOAN_ID) %>%
  summarize(F180_DTE = min(period)) %>%
  left_join(slimPerformanceFile, by = c("LOAN_ID" = "LOAN_ID", "F180_DTE" = "period")) %>%
  select(LOAN_ID, F180_DTE, act_upb) %>%
  rename(F180_UPB = act_upb)

fce_table <- slimPerformanceFile %>%
	filter((z_zb_code == '02' | z_zb_code == '03' | z_zb_code == '09' | z_zb_code == '15') | (dlq_status >= 6 & dlq_status < 999)) %>%
	group_by(LOAN_ID) %>%
	summarize(FCE_DTE = min(period)) %>%
	left_join(slimPerformanceFile, by = c("LOAN_ID" = "LOAN_ID", "FCE_DTE" = "period")) %>%
	select(LOAN_ID, FCE_DTE, act_upb, zb_upb) %>%
	mutate(FCE_UPB = zb_upb + act_upb) %>%
	select(LOAN_ID, FCE_DTE, FCE_UPB)

fmod_dte_table <- slimPerformanceFile %>%
	filter(mod_ind == 'Y', z_zb_code == '') %>%
	group_by(LOAN_ID) %>%
	summarize(FMOD_DTE = min(period))

fmod_table <- slimPerformanceFile %>%
  filter(mod_ind == 'Y', z_zb_code == '') %>%
	left_join(fmod_dte_table, by = c("LOAN_ID" = "LOAN_ID")) %>%
  filter((((as.numeric(substr(period,1,4))*12) + as.numeric(substr(period,6,7)))  <=  ((as.numeric(substr(FMOD_DTE,1,4))*12) + as.numeric(substr(FMOD_DTE,6,7))) + 3)) %>%
  group_by(LOAN_ID) %>%
  summarize(FMOD_UPB = max(act_upb)) %>%
  left_join(fmod_dte_table, by = "LOAN_ID") %>%
  left_join(slimPerformanceFile, by = c("LOAN_ID" = "LOAN_ID", "FMOD_DTE" = "period")) %>%
	select(LOAN_ID, FMOD_DTE, FMOD_UPB, maturity_date)

rm(fmod_dte_table)

### Compute NUM_PERIODS_120 field (the number of periods elapsed from origination to a loan becoming at least 120 days delinquent)
num_120_table <- f120_table %>%
	left_join(acquisitionFile, by = 'LOAN_ID') %>%
	mutate(
	  z_num_periods_120 = (((as.numeric(substr(F120_DTE,1,4))*12) + as.numeric(substr(F120_DTE,6,7))) - ((as.numeric(substr(FRST_DTE,1,4))*12) + as.numeric(substr(FRST_DTE,6,7))) + 1)
	) %>%
  select(LOAN_ID, z_num_periods_120)

rm(acquisitionFile)

### Compute MODTRM_CHNG field (a flag for whether the term of a loan was changed as part of a loan modification)
orig_maturity_table <- slimPerformanceFile %>%
  filter(!is.na(maturity_date)) %>%
	group_by(LOAN_ID) %>%
	summarize(maturity_date_period = min(period)) %>%
  left_join(slimPerformanceFile, by = c("LOAN_ID" = "LOAN_ID", "maturity_date_period" = "period")) %>%
  select(LOAN_ID, maturity_date) %>%
  rename(orig_maturity_date = maturity_date)

trm_chng_table <- slimPerformanceFile %>%
  group_by(LOAN_ID) %>%
  mutate(
    prev_rem_mths = lag(rem_mths, n = 1, order_by = period),
    trm_chng = rem_mths - prev_rem_mths,
    did_trm_chng = if_else(trm_chng >= 0, 1, 0)
  ) %>%
  filter(did_trm_chng == 1) %>%
  group_by(LOAN_ID) %>%
  summarize(trm_chng_dt = min(period))

modtrm_table <- fmod_table %>%
	left_join(orig_maturity_table, by = 'LOAN_ID') %>%
  left_join(trm_chng_table, by = 'LOAN_ID') %>%
	mutate(MODTRM_CHNG = if_else(maturity_date != orig_maturity_date | !is.na(trm_chng_dt), 1, 0)) %>%
	select(LOAN_ID, MODTRM_CHNG)

rm(orig_maturity_table)
rm(trm_chng_table)

### Compute MODTRM_UPB field (a flag for whether the balance of a loan was changed as part of a loan modification)
pre_mod_upb_table <- slimPerformanceFile %>%
	left_join(fmod_table, by = 'LOAN_ID') %>%
	filter(period < FMOD_DTE) %>%
	group_by(LOAN_ID) %>%
	summarize(pre_mod_period = max(period)) %>%
	left_join(slimPerformanceFile, by = c("LOAN_ID" = "LOAN_ID", "pre_mod_period" = "period")) %>%
	rename(pre_mod_upb = act_upb)

rm(slimPerformanceFile)

modupb_table <- fmod_table %>%
	left_join(pre_mod_upb_table, by = 'LOAN_ID') %>%
	mutate(MODUPB_CHNG = if_else(FMOD_UPB >= pre_mod_upb, 1, 0)) %>%
	select(LOAN_ID, MODUPB_CHNG)

rm(pre_mod_upb_table)

### Create the fourth base table by joining the first-DQ-occurence and loan modification tables
baseTable4 <- baseTable3 %>%
	left_join(f30_table, by = 'LOAN_ID') %>%
	left_join(f60_table, by = 'LOAN_ID') %>%
	left_join(f90_table, by = 'LOAN_ID') %>%
	left_join(f120_table, by = 'LOAN_ID') %>%
	left_join(f180_table, by = 'LOAN_ID') %>%
	left_join(fce_table, by = 'LOAN_ID') %>%
	left_join(fmod_table, by = 'LOAN_ID') %>%
	left_join(num_120_table, by = 'LOAN_ID') %>%
	left_join(modtrm_table, by = 'LOAN_ID') %>%
	left_join(modupb_table, by = 'LOAN_ID') %>%
  mutate(
    F30_UPB = if_else(is.na(F30_UPB) & !is.na(F30_DTE), orig_amt, F30_UPB),
    F60_UPB = if_else(is.na(F60_UPB) & !is.na(F60_DTE), orig_amt, F60_UPB),
    F90_UPB = if_else(is.na(F90_UPB) & !is.na(F90_DTE), orig_amt, F90_UPB),
    F120_UPB = if_else(is.na(F120_UPB) & !is.na(F120_DTE), orig_amt, F120_UPB),
    F180_UPB = if_else(is.na(F180_UPB) & !is.na(F180_DTE), orig_amt, F180_UPB),
    FCE_UPB = if_else(is.na(FCE_UPB) & !is.na(FCE_DTE), orig_amt, FCE_UPB)
  )

rm(baseTable3)
rm(f30_table)
rm(f60_table)
rm(f90_table)
rm(f120_table)
rm(f180_table)
rm(fce_table)
rm(fmod_table)
rm(num_120_table)
rm(modtrm_table)
rm(modupb_table)

### Create the fifth base table by computing loan status fields according to specific rules
baseTable5 <- baseTable4 %>%
  mutate(
    LAST_DTE = if_else(disp_dte != '', disp_dte, LAST_ACTIVITY_DATE),
    repch_flag = if_else(repch_flag == 'Y', 1, 0),
    PFG_COST = prin_forg_upb,
    MOD_FLAG = if_else(!is.na(FMOD_DTE), 1, 0),
    MODFG_COST = if_else(mod_ind == 'Y', 0, NA_real_),
    MODFG_COST = if_else(mod_ind == 'Y' & PFG_COST > 0, PFG_COST, 0),
    MODTRM_CHNG = if_else(is.na(MODTRM_CHNG), 0, MODTRM_CHNG),
    MODUPB_CHNG = if_else(is.na(MODUPB_CHNG), 0, MODUPB_CHNG),
    CSCORE_MN = if_else(!is.na(CSCORE_C) & CSCORE_C < CSCORE_B, CSCORE_C, CSCORE_B),
    CSCORE_MN = if_else(is.na(CSCORE_MN), CSCORE_B, CSCORE_MN),
    CSCORE_MN = if_else(is.na(CSCORE_MN), CSCORE_C, CSCORE_MN),
    ORIG_VAL = round(orig_amt/(oltv/100), digits = 2),
    dlq_status = if_else(dlq_status == 'X' | dlq_status == 'XX', '999', dlq_status),
    z_last_status = as.numeric(dlq_status),
    LAST_STAT = case_when(
      zb_code == '09' ~ 'F',
      zb_code == '03' ~ 'S',
      zb_code == '02' ~ 'T',
      zb_code == '06' ~ 'R',
      zb_code == '15' ~ 'N',
      zb_code == '16' ~ 'L',
      zb_code == '01' ~ 'P',
      z_last_status < 999 & z_last_status >= 9 ~ '9',
      z_last_status == 8 ~ '8',
      z_last_status == 7 ~ '7',
      z_last_status == 6 ~ '6',
      z_last_status == 5 ~ '5',
      z_last_status == 4 ~ '4',
      z_last_status == 3 ~ '3',
      z_last_status == 2 ~ '2',
      z_last_status == 1 ~ '1',
      z_last_status == 0 ~ 'C',
      TRUE ~ 'X'
    ),
    FCC_DTE = if_else(FCC_DTE == '' & (LAST_STAT == 'F' | LAST_STAT == 'S' | LAST_STAT == 'N' | LAST_STAT == 'T'), zb_date, FCC_DTE),

    COMPLT_FLG = if_else(DISP_DTE != '', 1, 0),
    COMPLT_FLG = if_else(LAST_STAT != 'F' & LAST_STAT != 'S' & LAST_STAT != 'N' & LAST_STAT != 'T', NA_real_, COMPLT_FLG),
    INT_COST = round(if_else(COMPLT_FLG == 1 & LPI_DTE != '', ((((as.numeric(substr(LAST_DTE,1,4))*12) + as.numeric(substr(LAST_DTE,6,7))) - ((as.numeric(substr(LPI_DTE,1,4))*12) + as.numeric(substr(LPI_DTE,6,7)))) * (((LAST_RT / 100) - 0.0035) / 12) * (LAST_UPB + (-1 * NON_INT_UPB))), NA_real_), digits = 2),

    INT_COST = if_else(COMPLT_FLG == 1 & is.na(INT_COST), 0, INT_COST),
    FCC_COST = if_else(COMPLT_FLG == 1 & is.na(FCC_COST), 0, FCC_COST),
    PP_COST = if_else(COMPLT_FLG == 1 & is.na(PP_COST), 0, PP_COST),
    AR_COST = if_else(COMPLT_FLG == 1 & is.na(AR_COST), 0, AR_COST),
    IE_COST = if_else(COMPLT_FLG == 1 & is.na(IE_COST), 0, IE_COST),
    TAX_COST = if_else(COMPLT_FLG == 1 & is.na(TAX_COST), 0, TAX_COST),
    PFG_COST = if_else(COMPLT_FLG == 1 & is.na(PFG_COST), 0, PFG_COST),
    CE_PROCS = if_else(COMPLT_FLG == 1 & is.na(CE_PROCS), 0, CE_PROCS),
    NS_PROCS = if_else(COMPLT_FLG == 1 & is.na(NS_PROCS), 0, NS_PROCS),
    RMW_PROCS = if_else(COMPLT_FLG == 1 & is.na(RMW_PROCS), 0, RMW_PROCS),
    O_PROCS = if_else(COMPLT_FLG == 1 & is.na(O_PROCS), 0, O_PROCS),

    NET_LOSS = round(if_else(COMPLT_FLG == 1, (LAST_UPB + FCC_COST + PP_COST + AR_COST + IE_COST + TAX_COST + PFG_COST + INT_COST + -1*NS_PROCS + -1*CE_PROCS + -1*RMW_PROCS + -1*O_PROCS), NA_real_), digits = 2),
    NET_SEV = round(if_else(COMPLT_FLG == 1, (NET_LOSS / LAST_UPB), NA_real_), digits = 6),
  )

rm(baseTable4)

### Compute MODIR_COST, MODFB_COST and other loan modification fields
modir_table <- baseTable1 %>%
  left_join(performanceFile, by = 'LOAN_ID') %>%
  filter(mod_ind == 'Y') %>%
  mutate(
    non_int_upb = if_else(is.na(non_int_upb), 0, non_int_upb),
    modir_cost = round(if_else(mod_ind == 'Y', (((orig_rt - curr_rte) / 1200) * act_upb), 0), digits = 2),
    modfb_cost = round(if_else(mod_ind == 'Y' & non_int_upb > 0, (curr_rte / 1200) * non_int_upb, 0), digits = 2)
  ) %>%
  group_by(LOAN_ID) %>%
  summarize(
    MODIR_COST = sum(modir_cost),
    MODFB_COST = sum(modfb_cost)
  ) %>%
  mutate(
    MODTOT_COST = round(MODFB_COST + MODIR_COST, 2)
  )

rm(performanceFile)

### Create the sixth base table by joining the remaining loan modifcation fields to the rest of our data
baseTable6 <- baseTable5 %>%
  left_join(modir_table, by = 'LOAN_ID') %>%
  mutate(
    COMPLT_FLG = as.character(COMPLT_FLG),
    COMPLT_FLG = if_else(is.na(COMPLT_FLG), '', COMPLT_FLG),
    non_int_upb =if_else(COMPLT_FLG == '1' & is.na(non_int_upb), 0, non_int_upb),
    MODIR_COST = round(if_else(COMPLT_FLG == '1', MODIR_COST + ((((as.numeric(substr(LAST_DTE,1,4))*12) + as.numeric(substr(LAST_DTE,6,7))) - ((as.numeric(substr(zb_date,1,4))*12) + as.numeric(substr(zb_date,6,7)))) * ((orig_rt - LAST_RT) / 1200) * LAST_UPB), MODIR_COST), 2),
    MODFB_COST = round(if_else(COMPLT_FLG == '1', MODFB_COST + ((((as.numeric(substr(LAST_DTE,1,4))*12) + as.numeric(substr(LAST_DTE,6,7))) - ((as.numeric(substr(zb_date,1,4))*12) + as.numeric(substr(zb_date,6,7)))) * (LAST_RT / 1200) * non_int_upb), MODFB_COST), 2),
    COMPLT_FLG = as.numeric(COMPLT_FLG),
	orig_rt <- round(as.numeric(orig_rt),3)
  )

rm(baseTable1)
rm(baseTable5)
rm(modir_table)

### Create the seventh and final base table by selecting and reordering only the data fields we need for our statistical summary analysis
baseTable7 <- select(baseTable6,
  LOAN_ID, ORIG_CHN, SELLER, orig_rt, orig_amt,
  orig_trm, oltv, ocltv, num_bo, dti,
  CSCORE_B, FTHB_FLG, purpose, PROP_TYP, NUM_UNIT,
  occ_stat, state, zip_3, mi_pct, CSCORE_C,
  relo_flg, MI_TYPE, AQSN_DTE, ORIG_DTE, FRST_DTE,
  LAST_RT, LAST_UPB, msa, FCC_COST, PP_COST,
  AR_COST, IE_COST, TAX_COST, NS_PROCS, CE_PROCS,
  RMW_PROCS, O_PROCS, repch_flag, LAST_ACTIVITY_DATE,
  LPI_DTE, FCC_DTE, DISP_DTE, SERVICER, F30_DTE,
  F60_DTE, F90_DTE, F120_DTE, F180_DTE, FCE_DTE,
  F180_UPB, FCE_UPB, F30_UPB, F60_UPB, F90_UPB,
  MOD_FLAG, FMOD_DTE, FMOD_UPB, MODIR_COST, MODFB_COST,
  MODFG_COST, MODTRM_CHNG, MODUPB_CHNG, z_num_periods_120, F120_UPB,
  CSCORE_MN, ORIG_VAL, LAST_DTE, LAST_STAT, COMPLT_FLG,
  INT_COST, PFG_COST, NET_LOSS, NET_SEV, MODTOT_COST)

rm(baseTable6)

### Export the dataframe as a .csv
baseTable7 %>%
  dplyr::mutate_if(is.double, function(x) dplyr::if_else(is.na(x), NA_character_, format(x, scientific = FALSE, drop0trailing = TRUE, trim = TRUE))) %>%
  data.table::fwrite(paste0(fileYear, fileQtr, "_stat.csv"), sep = ",", na = "NULL", logical01 = TRUE, quote = TRUE, scipen = 100, col.names = TRUE, row.names = FALSE)

