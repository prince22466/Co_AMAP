####################################################################
# This code aggregates the Loan Performance data into a one-row-per-loan analytical dataset.
# Files are sequentially read in, modified, and then exported to .csv format in your working directory.
# We recommend running this code and LPPUB_StatSummary.R to ensure you have downloaded the data in full.
# Please be sure this code, along with LPPUB_StatFile_Production.R, is in the same directory as the Loan Performance data.
####################################################################

### Required packages
# Tested with data.table v.1.14.8 and dplyr v1.1.4
if (!(require(data.table))) install.packages ("data.table")
if (!(require(dplyr))) install.packages ("dplyr")

### Set up a function to read in the Loan Performance files
load_lppub_file <- function(filename, col_names, col_classes){
  df <- fread(filename, sep = "|", col.names = col_names, colClasses = col_classes)
  return(df)
}

### Define the set of files to read in. PLEASE EDIT TO ACCOMODATE YOUR ANALYSIS AND INCLUDE LATER DATA RELEASES ###
starting_file <- 0 #0 starts on 2000Q1
ending_file <- 95 #95 ends on 2023Q4
setwd("~/lppub_r_code/")
#Process the files (outputs to csv)
for (file_number in starting_file:ending_file){

	#Set up file names
	fileYear <- file_number %/% 4
	if(nchar(fileYear) == 1){
	  fileYear <- paste0('0', fileYear)
	}
	fileYear <- paste0('20', fileYear)
	fileQtr <- (file_number %% 4) + 1
	fileQtr <- paste0('Q',fileQtr)
	FileName <- paste0(fileYear, fileQtr, '.csv')

	#Run helper file
	source('LPPUB_StatFile_Production.R')
}
