raw/2024Q1.csv is downloaded from https://datadynamics.fanniemae.com/data-dynamics/#/downloadLoanData/Single-Family, Acquisition and Performance 2024Q1. And this file is large thus not uploaded to github.

raw/cliper.py is to slice lines from 2024Q1.csv and store the new file into raw_process

2024Q1_200lines.csv is the first 200 lines of 2024Q1.csv, which works as a sample file for testing

raw_process/ippub_statfile.py is to transform the file(such as 2024Q1_200lines.csv), and store the new file in data_prepare.
ippub_statfile.py mainly does:
- select single loans which are acquired in the corresponding month but not prepaid in the same month
- add a column of indicator which tell the loan is prepaid or not in the next 3 month using certain rules.

data_prepare/dataprepare_training.ipynb is to split the file(such as 2024Q1_sample_1.csv) into training and validation data which are stored in model_training.
dataprepare_training.ipynb mainly does:
- add a column of market rate, which is the average market mortgage rate for the quarter of the data.
- select columns are needed for the model training
- split data into training and validation ones and store them in model_training folder for model training.

data preparation procedure:
raw -> raw_process -> data_prepare -> model_training
