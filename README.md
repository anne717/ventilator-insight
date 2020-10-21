# ventilator-insight

Code behind ExtuMate, a tool to help ICU doctors determine when they can extubate a patient successfully.

## About

This repository contains the code I wrote as a Health Data Science Fellow at the 2020 Fall Insight Data science session in Boston. 

The goal of my project was to help doctors decide when it’s safe to take a patient off a ventilator. On the one hand, maintaining a patient on ventilation for too long poses a significant health risk to the patient. On the other hand, prematurely weaning a patient off the ventilator (a process known as extubation) may result in the patient having to be put back on the ventilator- an invasive procedure, which also carries substantial risks to the patient. 
To address this problem I created ExtuMate, a web application which takes patient biometric information, vital signs and lab values as its input and outputs the probability of extubation being unsuccessful (defined as the patient needing to be put back on the ventilator within the next 48 hours). ExtuMate also returns the patient-specific features that are driving this probability. ICU doctors can use this tool to better assess whether extubating their patient is likely to be successful and also to gain an insight to the patient variables that are contributing to this probability.  

For this project, I used data from the MIMIC-IV dataset, a relational database containing real hospital stays for ICU patients admitted to the Beth Israel Deaconess Medical Center in Boston, MA, USA. 
The MIMIC-IV download includes all data tables as separate `.csv` files. In order to facilitate querying the database (where the largest table was > 29 GB in size), I constructed a PostgresSQL database from the raw `.csv` data tables. This allowed me to query the patient event tables to: 
1) Extract the target classes:  Query the procedureevents table to find all ventilation events and use datetime information to identify patients that were reintubated within 48 hours of extubation.

2) Extract vital signs and lab values as features for each patient by querying the 27GB chartevents table using the chart `itemid` (an identification code specific to each event type where there are > 3500 individual event codes).

I encountered a number of challenges related to the nature of the raw data. For example:
- multiple ways of collecting the same information on a patient (e.g. the same patient could have a blood pressure cuff reading, as well as a blood pressure reading from the central line sensor logged at exactly the same time). 

- missing data. 

- “mistakes” in the data caused by user input error.

After cleaning the data, I was able to create a logistic regression model to classify patients based on the likelihood that they will need to be reintubated following extubation. Developing the model was an iterative process that involved going back and forth between the database to consider additional features and extra features that could be engineered from existing features. The final model improved on the ability of doctors to identify patients that need to be reintubated by 55%. The training data was also used to find measurable features values that contribute to the result classification (using LIME).
The model was deployed to a web application using streamlit and Heroku.


## Code

The notebooks should be run in the following order:
01explore_vents.ipynb: find patients that were intubated and determine whether they were reintubated within 48 hours
02_extract_sql.ipynb: uses pyscopg2 to query a PostgresSQL database containing all the MIMIC-IV tables and saves query tables as feather files
03_eventfeature_processing.ipynb: cleans data by scanning for duplicate readings and returns the cleaned data as a feather file
04_patient_age_and_sex.ipynb: obtains patient data from patients table not in the PostgreSQL database
05a_clean_data_labs_strip.ipynb: handles missing values, erroneous values and outliers
06a_EDA_labs_strip.ipynb: Exploratory data analysis on the dataset, inclusion of additional engineered features
07a_log_model_without pipeline.ipynb: notebook containing the finalized model, together with explainable features. The model, processing pipeline and a subset of the training data are returned and used for the web application.
