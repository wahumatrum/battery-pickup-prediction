# battery pickups 
>**Note:**
Redacted version of our delivery repo (December of 2022) for a german battery collecting company.   
You can watch [our presentation](https://www.youtube.com/watch?v=ODgKSD7HrL4&list=PLSizAuhe-ZaPGwjn6q2gOgc8L1p7BKgWF&index=22&pp=iAQB) on our project (ca. 10 min).  
Our Dashboard is not contained in this repository.

# The project summary

**Problem statement**

Battery collection peaks result in late pick-ups at the customer. 

**Project goal** 

- Increase customer happiness using data driven approaches.   
- Research questions of logistic provider performance: late pick-ups vs. On-time pick-ups. 
- Pattern detection in late pick-ups. 
- Predicting when containers are at capacity or close to it.

# Setup

Use the [requirements](requirements.txt) file in this repo to create a new environment. For this you can either use `make setup` or run the commands below

1. you can run the makefile

```BASH
make setup
```

OR

2. run the commands manually
  
```BASH
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Code
It is important to run all notebooks in order of the file name prefix starting with 00_.

## [Data cleaning and geocoding](python/00_base_cleaning_and_geocoding.ipynb)

This is a Jupyter notebook in which basic data cleaning and geocoding for the collection points are done. The data used in this notebook is the original excel data GRS Service GmbH provided us exported as csv file.

## [Data Cleaning](python/01_data_cleaning.ipynb)

This is a Jupyter notebook in which we dealed with missing values, duplicates, data types, transformation of data and dealing with outliers.

## [Prediction](python/02_prediction_db.ipynb)

This is a Jupyter notebook in which we dealt with reading data and selecting columns, plausibility checks, prepare data frame for later calculations, calculate mean weights of container-types from data, calculate the day when a collection point's containers are full, create a list of data frames for each collection point, automated predictions for all collection_points, store results into DB, diagrams and some tests.

## [Prediction accuracy](python/03_prediction_accuracy.ipynb)

This is a Jupyter notebook in which a prediction of chosen features can be done. After the prediction the accuracy can be calculated.

## [Testing](python/04_testing.ipynb)

This is a Jupyter notebook in which testing of plausibility filtering and splitting, filtering of single pick-ups and single initial deliveries are done.

## [Create Status Table](python/05_create_status_table.ipynb)

In this Jupyter notebook a table is created (code for storing in a mysql and excel file is provided) that provides more detail statistics about battery collection for each collection point:

* amount of containers in stock separately for every container type
* capacity in kg of collection poin
* daily collection in kg
* date of last pick up
* days since last pick up
* kg of batteries collected since last pick up
* % of capacity reached since last pick up
* remaining days until colleption point is at capacity
* date when collection point is at capacity

## Python modules

### [geocode.py](python/geocode.py)

Python module for retrieving lat/long data from Google API.

### [prediction.py](python/prediction.py)

The main python module including the core prediction code.
