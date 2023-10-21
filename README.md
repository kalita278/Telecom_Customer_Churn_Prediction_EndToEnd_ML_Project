# Telecom_Customer_Churn_Prediction_End_to_End_Machine_Learning_Project

### **DOMAIN:** Telecom

## **CONTEXT:**
A telecom company wants to use their historical customer data and leverage machine learning to predict behaviour in an attempt to retain customers. The end goal is to develop focused customer retention programs.

## **DATA  DESCRIPTION:** 
Each  row  represents  a  customer,  each  column  contains  customer’s  attributes  described  on  the  column  Metadata.  The data set includes information about:

•Customers who left within the last month – the column is called Churn

•Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies

•Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges

•Demographic info about customers – gender, age range, and if they have partners and dependents.

## **PROJECT  OBJECTIVE:** 
The  objective,  as  a  data  scientist  hired  by  the  telecom  company,  is  to  build  a  model  that  will  help  to  identify  the potential customers who have a higher probability to churn. This will help the company to understand the pain points and patterns of customer churn and will increase the focus on strategising customer retention.

## **PROJECT STRUCTURE:**

**src:** consists of Python scripts

**artifacts:** consists of files (artifacts) in each step

**data:** consists of data

**notebook:** consists of Jupyter Notebooks



## **DATA EXPLORATION AND DATA CLEANING:**

Combined the two files to create a single file with all the relevant variables and perform the necessary data quality checks and cleaning. In data cleaning, identified the missing values/unexpected values in the dataset and since the number of missing values is very less, dropped the missing value rows. Also, checked if there is any outliers in the dataset. Make sure that the data types of the variables are appropriate as required for our analysis (Here, converted all the categorical variable data types to category and continuous variable data types to int or float).

## **DATA ANALYSIS:**

Checked the distribution of data for the continuous (histogram charts) and categorical (pie charts) variables along with the target variable (pie charts). Performed uni-variate, bivariate, and multivariate analysis of the dataset, for example, 5-point summary of the continuous variable, pair plot, correlation matrix plot, and boxplot to detect outliers.


## **DATA PREPROCESSING:**

Here, removed the unwanted/irrelevant independent variable/feature based on the above analysis from the dataset. Also, encoded the categorical variable using one-hot encoding and label encoding based on the categories, and scaled the independent variable using standard scaler.


## **MODEL BUILDING, EVALUATION AND IMPROVEMENT:**

Used k nearest neighbour, logistic regression, support vector machine (SVM), naive bayes, decision tree, random forest, Ada boost and gradient boost algorithm to predict the churn customer and evaluate the model using confusion metrix, accuracy score, recall score, precision score and f1 score. Further tunned the models using GridsearchCV and compared all the models to find the best model.

## **MLOPS:**
Used mlops tools to version the model and data that becomes scalable and reproducable in future.

## **MLOPS Tools:**
MLFlow, DVC, GIT, Dagshub


## **MODEL DEPLOYMENT:**

**Web Application:** Streamlit

**Container:** Docker

**Cloud deployment:** Microsoft Azure

**App link:** https://telecomchurnprediction.azurewebsites.net/

**Dagshub Repository link:** https://dagshub.com/kalita278/Telecom_Customer_Churn_Prediction_EndToEnd_ML_Project


### **How to use it**
To use this project, first clone the repo on your device using the command below:

git clone https://github.com/kalita278/Telecom_Customer_Churn_Prediction_EndToEnd_ML_Project.git

## **Installation**
pip install -r requirements.txt

## **How to run the project**
1. Run: "streamlit run churn_prediction_web_app.py" to get the localhost server link. 

2. Go to browser and load  http://127.0.0.1:8501/


