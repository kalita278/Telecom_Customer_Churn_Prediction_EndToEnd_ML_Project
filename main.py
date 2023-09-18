import os
from src.components.data_injestion import DataInjestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.utils import evaluate_metrix
import argparse
import pandas as pd
import mlflow
import sklearn
from urllib.parse import urlparse


def split():
    
    print("Initiate: Split the data into train and test is initiated....")
    obj=DataInjestion()
    obj.initiate_data_injestion()
    print("Spliting is completed .^.")
    

def train_and_evaluate():

    train_data = pd.read_csv('artifacts/data_injestion/train.csv')
    test_data = pd.read_csv('artifacts/data_injestion/test.csv')

    print("Initiate: Data Validation is initiated....")

    data_validation = DataValidation()
    data_validation.initiate_data_validation(train_data,test_data)

    print("Data Validation is completed .^.")
    print("Initiate: Data Transfomation is initiated....")

    data_transformation = DataTransformation()
    train_arr, test_arr, _ =data_transformation.initiate_data_transformation(train_data,test_data)

    print("Data Transformation is completed .^.")
    print("Initiate: Model Training is initiated....")
    
    model_trainer = ModelTrainer()
    x_train, y_train, x_test, y_test, model = model_trainer.initiate_model_trainer(train_arr,test_arr)

    print("Model Training is completed .^.")
    print("Initiate: Model Evaluation is initiated....")

    model_evaluate = ModelEvaluation()
    print("Accuracy Score for of the model:",model_evaluate.initiate_model_evaluation(x_train, y_train, x_test, y_test, model))


    print("Model Evaluation is completed .^.")

    mlflow.set_tracking_uri("https://dagshub.com/kalita278/Telecom_Customer_Churn_Prediction_End_to_End_ML_Project.mlflow")
    os.environ['MLFLOW_TRACKING_USERNAME'] = "kalita278"
    os.environ['MLFLOW_TRACKING_PASSWORD'] = "eb766988711ad6134d253dc8b82c079114a8a36c"

    with mlflow.start_run():

        # Get predictions
        prediction = model.predict(x_test)

        # Get metrics
        (roc_auc_scr, accuracy, precision, recall, f1) = evaluate_metrix(x_test, y_test, model)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric("roc_auc_score", roc_auc_scr)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store !="file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="Decision Tree Model")
        else:
            mlflow.sklearn.log_model(model, "model")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Split or Train_and_evaluate step:', dest='step')
    subparsers.required = True
    split_parser = subparsers.add_parser('split')
    split_parser.set_defaults(func=split)
    train_parser = subparsers.add_parser('train_and_evaluate')
    train_parser.set_defaults(func=train_and_evaluate)
    parser.parse_args().func()