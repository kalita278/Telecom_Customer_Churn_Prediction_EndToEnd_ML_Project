from src.components.data_injestion import DataInjestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
import argparse
import pandas as pd

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
    x_train, y_train, x_test, y_test, model, parameter = model_trainer.initiate_model_trainer(train_arr,test_arr)

    print("Model Training is completed .^.")
    print("Initiate: Model Evaluation is initiated....")

    model_evaluate = ModelEvaluation()
    print("Accuracy Score for of the model:",model_evaluate.initiate_model_evaluation(x_train, y_train, x_test, y_test, model, parameter))


    print("Model Evaluation is completed .^.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='Split or Train_and_evaluate step:', dest='step')
    subparsers.required = True
    split_parser = subparsers.add_parser('split')
    split_parser.set_defaults(func=split)
    train_parser = subparsers.add_parser('train_and_evaluate')
    train_parser.set_defaults(func=train_and_evaluate)
    parser.parse_args().func()