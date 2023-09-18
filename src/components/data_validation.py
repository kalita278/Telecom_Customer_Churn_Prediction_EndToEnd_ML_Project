import os
from dataclasses import dataclass
import yaml
import pandas as pd

@dataclass
class DataValidationConfig:
    train_data_validation_file_path = os.path.join('artifacts\data_validation','train_data_validation.txt')
    test_data_validation_file_path = os.path.join('artifacts\data_validation','test_data_validation.txt')


class DataValidation:
    def __init__(self):
        self.data_validation_config = DataValidationConfig()
    
    def initiate_data_validation(self, train_path, test_path):

        Train_data_validation_status = None
        Test_data_validation_status = None


        #train_df = pd.read_csv(train_path)
        #test_df = pd.read_csv(test_path)
        train_df_cols = list(train_path.columns)
        test_df_cols = list(test_path.columns)

        schema_all = yaml.safe_load(open('schema.yaml'))
        all_schema = schema_all.keys()

        #all_schema = schema_cols.keys()
        
        for col in train_df_cols:
            if col not in all_schema:
                Train_data_validation_status = False
                with open(self.data_validation_config.train_data_validation_file_path, 'w') as f:
                    f.write(f"Train Data Validation status: {Train_data_validation_status}")
            else:
                Train_data_validation_status = True
                with open(self.data_validation_config.train_data_validation_file_path, 'w') as f:
                    f.write(f"Train Data Validation status: {Train_data_validation_status}")
        
        for col in test_df_cols:
            if col not in all_schema:
                Test_data_validation_status = False
                with open(self.data_validation_config.test_data_validation_file_path, 'w') as f:
                    f.write(f"Test Data Validation status: {Test_data_validation_status}")
            else:
                Test_data_validation_status = True
                with open(self.data_validation_config.test_data_validation_file_path, 'w') as f:
                    f.write(f"Test Data Validation status: {Test_data_validation_status}")
        
        

        return (self.data_validation_config.train_data_validation_file_path,
        self.data_validation_config.train_data_validation_file_path)