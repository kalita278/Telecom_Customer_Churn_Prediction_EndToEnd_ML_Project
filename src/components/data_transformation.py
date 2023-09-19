from dataclasses import dataclass
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_filepath = os.path.join('artifacts\data_transformation','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()

    def get_transform_data(self):

        numerical_columns=['tenure','MonthlyCharges','TotalCharges']
        onehot_encoding_cat_columns=['Partner','Dependents','PaperlessBilling','MultipleLines','InternetService','OnlineSecurity',
                                     'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                                     'StreamingMovies','Contract','PaymentMethod','SeniorCitizen']


        numerical_columns_pipeline=Pipeline([("Imputer",SimpleImputer(strategy='mean')),
                                            ("Scaling",StandardScaler(with_mean=False))])
        cat_columns_pipeline=Pipeline([("Imputer",SimpleImputer(strategy='most_frequent')),
                                                     ("Onehot_encoding",OneHotEncoder()),
                                            ("Scaling",StandardScaler(with_mean=False))])                                  
        
        preprocessor = ColumnTransformer([("numerical_columns_pipeline",numerical_columns_pipeline,numerical_columns),
                                          ("Cat_onehot_encoding",cat_columns_pipeline,onehot_encoding_cat_columns)]
                                            )
        return preprocessor
    
    def initiate_data_transformation(self,train_path,test_path):
        #train_df = pd.read_csv(train_path)
        #test_df = pd.read_csv(test_path)
        train_df = train_path.drop(index=train_path['TotalCharges'][train_path["TotalCharges"]==' '].index,axis=0)
        test_df = test_path.drop(index=test_path['TotalCharges'][test_path["TotalCharges"]==' '].index,axis=0)

        preprocessing_obj = self.get_transform_data()

        input_feature_train_df = train_df.drop(['gender','PhoneService','customerID'],axis=1)
        target_feature_train_df = train_df['Churn']

        input_feature_test_df = test_df.drop(['gender','PhoneService','customerID'],axis=1)
        target_feature_test_df = test_df['Churn']

        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]


        save_object(file_path=self.data_tranformation_config.preprocessor_obj_filepath,
                    obj=preprocessing_obj)
        
        return (train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_filepath)
    


