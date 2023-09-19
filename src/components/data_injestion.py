import os
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataInjestionConfig:
    train_data_path: str=os.path.join('artifacts\data_injestion',"train.csv")
    test_data_path: str=os.path.join('artifacts\data_injestion',"test.csv")
    raw_data_path: str=os.path.join('artifacts\data_injestion',"raw.csv")

class DataInjestion:
    def __init__(self):
        self.injestion_config=DataInjestionConfig()

    def initiate_data_injestion(self):


        #try:
        df1 = pd.read_csv("data\TelcomCustomer-Churn_1.csv")
        df2 = pd.read_csv("data\TelcomCustomer-Churn_2.csv")
        df= pd.merge(df1,df2,on='customerID')

        os.makedirs(os.path.dirname(self.injestion_config.train_data_path),exist_ok=True)
        df.to_csv(self.injestion_config.raw_data_path,index=False,header=True)

        train_set,test_set = train_test_split(df,test_size=0.2,random_state=23)
        train_set.to_csv(self.injestion_config.train_data_path,index=False,header=True)
        test_set.to_csv(self.injestion_config.test_data_path,index=False,header=True)

        return(self.injestion_config.train_data_path,
                self.injestion_config.test_data_path)
        #except:
         #   pass

#if __name__ == "__main__":
 #   obj=DataInjestion()
  #  train_data,test_data = obj.initiate_data_injestion()

   # data_validation = DataValidation()
    #data_validation.initiate_data_validation(train_data,test_data)


    #data_transformation = DataTransformation()
    #rain_arr, test_arr, _ =data_transformation.initiate_data_transformation(train_data,test_data)
    
    #model_trainer = ModelTrainer()
    #x_train, y_train, x_test, y_test, model = model_trainer.initiate_model_trainer(train_arr,test_arr)

    #model_evaluate = ModelEvaluation()
    #Sprint("Accuracy Score for of the model:",model_evaluate.initiate_model_evaluation(x_train, y_train, x_test, y_test, model))
    
