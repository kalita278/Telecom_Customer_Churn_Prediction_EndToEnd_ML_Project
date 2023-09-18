import os
from src.utils import save_object
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from dataclasses import dataclass
import yaml

@dataclass
class ModelTrainerConfig:
    model_trainer_config_file_path = os.path.join('artifacts\model_trainer','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array,test_array):

        x_train, y_train, x_test, y_test = (train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])


        param = yaml.safe_load(open("params.yaml"))
        
        model = DecisionTreeClassifier(random_state=1)
        model.fit(x_train, y_train)

        save_object(file_path=self.model_trainer_config.model_trainer_config_file_path,obj=model)

        return (x_train, y_train, x_test, y_test,model)


        