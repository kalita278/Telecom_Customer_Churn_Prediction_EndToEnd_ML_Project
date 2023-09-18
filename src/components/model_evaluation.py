import os
from dataclasses import dataclass
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from src.utils import save_json, evaluate_metrix

@dataclass
class ModelEvaluationConfig:
    model_evaluation_config_filepath = os.path.join('artifacts\model_evaluation','model_evaluation.json')

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation_config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, x_train, y_train,x_test, y_test, model):
        

        roc_auc_scr_train, accuracy_train, precision_train, recall_train, f1_train = evaluate_metrix(x_train, y_train, model)
        roc_auc_scr_test, accuracy_test, precision_test, recall_test, f1_test = evaluate_metrix(x_test, y_test, model)

        scores = {"Training Accuracy of the Model {}".format(model):
            {
            'roc_auc': roc_auc_scr_train,
            'accuracy': accuracy_train,
            'precision': precision_train,
            'recall': recall_train,
            'f1': f1_train
            },
            "Testing Accuracy of the Model {}".format(model):
            {
            'roc_auc': roc_auc_scr_test,
            'accuracy': accuracy_test,
            'precision': precision_test,
            'recall': recall_test,
            'f1': f1_test
            }
        }

        save_json(self.model_evaluation_config.model_evaluation_config_filepath, scores)
        #with open(self.model_evaluation_config.model_evaluation_config_filepath, 'w') as f:
        #            json.dump(scores,f, indent=4)
        
        return scores