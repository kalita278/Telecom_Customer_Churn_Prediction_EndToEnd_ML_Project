import os
import numpy as np
import pandas as pd
import dill
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import json
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def save_object(file_path,obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'wb') as file_obj:
        pickle.dump(obj,file_obj)

def save_json(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path,exist_ok=True)

    with open(file_path,'w') as f:
        json.dump(obj,f, indent=4)

def evaluate_metrix(x, y, model):


    y_proba = model.predict_proba(x)[:, 1]
    y_pred = model.predict(x)


    roc_auc_scr = round(roc_auc_score(y, y_proba),4)
    accuracy = round(accuracy_score(y, y_pred),4)
    precision = round(precision_score(y, y_pred, pos_label='Yes'),4)
    recall = round(recall_score(y, y_pred, pos_label='Yes'),4)
    f1 =  round(f1_score(y, y_pred, pos_label='Yes'),4)

    return roc_auc_scr, accuracy, precision, recall, f1


