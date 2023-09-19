import numpy as np
import pickle
import streamlit as st
from src.pipeline.predict_pipeline import Prediction_Pipeline


def churn_predict(input_data):
    input_data_flat = flat(input_data)
    print(input_data_flat)
    input_data_array = np.array([input_data_flat],dtype=(object))
    input_data_scaled = loaded_model_scaled.transform(input_data_array)
    input_data_reshape = input_data_scaled.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshape)
    print(prediction)
    
    
    if prediction[0] == 1:
        return 'the customer is churn'
    else:
        return 'the customer is not churn'
    
def main():
    st.title("Telecom Customer Churn Prediction")
    st.header("Enter the value of the following parameters:")
    
    SeniorCitizen = st.selectbox('Is the person Senior Citizen', ('yes','no'))
    Partner = st.selectbox('Do the person have partner', ('yes','no'))
    Dependents = st.selectbox('Do the person have dependents', ('yes','no'))
    tenure = st.number_input('tenure')
    PaperlessBilling = st.selectbox('Paperless billing', ('yes','no'))
    MonthlyCharges = st.number_input('Monthly Charges')
    TotalCharges = st.number_input('Total Charges')
    MultipleLines = st.selectbox('Multiple lines used', ('yes','no','No phone service'))
    InternetService = st.selectbox('Internet Service used', ('DSL','no','Fiber Optic'))
    OnlineSecurity = st.selectbox('Online Security', ('yes','no','No internet service'))
    OnlineBackup = st.selectbox('Online Backup', ('yes','no','No internet service'))
    DeviceProtection = st.selectbox('Device Protection', ('yes','no','No internet service'))
    TechSupport = st.selectbox('Tech Support', ('yes','no','No internet service'))
    StreamingTV = st.selectbox('Streaming TV', ('yes','no','No internet service'))
    StreamingMovies = st.selectbox('Streaming movies', ('yes','no','No internet service'))
    Contract = st.selectbox('Contract', ('Month-to-month','One year','Two year'))
    PaymentMethod = st.selectbox('Payment Method', ('Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check'))
        
    data = [[SeniorCitizen,Partner,Dependents,tenure,PaperlessBilling,
                                     MonthlyCharges,TotalCharges,MultipleLines,InternetService,
                                     OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,
                                     StreamingTV,StreamingMovies,Contract,PaymentMethod]]
    
    Churn_pred = ' '
    
    if st.button("Predict customer Churn"):
        obj = Prediction_Pipeline()
        Churn_pred = obj.predict(data=data)
        
    st.success(Churn_pred)
    
if __name__ == '__main__':
    main()
    


        
    
    
    