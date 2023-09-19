import streamlit as st
from src.pipeline.predict_pipeline import Prediction_Pipeline
import pandas as pd

    
def main():
    st.title("Telecom Customer Churn Prediction")
    st.header("Enter the value of the following parameters:")
    
    SeniorCitizen = st.selectbox('Is the person Senior Citizen', (1,0))
    Partner = st.selectbox('Do the person have partner', ('Yes','No'))
    Dependents = st.selectbox('Do the person have dependents', ('Yes','No'))
    tenure = st.number_input('tenure')
    PaperlessBilling = st.selectbox('Paperless billing', ('Yes','No'))
    MonthlyCharges = st.number_input('Monthly Charges')
    TotalCharges = st.number_input('Total Charges')
    MultipleLines = st.selectbox('Multiple lines used', ('Yes','No','No phone service'))
    InternetService = st.selectbox('Internet Service used', ('DSL','no','Fiber optic'))
    OnlineSecurity = st.selectbox('Online Security', ('Yes','No','No internet service'))
    OnlineBackup = st.selectbox('Online Backup', ('Yes','No','No internet service'))
    DeviceProtection = st.selectbox('Device Protection', ('Yes','No','No internet service'))
    TechSupport = st.selectbox('Tech Support', ('Yes','No','No internet service'))
    StreamingTV = st.selectbox('Streaming TV', ('Yes','No','No internet service'))
    StreamingMovies = st.selectbox('Streaming movies', ('Yes','No','No internet service'))
    Contract = st.selectbox('Contract', ('Month-to-month','One year','Two year'))
    PaymentMethod = st.selectbox('Payment Method', ('Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check'))
        
    data = [[SeniorCitizen,Partner,Dependents,tenure,MultipleLines,InternetService,OnlineSecurity,
             OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,
             PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges]]
    df = pd.DataFrame(data,columns=['SeniorCitizen','Partner','Dependents','tenure','MultipleLines','InternetService','OnlineSecurity',
                                     'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                                     'StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'])
    
    Churn_pred = ' '
    
    if st.button("Predict customer Churn"):
        obj = Prediction_Pipeline()
        Churn_pred = obj.predict(data=df)
        
    st.success(Churn_pred[0])
    
if __name__ == '__main__':
    main()
    


        
    
    
    