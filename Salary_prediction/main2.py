import joblib
import numpy as np
from PIL import Image
from utils import process
import streamlit as st


# Loading the model
model = joblib.load('Models/RF_reg.pkl')


# Streamlit 


st.title('Salary prediction')
st.divider()

st.video('04.09.2023_20.17.22_REC.mp4')
st.divider()

Image = Image.open('Salary_Expectation.jpg')
st.image(Image , caption= 'let is go to predict your salary'  , use_column_width=True)
st.divider()

# Input Fields
Age = st.slider('Age' , min_value=21 , max_value=62 , step=1)
st.divider()
Gender = st.radio('Gender' , options=['Male', 'Female', 'Other'])
st.divider()
Education = st.selectbox('Education' ,options=['Bachelors', 'Masters', 'PHD', 'High School'])
st.divider()
Title = st.selectbox('Title' , options=['Software/Developer', 'Data Analyst/scientist',
       'Manager/Director/VP', 'Sales', 'Marketing/Social Media',
       'Customer Service/Receptionist', 'IT/Technical Support',
       'Product/Designer', 'Financial/Accountant', 'HR/Human Resources',
       'Operations/Supply Chain'])
st.divider()
Experience = st.slider('Experience' ,value=0 , min_value=0 , max_value=34 , step=1)
st.divider()
Positions = st.radio('Positions' ,  options=['Junior', 'Team Leader', 'Senior', 'Manager'])
st.divider()

if st.button('Your Salary ..... '):
    # Concatenate Features
    new_data = np.array([Age , Gender , Education , Title , Experience ,  Positions])

    # Call to process Function from utils file 
    X_processed =  process(X_new=new_data)

    # Model prediction
    y_pred =  model.predict(X_processed)[0]

    # Round y_pred
    y_pred = round(y_pred , 2)

    st.success(f'your salary is : {y_pred}')








