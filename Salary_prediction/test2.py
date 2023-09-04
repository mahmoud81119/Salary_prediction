import joblib
import numpy as np
from utils import process
import streamlit as st

# Loading the model
model = joblib.load('Models/RF_reg.pkl')

def Salary_prediction():
    # Streamlit
    st.title('Salary prediction')
    st.divider()

    # Input Fields
    Age = st.number_input('Age', min_value=21, max_value=62, step=1)
    Gender = st.radio('Gender', options=['Male', 'Female', 'Other'])
    Education = st.selectbox('Education', options=['Bachelors', 'Masters', 'PHD', 'High School'])
    Title = st.selectbox('Title', options=['Software/Developer', 'Data Analyst/scientist',
                                        'Manager/Director/VP', 'Sales', 'Marketing/Social Media',
                                        Customer Service/Receptionist', 'IT/Technical Support',
                                        'Product/Designer', 'Financial/Accountant', 'HR/Human Resources',
                                        'Operations/Supply Chain'])
    Experience = st.slider('Experience', value=0, min_value=0, max_value=34, step=1)
    Positions = st.radio('Positions', options=['Junior', 'Team Leader', 'Senior', 'Manager'])

    if st.button('Your Salary ..... '):
        # Concatenate Features
        new_data = np.array([Age, Gender, Education, Title, Experience, Positions])

        # Call to process Function from utils file
        X_processed = process(X_new=new_data)

        # Model prediction
        y_pred = model.predict(X_processed)[0]

        # Round y_pred
        y_pred = round(y_pred, 2)

        st.success(f'Your salary is: {y_pred}')


# Run Via Terminal
if __name__ == '__main__':
    # Call the above function
    Salary_prediction()