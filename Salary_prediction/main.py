# import Libraries
import numpy as np
import joblib
from fastapi import FastAPI , Form

# My Custom Function
from utils import process


# Load the model
model = joblib.load('Models/RF_reg.pkl')


# Initilaize An App

app = FastAPI()



# Function to prediction
@app.post('/Salary')
async def Salary_prediction(Age:float = Form(...),Gender:str = Form(... , description='Gender' , enum =['Male', 'Female', 'Other'] )
                            ,Education:str = Form(... , description='Education' , enum = ['Bachelors', 'Masters', 'PHD', 'High School'])
                            ,Title:str = Form(... , description= 'Title' , enum = ['Software/Developer', 'Data Analyst/scientist', 'Manager/Director/VP', 'Sales', 'Marketing/Social Media',
                                                                                    'Customer Service/Receptionist', 'IT/Technical Support','Product/Designer', 'Financial/Accountant', 'HR/Human Resources',
                                                                                        'Operations/Supply Chain'])
                            ,Experience:float = Form(...), 
                            Positions:str = Form (... , description='Positions' , enum = ['Junior', 'Team Leader', 'Senior', 'Manager'])):


    # Concatenate All features
    new_data  = [Age , Gender , Education , Title , Experience , Positions]

    # Call the Function 'process' from utils file
    X_processed = process(X_new=new_data)

    # To Model
    y_pred = model.predict(X_processed)[0]

    y_pred_rounded = round(y_pred , 1)

    return y_pred_rounded



@app.get('/name')
async def welcome(username : str):
    return f'welcom : {username}'


