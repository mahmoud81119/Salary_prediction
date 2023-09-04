import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Define the input data schema
class InputData(BaseModel):
    Age: int
    Gender: str
    Education: str
    Title: str
    Experience: int
    Positions: str

# Load the trained model
model = joblib.load('Models/xgb_reg.pkl')

# Initialize the FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post('/predict')
def predict_salary(data: InputData):
    # Extract the features from the input data
    features = [data.Age, data.Gender, data.Education, data.Title, data.Experience, data.Positions]
    
    # Make predictions using the loaded model
    prediction = model.predict([features])[0]
    
    # Return the prediction as a JSON response
    return {'prediction': prediction}

# Run the application with uvicorn server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8010)