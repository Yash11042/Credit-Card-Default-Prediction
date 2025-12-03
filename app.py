from fastapi import FastAPI , HTTPException , Depends , Header
from pydantic import BaseModel
from typing import TypedDict , Annotated , Literal , Optional
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

#load the trained model

model = joblib.load('model.pkl')

#initialize the app

app = FastAPI(title="Credit Card Default Prediction API" , description="An API to predict credit card default using a trained ML model" , version="1.0.0")

#pydantic model for input data validation   

class CreditCardData(BaseModel):
    LIMIT_BAL: float
    SEX: Literal[1, 2]
    EDUCATION: Literal[1, 2, 3, 4]
    MARRIAGE: Literal[1, 2, 3]
    AGE: float
    PAY_0: float
    PAY_2: float
    PAY_3: float
    PAY_4: float
    PAY_5: float
    PAY_6: float
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float
    

#home
@app.get('/home')
def home():
    return {"message": "Welcome to the Credit Card Default Prediction API"}

@app.get('/health')
def health_check():
    return {"status": "OK"}

@app.post('/predict')
def predict_default(data: CreditCardData):
    try:
        #convert the input data to a dataframe
        input_data = pd.DataFrame([data.dict()])
        
        #make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        #return the prediction
        return {
            "prediction": int(prediction[0]),
            "probability": {
                "no_default": float(prediction_proba[0][0]),
                "default": float(prediction_proba[0][1])
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))