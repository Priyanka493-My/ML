import numpy as np
import joblib
from fastapi import FastAPI,HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from pathlib import Path
import uvicorn

def predict_single(dv,model,customer):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:,1]
    return y_pred[0]

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR.parent / 'Classification' /'churn-model.bin'

dv, model = joblib.load(model_path)

app = FastAPI(title='churn')
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
  print('--- VALIDATION ERROR DETAILS ---')
  print(exc.errors())
  print('--------------------------------')
  from fastapi.responses import JSONResponse

  return JSONResponse(status_code=422, content={'detail': exc.errors()})

class Customer(BaseModel):
    customerid: str
    gender: str
    seniorcitizen: int
    partner: str
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float

@app.post('/predict')
async def predict(customer:Customer):
    try:
        customer = customer.model_dump()
        prediction = predict_single(dv,model,customer)
        churn = prediction >= 0.5
        return {
              'churn_probability': float(prediction),
              'churn': bool(churn),
               }
    except Exception as e:
        print('Prediction Error:', str(e))
        raise e

if __name__ == '__main__':
  uvicorn.run(app, host='0.0.0.0', port=9696)