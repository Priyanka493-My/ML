from fastapi import FastAPI

app = FastAPI(title='churn')

@app.get('/ping')
def ping():
    return {'status':"PONG"}