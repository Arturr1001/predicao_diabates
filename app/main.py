from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()
model = joblib.load("app/diabetes_model.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens. Para produção, especifique o domínio do frontend.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Patient(BaseModel):
    gender: str
    age: float
    race: str
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    hbA1c_level: float
    blood_glucose_level: int
    # As colunas one-hot serão montadas na função predict

@app.post("/predict")
def predict(p: Patient):
    # One-hot encoding manual para as raças
    races = ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]
    race_dict = {f"race:{r}": 1 if p.race == r else 0 for r in races}
    data_dict = {
        "gender": [p.gender],
        "smoking_history": [p.smoking_history],
        "age": [p.age],
        "bmi": [p.bmi],
        "hbA1c_level": [p.hbA1c_level],
        "blood_glucose_level": [p.blood_glucose_level],
        "hypertension": [p.hypertension],
        "heart_disease": [p.heart_disease],
        **{k: [v] for k, v in race_dict.items()}
    }
    df = pd.DataFrame(data_dict)
    pred = model.predict(df)[0]
    return {"diabetes": int(pred)}
