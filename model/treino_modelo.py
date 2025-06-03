import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Carregar os dados
df = pd.read_csv("diabetes_dataset.csv")

# Remover colunas desnecessárias
df = df.drop(columns=["year", "location", "clinical_notes"])

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

categorical = ["gender", "smoking_history"]
numerical = ["age", "bmi", "hbA1c_level", "blood_glucose_level"]

# Pré-processamento
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first"), categorical),
    ("num", StandardScaler(), numerical)
], remainder="passthrough")

# Modelo + Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Divisão de dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinamento
pipeline.fit(X_train, y_train)

# Garantir que a pasta app existe
os.makedirs("../projeto-diabetes/app", exist_ok=True)
# Salvar modelo
try:
    joblib.dump(pipeline, "../app/projeto-diabetes/diabetes_model.pkl")
    print("Modelo salvo em ../app/projeto-diabetes/diabetes_model.pkl")
except Exception as e:
    print(f"Erro ao salvar modelo: {e}")
