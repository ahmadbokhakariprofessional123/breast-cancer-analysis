from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("models/breast_cancer_model.pkl")


class TumorData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float


@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API"}


@app.post("/predict")
def predict(data: TumorData):

    features = np.array([
        data.radius_mean,
        data.texture_mean,
        data.perimeter_mean,
        data.area_mean,
        data.smoothness_mean,
        data.compactness_mean,
        data.concavity_mean,
        data.concave_points_mean,
        data.symmetry_mean,
        data.fractal_dimension_mean,
        data.radius_se,
        data.texture_se
    ]).reshape(1, -1)

    prediction = model.predict(features)

    if prediction[0] == 1:
        return {"prediction": "Malignant tumor detected"}
    else:
        return {"prediction": "Benign tumor detected"}