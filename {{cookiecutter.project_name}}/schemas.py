from pydantic import BaseModel, Field
from typing import List

class PredictionRequest(BaseModel):
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    passenger_count: int

# Для списка запросов
class BatchPredictionRequest(BaseModel):
    data: List[PredictionRequest]

class PredictionResponse(BaseModel):
<<<<<<< HEAD:{{cookiecutter.project_name}}/schemas.py
    predictions: List[str]
=======
    predictions: List[str]
>>>>>>> 7fd6aad200107a9da3b86fe21a310961d659f746:schemas.py
