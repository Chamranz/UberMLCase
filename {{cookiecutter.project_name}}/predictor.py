import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURE_ORDER = [
    'pickup_latitude',
    'pickup_longitude',
    'dropoff_latitude',
    'dropoff_longitude',
    'passenger_count'
]

def train_and_save_model():
    df = pd.read_csv("uber.csv")
    df = df[(df['fare_amount'] > 0) & (df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
    X = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count']]
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    predictor = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5
    )

    predictor.fit(X_train, y_train)
    joblib.dump(predictor, "predictor.pkl")

def load_model():
    return joblib.load("predictor.pkl")

def predicted_prices(model, features_list):

    """
    Делает предсказания стоимости поездок на основе входных признаков/
    
    Parameters:
        model (GradientBoostingRegressor): Обученная модель.
        features_list (list of dict): Список словарей, каждый из которых содержит значения признаков:
            - pickup_latitude (float)
            - pickup_longitude (float)
            - dropoff_latitude (float)
            - dropoff_longitude (float)
            - passenger_count (int)
    """

    df = pd.DataFrame(features_list, columns=FEATURE_ORDER)

    predictions = model.predict(df)

    return predictions.tolist()