import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Признаки в правильном порядке. Для подстраховки, если JSON-посылка будет некорректна
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
    # Выбор признаков
    X = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count']]
    y = df['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Против утечки данных из тестовой выборки

    predictor = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42)
    predictor.fit(X_train, y_train)
    joblib.dump(predictor, "predictor.pkl")


def load_model():
    return joblib.load("predictor.pkl")


def predicted_prices(model, features_list: list):
    df = pd.DataFrame(features_list, columns=FEATURE_ORDER)
    predictions = model.predict(df)
    return predictions.tolist()