import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
 
# 1. Загрузка и предобработка данных
df = pd.read_csv("uber.csv")
 
# Удалим пропуски и выбросы
df = df.dropna()
df = df[(df['fare_amount'] > 0) & (df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]
 
# Выбор признаков
X = df[['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count']]
y = df['fare_amount']
 
# 2. Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# 3. GradientBoostingRegressor
gbr_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gbr_model.fit(X_train, y_train)
gbr_preds = gbr_model.predict(X_test)
 
# 4. KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
 
# 5. LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
 
# 6. Оценка моделей
def print_metrics(name, y_true, y_pred):
    print(f"\n{name}")
    print(f"RMSE: {mean_squared_error(y_true, y_pred):.3f}")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.3f}")
    print(f"R2: {r2_score(y_true, y_pred):.3f}")
 

print_metrics("GradientBoostingRegressor", y_test, gbr_preds)
print_metrics("KNeighborsRegressor", y_test, knn_preds)
print_metrics("LinearRegression", y_test, lr_preds)