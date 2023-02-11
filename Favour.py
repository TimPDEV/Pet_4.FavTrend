

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Загрузить данные
data = pd.read_csv('data.csv')

# Разделить данные на признаки и целевую переменную
X = data.drop(['target'], axis=1)
y = data['target']

# Создать модель линейной регрессии
model = LinearRegression()

# Обучить модель с использованием данных
model.fit(X, y)

# Выполнить прогнозирование на основе модели
predictions = model.predict(X)

# Проверить качество модели, используя среднюю абсолютную ошибку
mae = mean_absolute_error(y, predictions)
print('MAE:', mae)
