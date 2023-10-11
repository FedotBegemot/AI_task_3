import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = np.genfromtxt('04cars.dat', delimiter=';')
# Матрица с признаками
X = data[:, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]
# Вектор с целевой переменной для классификации
y = data[:, 2]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Функция разделяет матрицу признаков X и вектор целевой переменной y на обучающие (X_train, y_train)
# и тестовые (X_test, y_test) наборы данных

# Переменные для отслеживания наилучшего значения k и точности
best_k = None
best_accuracy = 0

# нахождение лучшего значения k
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    # Модель KNN обучается на данных x_train и y_train
    knn.fit(X_train, y_train)
    # Модель используется для предсказания классов на тестовых данных
    y_pred = knn.predict(X_test)
    # Вычисляется точность модели сравнивая предсказанные значения и истинные
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

# обучение модели с наилучшим значением k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# матрица сопряженности для оценки качества модели
conf_matrix = confusion_matrix(y_test, y_pred)
print("Матрица сопряженности:")
print(conf_matrix)

print(f"Наилучшее значение k: {best_k}")
# % ошибок = 1 - точность.
error_rate = 1 - accuracy_score(y_test, y_pred)
print(f"Процент ошибок: {error_rate * 100:.2f}%")
