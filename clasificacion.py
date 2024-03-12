import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')

# Presentacion de los datos
print(data.head())
print(data.describe())
print(data.info())

imput = SimpleImputer(missing_values=np.nan, strategy='mean')
# Eliminamos las filas con valores nulos
data = data.dropna()

# Reemplazamos los valores nulos por la media
data = pd.DataFrame(imput.fit_transform(data), columns=data.columns)

# Codificamos las variables categoricas
data = pd.get_dummies(data, columns=['Loan_ID'])

X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

# Separamos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamos el modelo k-NN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predecimos los valores de prueba
y_pred = knn.predict(X_test)

# Evaluamos el modelo
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', accuracy)

