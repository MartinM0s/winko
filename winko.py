path = 'https://raw.githubusercontent.com/Kamil128/SDA_Excercise/main/data/wine/WineQT.csv'

import pandas as pd
import numpy as np

df = pd.read_csv(path)

# print(df)
# print(df.shape)

#sprawdzenie rodzaju danych - float64 i int64, nie trzeba nic zmieniać
#same dane ciągłe, nie mamy danych kategorycznych
#nie ma  null, więc nie musimy uzupełniać
# print(df.info())

#wyrzucamy jedyną niepotrzebna kolumnę 'Id'
col_to_drop = ['Id']
df.drop(col_to_drop, axis=1, inplace=True)

#definiujemy X i y
X = df.drop(['quality'], axis=1)
y = df['quality']


#tworzenie zbioru treningowego i testowego
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42,
                                                      test_size=0.2,
                                                      stratify=y)


#tworzenie pipelinu
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline(
    [
     ('standard_scaller', StandardScaler()),
    ]
)

X_train_tr = num_pipeline.fit_transform(X_train)
print(X_train_tr)

#wybieramy model
#1. regresja logistyczna - dobra do predykcji klas binarnych
#2.

