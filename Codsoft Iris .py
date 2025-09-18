import pandas as pd
data=pd.read_csv("/content/drive/MyDrive/IRIS.csv")
data.head()
data.describe()
data.isnull().sum()

from sklearn.model_selection import train_test_split
x=data['sepal_length']
y=data['sepal_width']
x_tr, x_t, y_tr, y_t = train_test_split(x, y, random_state=20, test_size=0.28)

print("Train shapes:", x_tr.shape, y_tr.shape)
print("Test shapes:", x_t.shape, y_t.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
x = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']
x_tr, x_t, y_tr, y_t = train_test_split(x, y, test_size=0.24, random_state=22)
clf = RandomForestClassifier(n_estimators=50, random_state=22)
clf.fit(x_tr, y_tr)
y_pred = clf.predict(x_t)
print("Accuracy:", accuracy_score(y_t, y_pred))
print("\nClassification Report:\n", classification_report(y_t, y_pred))
import matplotlib.pyplot as plt
plt.plot(x_t,y_t,'o',label='actual')
plt.plot(x_t,y_pred,'o',label='predicted')
plt.legend()
plt.show()
