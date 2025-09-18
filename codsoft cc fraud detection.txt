import pandas as pd
data=pd.read_csv("/content/drive/MyDrive/Untitled folder/creditcard.csv")
data.head()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
x=data.drop(columns='Class')
y=data['Class']
x_tr, x_t, y_tr, y_t = train_test_split(x, y, random_state=10, test_size=0.20)
l=LogisticRegression(max_iter=100, solver='liblinear')
l.fit(x_tr,y_tr)
y_pred=l.predict(x_t)
print("Accuracy:",accuracy_score(y_t,y_pred))
print("confusion matrx ",confusion_matrix(y_t,y_pred))
print("classification report",classification_report(y_t,y_pred))

import matplotlib.pyplot as plt
plt.bar(data['Amount'],data['V1'],data['V2'],data['V10'])
plt.show()




