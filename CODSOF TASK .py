pip install scikit-learn xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("/content/IMDb Movies India.csv", encoding='latin1')
print(data.head())
data.info()
data.isnull().sum()
data=data.drop_duplicates()

np.random.seed(0)
x=np.random.normal(10,2,5)
y=np.random.normal(10,2,5)
plt.plot(x,y,color='yellow')
plt.show()
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
data = pd.read_csv("/content/drive/MyDrive/Untitled folder/IMDb Movies India.csv", encoding='latin1')


data['Duration'] = data['Duration'].astype(str).str.replace('min', '').str.strip()
data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce')
data['Votes'] = data['Votes'].astype(str).str.replace(',', '')
data['Votes'] = pd.to_numeric(data['Votes'], errors='coerce')
data = data.dropna(subset=['Duration', 'Votes', 'Rating'])


x = data[['Duration', 'Votes']]
y = data['Rating']

x_tr, x_t, y_tr, y_t = train_test_split(x, y, random_state=20, test_size=0.28)

print("Train shapes:", x_tr.shape, y_tr.shape)
print("Test shapes:", x_t.shape, y_t.shape)

f = XGBRegressor()
model = f.fit(x_tr, y_tr)
pre = model.predict(x_t)
print("Predictions:", pre)
mmse=mean_squared_error(y_t,pre)
print("mse",mmse)


plt.plot(x,y,'o')
plt.show()
