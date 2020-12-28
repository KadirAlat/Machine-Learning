import pandas as pd
df = pd.read_csv('datasets/carprices.csv')
print(df)
dumm = pd.get_dummies(df['Car Model'])
merged = pd.concat([df,dumm],axis='columns')
final = merged.drop(['Car Model','Mercedez Benz C class'],axis='columns')
print(final)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
x = final.drop(['Sell Price($)'],axis='columns')
y = final['Sell Price($)']
model.fit(x,y)
print(model.predict([[45000,4,0,0]]))
print(model.predict([[86000,7,0,1]]))
print(model.score(x,y))
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder
le = LabelEncoder()
dfornek = df
dfornek['Car Model'] = le.fit_transform(dfornek['Car Model'])
print(dfornek)
X = dfornek[['Car Model','Mileage','Age(yrs)']].values
print(X)
Y = dfornek['Sell Price($)']
from sklearn.compose import  ColumnTransformer
ct = ColumnTransformer([('Car Model',OneHotEncoder(),[0])],remainder='passthrough')
X = ct.fit_transform(X)
print(X)
X = X[:,1:]
print(X)
model.fit(X,Y)
print(model.predict([[0,0,45000,4]]))