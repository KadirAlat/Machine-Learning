import pandas as pd
df = pd.read_csv('datasets/titanic.csv')
print(df.head(5))
df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
print(df.head(6))
inputs = df.drop('Survived',axis='columns')
target = df.Survived
inputs.Sex = inputs.Sex.map({'male':1,'female':2})
print(inputs.Age[:10])
inputs.Age = inputs.Age.fillna(inputs.Age.mean())
print(inputs.head(10))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))