import pandas as pd
from matplotlib import  pyplot as plt
df = pd.read_csv('datasets/HR_comma_sep.csv')
left = df[df.left==1]
retained = df[df.left==0]
print(df.groupby('left').mean())
pd.crosstab(df.salary,df.left).plot(kind='bar')
#Above bar chart shows employees with high salaries are likely to not leave the company
plt.show()
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.drop('salary',axis='columns',inplace=True)
print(df_with_dummies.head())
X = df_with_dummies
y = df.left
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
model.predict(X_test)
model.score(X_test,y_test)


