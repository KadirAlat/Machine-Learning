import pandas as pd
from sklearn import linear_model
from word2number import w2n
import math

data = pd.read_csv('datasets/hiring.csv')
data.experience[0] = 'zero'
data.experience[1] = 'zero'
a=0
for i in data.experience:
    data.experience[a] = w2n.word_to_num(i)
    a=a+1
data.rename(columns={'test_score(out of 10)':'test_score','interview_score(out of 10)':'interview_score','salary($)':'salary'},inplace=True)
median = math.floor(data.test_score.median())
data.test_score.fillna(median,inplace=True)
print(data)
reg = linear_model.LinearRegression()
reg.fit(data[['experience','test_score','interview_score']],data.salary)
print(reg.coef_)
print(reg.intercept_)
print(reg.predict([[2,9,6]]))
print(reg.predict([[12,10,10]]))


