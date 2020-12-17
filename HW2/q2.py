import csv
import numpy as np
from matplotlib import pyplot as plt
independent_features = []
temp_independent_list=[]
dependent_features=[]
with open('q2_dataset.csv','r') as csv_file:
    reader = csv.reader(csv_file)
    a =1
    next(reader)
    for line in reader:
        for value in line:
           if (a == len(line)):
               pass
           else:
            temp_independent_list.append(value)
           if (a==len(line)-1):
            temp_independent_list.append(1)      #adding ones to end of independent feature rows
           if(a==len(line)):
               dependent_features.append(value)
           a = a + 1
        independent_features.append(temp_independent_list)
        temp_independent_list=[]
        a=1


def get_weights(train_set,train_label):
    transpose = np.transpose(train_set)
    multiplication = np.matmul(transpose,train_set)
    inverse = np.linalg.inv(multiplication)
    before_theta = np.matmul(inverse,transpose)
    theta = np.matmul(before_theta,train_label)
    return theta

def chose_test_fold(num,independent_features,dependent_features): # number between 1-5 will determine the partition that will use for testing others for training

    test_fold = []
    test_fold_label=[]
    train_fold = []
    train_fold_label = []
    if (num == 1):
        i=0
        while(i<100):
           test_fold.append(independent_features[i])
           test_fold_label.append(dependent_features[i])
           i += 1
        while(i<500):
            train_fold.append(independent_features[i])
            train_fold_label.append(dependent_features[i])
            i +=1
    elif (num == 2):
        i=0
        while(i<100):
           train_fold.append( independent_features[i])
           train_fold_label.append(dependent_features[i])
           i += 1
        while (i <200):
            test_fold.append(independent_features[i])
            test_fold_label.append(dependent_features[i])
            i += 1
        while(i<500):
            train_fold.append(independent_features[i])
            train_fold_label.append(dependent_features[i])
            i +=1

    elif (num == 3):
        i=0
        while(i<200):
           train_fold.append( independent_features[i])
           train_fold_label.append(dependent_features[i])
           i += 1
        while (i < 300):
            test_fold.append(independent_features[i])
            test_fold_label.append(dependent_features[i])
            i += 1
        while(i<500):
            train_fold.append(independent_features[i])
            train_fold_label.append(dependent_features[i])
            i +=1

    elif (num == 4):
        i=0
        while(i<300):
           train_fold.append(independent_features[i])
           train_fold_label.append(dependent_features[i])
           i += 1
        while (i < 400):
            test_fold.append(independent_features[i])
            test_fold_label.append(dependent_features[i])
            i += 1
        while(i<500):
            train_fold.append(independent_features[i])
            train_fold_label.append(dependent_features[i])
            i +=1

    elif (num == 5):
        i=0
        while(i<400):
           train_fold.append( independent_features[i])
           train_fold_label.append(dependent_features[i])
           i += 1
        while(i<500):
            test_fold.append(independent_features[i])
            test_fold_label.append(dependent_features[i])
            i +=1

    train_fold=np.array(train_fold,dtype=float)
    train_fold_label=np.array(train_fold_label,dtype=float)
    test_fold=np.array(test_fold,dtype=float)
    test_fold_label=np.array(test_fold_label,dtype=float)

    return (train_fold,train_fold_label,test_fold,test_fold_label)

def r_squared_error(real,estimation):
    mean = sum(real)/len(real)
    denominator_array = real-mean
    denominator =0
    numerator_array = estimation-mean
    for i1 in denominator_array:
        denominator += i1*i1
    numerator=0
    for i2 in numerator_array:
        numerator += i2*i2
    result = numerator/denominator
    return result
def mean_squared_error(real,estimation):
    x=real-estimation
    total = 0
    for i in x:
        total += i*i
    result = total / len(x)
    return result
def mean_absolute_error(real,estimation):
    x = real-estimation
    total = 0
    for i in x:
        if(i<0):
            i = abs(i)
            total += i
        else:
            total +=i
    result = total/len(x)
    return result
def mape_error(real,estimation):
    x = real-estimation
    temp = []
    a=0
    for i in x:
       temp.append(abs(i/real[a]))
       a += 1
    total = 0
    for a in temp:
        total += a
    result = total/len(x)
    return result
def lasso_regularization(theta):
    lamda = 1
    theta_penalty = 0
    for i in theta:
        theta_penalty += lamda*abs(i)
    theta = theta + theta_penalty
    return theta


folds_dictionary = {}
for i in range(1,6):
    folds_dictionary[i] = chose_test_fold(i,independent_features,dependent_features)
#print(folds_dictionary)

# first fold is test
theta_first = get_weights(folds_dictionary[1][0],folds_dictionary[1][1])
est1 = np.dot(folds_dictionary[1][2],np.transpose(theta_first))
real1 = folds_dictionary[1][3]
r_sq1 = r_squared_error(real1,est1)
print("R squared result for first fold",r_sq1)
mse1=mean_squared_error(real1,est1)
print("Mean square Error Result for first fold",mse1)
mae1=mean_absolute_error(real1,est1)
print("Mean absolute Error Result for first fold",mae1)
mape1=mape_error(real1,est1)
print("MAPE Error Result for first fold",mape1)


# second fold is test
theta_second = get_weights(folds_dictionary[2][0],folds_dictionary[2][1])
est2 = np.dot(folds_dictionary[2][2],np.transpose(theta_second))
real2 = folds_dictionary[2][3]
r_sq2 = r_squared_error(real2,est2)
print("R squared result for second fold",r_sq2)
mse2=mean_squared_error(real2,est2)
print("Mean square Error Result for second fold",mse2)
mae2=mean_absolute_error(real2,est2)
print("Mean absolute Error Result for second fold",mae2)
mape2=mape_error(real2,est2)
print("MAPE Error Result for second fold",mape2)


# third fold is test
theta_third = get_weights(folds_dictionary[3][0],folds_dictionary[3][1])
est3 = np.dot(folds_dictionary[3][2],np.transpose(theta_third))
real3 = folds_dictionary[3][3]
r_sq3 = r_squared_error(real3,est3)
print("R squared result for third fold",r_sq3)
mse3=mean_squared_error(real3,est3)
print("Mean square Error Result for third fold",mse3)
mae3=mean_absolute_error(real3,est3)
print("Mean absolute Error Result for third fold",mae3)
mape3=mape_error(real3,est3)
print("MAPE Error Result for third fold",mape3)



# fourth fold is test
theta_fourth = get_weights(folds_dictionary[4][0],folds_dictionary[4][1])
est4 = np.dot(folds_dictionary[4][2],np.transpose(theta_fourth))
real4 = folds_dictionary[4][3]
r_sq4 = r_squared_error(real4,est4)
print("R squared result for fourth fold",r_sq4)
mse4=mean_squared_error(real4,est4)
print("Mean square Error Result for fourth fold",mse4)
mae4=mean_absolute_error(real4,est4)
print("Mean absolute Error Result for fourth fold",mae4)
mape4=mape_error(real4,est4)
print("MAPE Error Result for fourth fold",mape4)



# fifth fold is test
theta_fifth = get_weights(folds_dictionary[5][0],folds_dictionary[5][1])
est5 = np.dot(folds_dictionary[5][2],np.transpose(theta_fifth))
real5 = folds_dictionary[5][3]
r_sq5 = r_squared_error(real5,est5)
print("R squared result for fifth fold",r_sq5)
mse5=mean_squared_error(real5,est5)
print("Mean square Error Result for fifth fold",mse5)
mae5=mean_absolute_error(real5,est5)
print("Mean absolute Error Result for fifth fold",mae5)
mape5=mape_error(real5,est5)
print("MAPE Error Result for fifth fold",mape5)

#Question 3.2

feature_set_independent=[]
for x1 in folds_dictionary[1][2]:
    feature_set_independent.append(x1)
    #print(type(x1))
for x2 in folds_dictionary[1][0]:
    feature_set_independent.append(x2)
feature_set_independent=np.array(feature_set_independent)

feature_set_dependent=[]
for x1 in folds_dictionary[1][3]:
    feature_set_dependent.append(x1)
for x2 in folds_dictionary[1][1]:
    feature_set_dependent.append(x2)
feature_set_dependent=np.array(feature_set_dependent)


#normalization of feature set
columns = []
columns_temp=[]
i=0
for i in range(len(feature_set_independent[1])):
    columns.append(feature_set_independent[:,i])

columns = np.array(columns)
sum_of_columns = []
for i in columns:
   sum_of_columns.append(sum(abs(i)))
for i in range(len(columns)):
    columns[i] = columns[i]/sum_of_columns[i]


#print(theta_first)
estimation = np.matmul(theta_first.transpose(),columns)
#print("Theta firsy",theta_first)
#print("columns",columns)
#print("estimation",estimation)
temp=[]
for i in range(500):
 temp.append(round(estimation[i] - feature_set_dependent[i],2))
estimation2 = temp
estimation2 = np.array(estimation2)
#print(estimation2)
H = np.matmul(2*columns,estimation2)
#print("H değerleri",H)
for i in range(len(H)):
    H[i] = H[i]/sum(H)
#print("H değerleri",H)

lmd=0.01
new_weights=[]
#print("Theta first",theta_first)
for i in range(len(H)):
    if(theta_first[i]>=0):
        new_weights.append(theta_first[i]-0.001*(H[i]+lmd))
    else:
        new_weights.append((theta_first[i]-0.001*(H[i]-lmd)))
#print("New weights",new_weights)


est1_lasso = np.dot(folds_dictionary[1][2],np.transpose(new_weights))
real1_lasso = folds_dictionary[1][3]
r_sq1_lasso = r_squared_error(real1_lasso,est1_lasso)
print("R squared result for first fold_lasso (FIRST FOLD)",r_sq1_lasso)
mse1_lasso=mean_squared_error(real1_lasso,est1_lasso)
print("Mean square Error Result for first fold_lasso (FIRST FOLD)",mse1_lasso)
mae1_lasso=mean_absolute_error(real1_lasso,est1_lasso)
print("Mean absolute Error Result for first fold_lasso (FIRST FOLD)",mae1_lasso)
mape1_lasso=mape_error(real1_lasso,est1_lasso)
print("MAPE Error Result for first fold _lasso (FIRST FOLD) ",mape1_lasso)

est2_lasso = np.dot(folds_dictionary[2][2],np.transpose(new_weights))
real2_lasso = folds_dictionary[2][3]
r_sq2_lasso = r_squared_error(real2_lasso,est2_lasso)
print("R squared result for second fold_lasso (SECOND FOLD)",r_sq2_lasso)
mse2_lasso=mean_squared_error(real2_lasso,est2_lasso)
print("Mean square Error Result for second fold_lasso (SECOND FOLD)",mse2_lasso)
mae2_lasso=mean_absolute_error(real2_lasso,est2_lasso)
print("Mean absolute Error Result for second fold_lasso (SECOND FOLD)",mae2_lasso)
mape2_lasso=mape_error(real2_lasso,est2_lasso)
print("MAPE Error Result for second fold _lasso (SECOND FOLD)",mape2_lasso)

est3_lasso = np.dot(folds_dictionary[3][2],np.transpose(new_weights))
real3_lasso = folds_dictionary[3][3]
r_sq3_lasso = r_squared_error(real3_lasso,est3_lasso)
print("R squared result for third fold_lasso (THIRD FOLD)",r_sq3_lasso)
mse3_lasso=mean_squared_error(real3_lasso,est3_lasso)
print("Mean square Error Result for third fold_lasso (THIRD FOLD)",mse3_lasso)
mae3_lasso=mean_absolute_error(real3_lasso,est3_lasso)
print("Mean absolute Error Result for third fold_lasso (THIRD FOLD)",mae3_lasso)
mape3_lasso=mape_error(real3_lasso,est3_lasso)
print("MAPE Error Result for third fold _lasso (THIRD FOLD)",mape3_lasso)

est4_lasso = np.dot(folds_dictionary[4][2],np.transpose(new_weights))
real4_lasso = folds_dictionary[4][3]
r_sq4_lasso = r_squared_error(real4_lasso,est1_lasso)
print("R squared result for fourth fold_lasso (FOURTH FOLD)",r_sq4_lasso)
mse4_lasso=mean_squared_error(real4_lasso,est4_lasso)
print("Mean square Error Result for fourth fold_lasso (FOURTH FOLD)",mse4_lasso)
mae4_lasso=mean_absolute_error(real4_lasso,est4_lasso)
print("Mean absolute Error Result for fourth fold_lasso (FOURTH FOLD)",mae4_lasso)
mape4_lasso=mape_error(real4_lasso,est4_lasso)
print("MAPE Error Result for fourth fold _lasso (FOURTH FOLD)",mape4_lasso)

est5_lasso = np.dot(folds_dictionary[5][2],np.transpose(new_weights))
real5_lasso = folds_dictionary[5][3]
r_sq5_lasso = r_squared_error(real5_lasso,est5_lasso)
print("R squared result for fifth fold_lasso (FIFTH FOLD)",r_sq5_lasso)
mse5_lasso=mean_squared_error(real5_lasso,est5_lasso)
print("Mean square Error Result for fifth fold_lasso (FIFTH FOLD)",mse5_lasso)
mae5_lasso=mean_absolute_error(real5_lasso,est5_lasso)
print("Mean absolute Error Result for fifth fold_lasso (FIFTH FOLD)",mae5_lasso)
mape5_lasso=mape_error(real5_lasso,est5_lasso)
print("MAPE Error Result for fifth fold _lasso (FIFTH FOLD)",mape5_lasso)




f1=["R-SQ","R-SQ(L1)"]
temp1 = [r_sq1,r_sq1_lasso]
plt.bar(f1,temp1)
plt.title("Fold 1")
plt.show()


f2=["MSE","MSE(L1)"]
temp2 = [mse1,mse1_lasso]
plt.bar(f2,temp2)
plt.title("Fold 1")
plt.show()

f3=["MAE","MAE(L1)"]
temp3 = [mae1,mae1_lasso]
plt.bar(f3,temp3)
plt.title("Fold 1")
plt.show()

f4=["MAPE","MAPE(L1)"]
temp4 = [mape1,mape1_lasso]
plt.bar(f4,temp4)
plt.title("Fold 1")
plt.show()


f1=["R-SQ","R-SQ(L1)"]
temp1 = [r_sq2,r_sq2_lasso]
plt.bar(f1,temp1)
plt.title("Fold 2")
plt.show()


f2=["MSE","MSE(L1)"]
temp2 = [mse2,mse2_lasso]
plt.bar(f2,temp2)
plt.title("Fold 2")
plt.show()

f3=["MAE","MAE(L1)"]
temp3 = [mae2,mae2_lasso]
plt.bar(f3,temp3)
plt.title("Fold 2")
plt.show()

f4=["MAPE","MAPE(L1)"]
temp4 = [mape2,mape2_lasso]
plt.bar(f4,temp4)
plt.title("Fold 2")
plt.show()


f1=["R-SQ","R-SQ(L1)"]
temp1 = [r_sq3,r_sq3_lasso]
plt.bar(f1,temp1)
plt.title("Fold 3")
plt.show()


f2=["MSE","MSE(L1)"]
temp2 = [mse3,mse3_lasso]
plt.bar(f2,temp2)
plt.title("Fold 3")
plt.show()

f3=["MAE","MAE(L1)"]
temp3 = [mae3,mae3_lasso]
plt.bar(f3,temp3)
plt.title("Fold 3")
plt.show()

f4=["MAPE","MAPE(L1)"]
temp4 = [mape3,mape3_lasso]
plt.bar(f4,temp4)
plt.title("Fold 3")
plt.show()


f1=["R-SQ","R-SQ(L1)"]
temp1 = [r_sq4,r_sq4_lasso]
plt.bar(f1,temp1)
plt.title("Fold 4")
plt.show()


f2=["MSE","MSE(L1)"]
temp2 = [mse4,mse4_lasso]
plt.bar(f2,temp2)
plt.title("Fold 4")
plt.show()

f3=["MAE","MAE(L1)"]
temp3 = [mae4,mae4_lasso]
plt.bar(f3,temp3)
plt.title("Fold 4")
plt.show()

f4=["MAPE","MAPE(L1)"]
temp4 = [mape4,mape4_lasso]
plt.bar(f4,temp4)
plt.title("Fold 4")
plt.show()



f1=["R-SQ","R-SQ(L1)"]
temp1 = [r_sq5,r_sq5_lasso]
plt.bar(f1,temp1)
plt.title("Fold 5")
plt.show()


f2=["MSE","MSE(L1)"]
temp2 = [mse5,mse5_lasso]
plt.bar(f2,temp2)
plt.title("Fold 5")
plt.show()

f3=["MAE","MAE(L1)"]
temp3 = [mae5,mae5_lasso]
plt.bar(f3,temp3)
plt.title("Fold 5")
plt.show()

f4=["MAPE","MAPE(L1)"]
temp4 = [mape5,mape5_lasso]
plt.bar(f4,temp4)
plt.title("Fold 5")
plt.show()

