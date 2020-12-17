import numpy as np
from matplotlib.image import imread
import os
from matplotlib import pyplot as plt

print(os.getcwd())
print(os.chdir("/Users/cumak/PycharmProjects/untitled/hw2/Van_gogh"))

all_pictures= []
names=[]

for file in os.listdir():
    names.append(file)
    temp = imread(file)
    all_pictures.append(temp)

#print(names)

grayscale = [128,128,128]
grayscale = np.array(grayscale)

"""for images in all_pictures:
    if(images.ndim==2):
        np.concatenate()"""
x=0



for images in all_pictures:
   if images.ndim == 2:
       temp= np.append(images,images)
       temp = np.append(temp, images)
       all_pictures[x]=temp.reshape(64,64,3)
       #print(x)
       #print(names[x])
       #print(images.shape)
   x = x+1


noisy_dataset = np.array(all_pictures)


"""data2 = Image.fromarray(all_pictures[35])
data2.save("deneme3.jpg")"""

temp=0
array_x = []
for x in all_pictures:
    all_pictures[temp] = x.reshape(4096,3)
    array_x.append(all_pictures[temp])
    temp = temp + 1

array_x = np.array(array_x)
#print(array_x.shape)

#print(array_x[0].shape)
all_pictures = np.array(all_pictures)

#print("deneme",all_pictures.shape)

for i in range (3):
    U,S,VT = np.linalg.svd(all_pictures[:,:,i],full_matrices=False)
    S = np.diag(S)
    #print(S)
    temp = []
    temp2=[]
    a=0
    for x in range (877):
        for y in range(877):
             if(x==y):
                temp.append(S[x][y])
                if (a >= 10):
                    pass
                else:
                    temp2.append(S[x][y])
        a = a+1
        if(a==100):
          break

    temp = np.array(temp,dtype=float)
    #print(temp)
    f=[]
    for x in range(1,101):
        f.append(x)
    plt.bar(f,temp)
    plt.show()
    #print(i)
    print(i+1,". proportion of variance",sum(temp2))


#Question 1.2


def noisy(image):
      row,col,ch= image.shape
      mean = np.mean(image)
      var = np.var(image)
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + 0.01*gauss
      return noisy

images_with_noise=[]

for i in noisy_dataset:
    images_with_noise.append(noisy(i))


temp=0
for x in images_with_noise:
    images_with_noise[temp] =images_with_noise[temp].reshape(4096,3)
    temp = temp + 1

images_with_noise = np.array(images_with_noise)
#print(images_with_noise.shape)


for i in range (3):
    U,S,VT = np.linalg.svd(images_with_noise[:,:,i],full_matrices=False)
    S = np.diag(S)
    #print(S)
    temp = []
    temp2=[]
    a=0
    for x in range (877):
        for y in range(877):
             if(x==y):
                temp.append(S[x][y])
                if (a >= 10):
                    pass
                else:
                    temp2.append(S[x][y])
        a = a+1
        if(a==100):
          break
    temp = np.array(temp,dtype=float)
    print(i+1,". proportion of variance with noise",sum(temp2))


#first = noisy(all_pictures[0])
#print(first)
