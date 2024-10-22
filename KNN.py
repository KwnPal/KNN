import math as m
import numpy as np
import statistics as st
import time

def KNN(pattern, classes, points, k): #pattern-->a set of patterns,point-->the point that we want to classify
    print("Using KNN")             #classes-->the categories of each pattern k--> The number of closest neighbors
    mainclasses=categories(classes) #stores categories
    classes=NumericClasses(classes) #converts categories to numeric
    results=[]
    for j in range(len(points)):
        neighbors=[] 
        maxcategory=[] # votes of categories
        dist=[]#An array type[category,distance]
        for i in range(len(pattern)):
            dist.append((classes[i],euclidian_distance(points[j],pattern[i])))
        dist.sort(key=lambda dist: dist[1])
        for i in range(k):
            neighbors.append(dist[i][0])
        # We find the number of each category
        for i in range(len(mainclasses)):
            maxcategory.append(neighbors.count(i))
        indexofmax=np.argmax(maxcategory)
        results.append(mainclasses[indexofmax])
    return results


def CB_KNN(pattern,classes,points): # Pattern--> a set of patterns,point-->the point that we want to classify
    print("Using CB_KNN")           # Classes--> the categories of each pattern 
    mainclasses=categories(classes) # Stores categories
    classes=NumericClasses(classes) # Converts categories to numeric
    results=[]
    for numofpoints in range(len(points)):
        neighbors=[]# K closest neighbors
        dist=[]# An array type[category, distance]
        harmonic=[]
        for i in range(len(pattern)):
            dist.append((classes[i],euclidian_distance(points[numofpoints],pattern[i])))
        dist.sort(key=lambda dist: dist[1])

        k=max_k(dist,mainclasses)
        for i in range(len(mainclasses)):
            temp_k=0
            for j in range(len(dist)):
                if(dist[j][0] == i and dist[j][1] != 0):
                    neighbors.append(dist[j][1])
                    temp_k+=1
                if(temp_k == k):
                    break
#--------Calculate the Harmonic Mean of the K elements of each class--------
        sum=0
        tmp=k
        for i in range(len(neighbors)):
            sum+=(1 / neighbors[i])
            if (i+1) == tmp:
                sum=k/sum
                tmp+=k
                harmonic.append(sum)
                sum=0
#----------------------------------------------------------------------------
        indexofmin=np.argmin(harmonic)
        results.append(mainclasses[indexofmin])	
    return results

#--------Function that calculates the euclidian distance of 2 vectors--------- 
def euclidian_distance(startpoint,endpoint):
    distance=0
    for i in range(len(startpoint)):
        distance+=(startpoint[i] - endpoint[i]) ** 2 #(x-y)^2
    return m.sqrt(distance) #sqrt(x-y)^2

#--------Function that converts str classes to numeric---------         
def NumericClasses(classes):
    mainclasses=categories(classes)
    for i in range(len(classes)):
        classes[i]=mainclasses.index(classes[i])
    return classes

#--------Function that returns the categories of the dataset---------
def categories(classes):
    mainclasses=[]
    classes=np.unique(classes)
    for i in classes:
        mainclasses.append(i)
    return mainclasses

#--------Function that calculates the K that has the best DC(Degree of Certainty)---------
def max_k(distance,mainclasses):
    k=len(mainclasses)
    if(k >= 10):
        k=5
    DC=[]
    k_array=[]
    while(k<=10):
        maxvalue=0 #index that shows the biggest DC
        maxcategory=[] #store the max vote 
        neighbors=[]#store the k closest neighbors
        
        for i in range(k):#Add k closest neighbors 
            neighbors.append(distance[i][0])

        for i in range(len(mainclasses)):#find the vote of each category
            maxcategory.append(neighbors.count(i))

        maxvalue=max(maxcategory)
        DC.append(maxvalue/k)
        k_array.append(k)
        k+=1
    
    maxvalue=np.argmax(DC)

    return k_array[maxvalue]
