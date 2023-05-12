import pandas as pd
from KNN import KNN,CB_KNN
import csv
import numpy as np
from sklearn.model_selection import KFold
import time





def main():
    datasets_names=['Datasets\iris.csv','Datasets\Bupa.csv','Datasets\Breast-Cancer.csv','Datasets\Heart.csv','Datasets\Vehicle.csv','Datasets\Housing.csv']
    for i in range(len(datasets_names)):
        dataset = pd.read_csv (datasets_names[i])
        dataset=dataset.iloc[:,:].values
        if(datasets_names[i]== 'Datasets\Housing.csv'):
            dataset=Housing(dataset)
       
        k_folds_cross_validation(dataset,datasets_names[i],1)
    # dataset = pd.read_csv ('Datasets\Bupa.csv')
    # dataset=dataset.iloc[:,:].values
    # k_folds_cross_validation(dataset,'heart',1)



def k_folds_cross_validation(dataset,dataset_name,repeats):
    Final=[]
    for numofrepeat in range(repeats):
        temp_results=[]
        results=[]
        size=len(dataset)
        kf = KFold(10,random_state=1,shuffle=True)
        for i, (train_rows, test_rows) in enumerate(kf.split(dataset)):
            data_test = []#test patterns
            data_test_class=[]
            times=0
            for row in test_rows:
                data_test.append(dataset[row][0:-1])
                data_test_class.append(dataset[row][-1])
            data_train = []
            data_train_class=[]
            for row in train_rows:
                data_train.append(dataset[row][0:-1])#train patterns
                data_train_class.append(dataset[row][-1])

            start=time.time()
            temp_results=CB_KNN(data_train,data_train_class,data_test)
            end=time.time()

            times=end-start#time needed to find the classes for len(dataset) test samples
            results.append(score(temp_results,data_test_class))  
        Final.append(sum(results)/len(results))
        # print("FINAL {:.2f}".format(Final[numofrepeat]))
    print("The score on the "+dataset_name+" dataset is: {:.2f}".format(sum(Final)/len(Final))," with time : {:.2f}".format(times)," sec")


def Housing(dataset):
    for i in range(len(dataset)):
        if dataset[i][-1]<12.5:
            dataset[i][-1]=1
        elif dataset[i][-1]>=12.5 and dataset[i][-1]<25:
            dataset[i][-1]=2
        elif dataset[i][-1]>=25 and dataset[i][-1]<37.5:
            dataset[i][-1]=3
        elif dataset[i][-1]>=37.5:
            dataset[i][-1]=4
    return dataset




def score(results,test_results):
    size=len(results)
    counter=0
    for i in range(size):
        if(results[i]==test_results[i]):
            counter+=1
    endscore=counter/size
    return endscore
   


if __name__ == "__main__":
    main()
