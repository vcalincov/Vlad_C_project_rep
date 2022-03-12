
import numpy as np
import pandas as pd
import pathlib
import warnings
from sklearn.model_selection import train_test_split

def logistic_fct(x):
    return 1.00 / (1 + np.exp(-x)) 

def dL_dx(x):
    return logistic_fct(x) * (1 - logistic_fct(x))


allAtributes = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes',	'Turbidity', 'Potability']
wdwsPath = pathlib.WindowsPath(r'D:\Transfer_Laptop\Facultate\Anul 3\SEM2\RNSF\P_RNSF_MLP\water_potability.csv')


allData = pd.read_csv(wdwsPath,names = allAtributes)#toate datele
#np.random.shuffle(allData.values)
written_labels = np.asarray(allData.iloc[0,:])


############# Solving Bad Data ############
medianValue = np.zeros(len(allData.iloc[0,:]))

for j in range (len(allData.iloc[1,:])):
    medianValue[j] = allData.iloc[1::,j].median()
    allData.iloc[:,j].fillna(medianValue[j], inplace = True)

############# Spliting labels from Info####
allLabels = allData.iloc[1::,-1]
allAttrValues = allData.iloc[1::,:-1]


#data split
attrSet_train, attrSet_test, labelSet_train, labelSet_test = train_test_split(allAttrValues, allLabels,test_size=0.25) 

attrSet_train = np.asarray(attrSet_train)
attrSet_test = np.asarray(attrSet_test)
labelSet_train = np.asarray(labelSet_train)
labelSet_test = np.asarray(labelSet_test)

############# Training init ################

learning_rate = 0.1 

inputDim = len(allAttrValues.iloc[0,:])
hiddenDim = inputDim + 1; #we add a hidden layer bias node
noIterations_training = 1;

input_hiddenW = np.random.uniform(-1,1, (inputDim, hiddenDim)) #input-output weights constitute a matrix with regards to the dimensionality of the input data and hidden layer passed data, it resembles a graph
hidden_outW = np.random.uniform(-1,1, hiddenDim) # output size is 1, we only have 1 output node dimensionaly so we get an array for the hidden layer to output weights

hidden_preA = np.zeros(hiddenDim)
hidden_postA = np.zeros(hiddenDim)


#### Tr_data String To Float Conversion ####

for i in range(len(attrSet_train[:,0])):
    for j in range(len(attrSet_train[0,:])):
        attrSet_train[i,j] = float(attrSet_train[i,j])

for i in range (len(labelSet_train[:])):
    labelSet_train[i] = float(labelSet_train[i])

############ Training Process ##############
############ Feed forward ##################
for iterations in range(noIterations_training):
    for i in range(len(attrSet_train[1::,0])):
        for hNode in range (hiddenDim):
            hidden_preA[hNode] = np.dot((attrSet_train[i,:]), input_hiddenW[:,hNode])
            hidden_postA[hNode] = logistic_fct(hidden_preA[hNode])

        out_preA = np.dot(hidden_postA, hidden_outW)
        out_postA = logistic_fct(out_preA)

        Err = out_postA - labelSet_train[i]


##############Back Propagation##############

        for hNodebp in range(hiddenDim):
            bpErr = Err * dL_dx(out_preA) #error signal is propagated backwards through the network
            hoGradient = bpErr  * out_postA
            hidden_outW[hNodebp] = hidden_outW[hNodebp] - learning_rate * hoGradient 

            for iNode in range(inputDim):
                input = attrSet_train[i,iNode] #value
                ihGradient = bpErr * dL_dx(hidden_preA[hNode]) * hidden_outW[hNodebp] * input
                input_hiddenW[iNode, hNodebp] = input_hiddenW[iNode, hNodebp] - learning_rate * ihGradient

########## Validation and Accuracy #########


### Test_data String To Float Conversion ###

for i in range(len(attrSet_test[:,0])):
    for j in range(len(attrSet_test[0,:])):
        attrSet_test[i,j] = float(attrSet_test[i,j])

for i in range(len(labelSet_test[:])):
    labelSet_test[i] = float(labelSet_test[i])

final_results = attrSet_test
no_correctClassification = 0

outputResult = np.zeros(len(labelSet_test))
for i in range(len(attrSet_test[0,:])):
    for hNode in range(hiddenDim):
        hidden_preA[hNode] = np.dot((attrSet_test[i,:]), input_hiddenW[:,hNode])
        hidden_postA[hNode] = logistic_fct(hidden_preA[hNode])

    out_preA = np.dot(hidden_postA, hidden_outW)
    out_postA = logistic_fct(out_preA)

    if out_postA > 0.5: output = 1
    else: output = 0
    
    np.append(outputResult,output)
    if output == labelSet_test[i]: no_correctClassification += 1

final_results = np.c_[final_results,outputResult]
final_results = np.insert(final_results,[0] ,written_labels, axis = 0)

accuracy = no_correctClassification * 100/len(attrSet_test[0,:])

print(final_results)
print('Accuracy score is:')
print(accuracy)
        