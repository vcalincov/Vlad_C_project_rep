import numpy as np
import pandas as pd
import pathlib
from sklearn import datasets, tree, ensemble, metrics
from sklearn.model_selection import train_test_split
import warnings

attrTags = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol_in_mg/dl',
            'fasting_blood_sugar_>_120_mg/dl', 'resting_electrocardiographic_results', 'maximum_heart_rate_achieved',
            'exercise_induced_angina', 'oldpeak',
            'the_slope_of_the_peak_exercise_ST_segment', 'number_of_major_vessels_(0-3)_colored_by_flouroscopy',
            'thal', 'heart_disease']
#Se construiește o listă de etichete sub formă de șir de caractere.
#Acestea vor fi atributele datelor.

location = pathlib.WindowsPath(r"C:\Users\ssd\PycharmProjects\pythonProject1\heart.dat")
allData = pd.read_csv(location, ' ', names=attrTags)
#Citirea efectuată prin metoda read_csv() din pandas
#Aceasta are si nevoie de poziția fișierului citit, poziție pe care o obținem astfel:
#în location se păstrează prin metoda WindowsPath atribuirea obiectului de tip poziție utilizat în read_csv()
#De menționat este că python nu identifică string-urile din Windows si este necesară conversia prin atașarea caracterului „r”
#înainte de secvența poziției fișierului

entire_data_set = allData.iloc[:, :-1]
entire_label_set = allData.iloc[:, 13]
#Datele sunt citite cu totul în acelși obiect allData
#Se despart datele într-o matrice cu totalitatea atributelor 270x13 și o matrice cu totalitatea etichetelor 270x1

data_set_train, data_set_test, label_set_train, label_set_test = train_test_split(entire_data_set,
                                                                                  entire_label_set,
                                                                                  test_size=0.25)
#împarțim datele de învățare și cele de test cu metoda train_test_split

#împarțire alternativă făcută manual

np.random.shuffle(allData.values)
# training_data_set = entire_data_set[:200]
# training_label_set = entire_label_set[:200]

# test_data_set = entire_data_set[200:270]
# checking_label_set = entire_label_set[200:270]

#perc_of_dim = np.array([1, 7, 10])
perc_of_dim = np.array([0.1, 0.5, 0.8])
in_bag_perc = np.array([0.25, 0.5, 0.85])
#Parametrii ce trebuie variați, băgați în liste de tip numpy:
#perc_of_dim = numărul de dimensiuni luate de random forest
#in_bag_perc = Oout of bag score

RFclf = ensemble.BaggingClassifier(n_estimators=10, oob_score=True)
for i in range(0, 3):
    for j in range (0, 3):
        RFclf.oob_score = in_bag_perc[i]
        RFclf.n_features = perc_of_dim[j]
        #Se parcurg cu două foruri cele două liste și se aplică clasificatorul Random Forest pentru fiecare

        #pentru 10 arbori :
        warnings.filterwarnings("ignore")

        #RFclf.fit(training_data_set, training_label_set)
        #predicted_label_set = RFclf.predict(test_data_set)

        RFclf.fit(data_set_train, label_set_train)
        label_prediction = RFclf.predict(data_set_test)
        #predicțiile etichetelor generate de algoritm

        accuracy = metrics.accuracy_score(label_set_test, label_prediction)
        #calculul preciziei cu metoda.accuracy_score()

        print('OOB = {} no_of_attributes = {} accuracy = {} '.format(RFclf.oob_score, RFclf.n_features, accuracy))
        #afișare
        #print(label_prediction)

    print('\n')

#alternativ se putea face si prin variatia max_features in interiorul clasificatorului random forest
#samples_for_bootstrap = ?

# for i in range(0, 3):
#     for j in range(0, 3):
#         RF1clf = ensemble.BaggingClassifier(n_estimators=10,max_samples=in_bag_perc[i], max_features=perc_of_dim[j], oob_score=True)
#
#         #RF1clf.oob_score = in_bag_perc[i]
#         #Se parcurg cu două foruri cele două liste și se aplică clasificatorul Random Forest pentru fiecare
#
#         #pentru 10 arbori :
#         warnings.filterwarnings("ignore")
#
#         #RFclf.fit(training_data_set, training_label_set)
#         #predicted_label_set = RFclf.predict(test_data_set)
#
#         RF1clf.fit(data_set_train, label_set_train)
#         label_prediction = RF1clf.predict(data_set_test)
#         #predicțiile etichetelor generate de algoritm
#
#         accuracy = metrics.accuracy_score(label_set_test, label_prediction)
#         #calculul preciziei cu metoda.accuracy_score()
#
#         print('no_of_samples = {} no_of_attributes = {} accuracy = {} '.format(RF1clf.max_samples, RF1clf.max_features, accuracy))
#         #afișare
#         #print(label_prediction)
#
#     print('\n')
