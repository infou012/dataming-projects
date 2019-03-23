# This project aims to help the prediction of breast cancer

from math import sqrt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
#
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#
from scipy.stats import ttest_ind

import numpy as np
import pylab as plt

def moyenne(liste):
	moyenne = 0.0
	for x in liste:
		moyenne = moyenne + x
	
	moyenne = moyenne/len(liste)
	return moyenne

def ecartType(liste):	
	variance = 0.0
	moy = moyenne(liste)
	print("MOYENNE    :"+str(moy))
	for x in liste:
		variance = variance + (x-moy)**2
	
	variance = variance/(len(liste)-1)
	ecartType = sqrt(variance)
	print("ECART TYPE :"+str(ecartType))


data_names = 	["Clump Thickness",
				"Uniformity of Cell Size",
				"Uniformity of Cell Shape",
				"Marginal Adhesion",
				"Single Epithelial Cell Size",
   				"Bare Nuclei",
   				"Bland Chromatin",
   				"Normal Nucleoli",
   				"Mitoses"]

data_input = []
data_res = []

path = "breast-cancer-wisconsin/breast-cancer-wisconsin.data.txt"

f = open(path,'r')
for line in f:
	data = line.split(",")
	if data[6] == "?":
		continue
	data[10] = float(data[10][0])
	for i in range(len(data)):
		data[i] = float(data[i])
	data_res.append(data.pop(10))
	data.pop(0)
	data_input.append(data)
f.close()

data_input = np.array(data_input)
data_res   = np.array(data_res)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_input, data_res, test_size = .14)

#on se base sur les 50 plus proches voisins
#model = KNeighborsClassifier(n_neighbors = 1)
#model.fit(Xtrain, Ytrain)
#predictions = model.predict(Xtest)

#accuracy = (Ytest == predictions).mean()

#for cl in range(5):
#	if cl%2 == 1 or cl == 0: 
#		continue
#	idx = Ytest == cl
#	acc = (predictions[idx] == cl).mean()
#	print 'Cas %d : %2.2f%%' % (cl, acc*100)

#proba de confondre une tumeur benigne (2) avec une tumeur maligne (4)

#predits4pour2 = [True if yt == 4 and pred == 4 else False
#				for (yt, pred) in zip(Ytest, predictions) ]

#X4pour2 = Xtest[predits4pour2,]

#print "tableau de %d listes" % len(X4pour2)
#print "ces listes contiennent %d elements" % len(X4pour2[0])
#print X4pour2

folds = StratifiedKFold(Ytrain, n_folds = 7)

accuracy = np.zeros((7,100))
valeurs_possibles = range(1,101)
i = 0
j = 1
for train, test in folds:
	xtrain = Xtrain[train,]
	xtest = Xtrain[test,]
	ytrain = Ytrain[train]
	ytest = Ytrain[test]
	j =  0
	for k in valeurs_possibles:
		model = KNeighborsClassifier(n_neighbors = k)
		model.fit(xtrain, ytrain)
		predictions = model.predict(xtest)
		accuracy[i, j] = (ytest == predictions).mean()
		j += 1
	i += 1

accuracy_moyenne = accuracy.mean(0)
#plt.plot(valeurs_possibles,accuracy_moyenne, linewidth = 2)
#plt.show()

meilleurk = valeurs_possibles[np.argmax(accuracy_moyenne)]

print "le meilleur nombre de voisin est %d" % meilleurk

model = KNeighborsClassifier(n_neighbors = meilleurk)
model.fit(Xtrain,Ytrain)
predictions = model.predict(Xtest)
print 'Performance finale : %2.2f%%' % ((predictions == Ytest).mean() * 100)

################
print "methode \"SVC\""

model = SVC(C=1, kernel = 'linear')
model.fit(Xtrain,Ytrain)
predictions = model.predict(Xtest)
print 'Performance globale : %2.2f%%' % ((predictions == Ytest).mean() * 100)


################
print "methode \"random forest\""

model = RandomForestClassifier(n_estimators = 100,
criterion = "gini", max_features = "sqrt")
model.fit(Xtrain,Ytrain)

predictions = model.predict(Xtest)
print 'Performance globale : %2.2f%%' % ((predictions == Ytest).mean() * 100)

################
#9 correspond au nombre de champs des listes contenues dans Xtest
print "\n\nTest de l'importance de chaque champs dans la prediction\n"
for i in range(9):
	val,p = ttest_ind(Xtest[Ytest==2,i],  Xtest[Ytest==4,i])
	print "%s :" % data_names[i]
	print '\tt statistic: %f' % val
	print '\tp value: %f' % p



#prediction pour les donnees dans le fichier

path_data_test = "cancer_prediction_test.txt"

df = open(path_data_test,'r')
dataset_test = []
for line in df:
	data_test = line.split(",")
	for i in range(len(data_test)):
		data_test[i] = float(data_test[i])
	dataset_test.append(data_test)
df.close()

dataset_test = np.array(dataset_test)
predictions = model.predict(dataset_test)
if(len(predictions)>0):
	print "prediction pour les donnees dans le fichier"
	print predictions
