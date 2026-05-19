import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,precision_score,f1_score,recall_score,classification_report

fraud = np.array([
    [2500, 10, 0],
    [4000, 2, 1],
    [3500, 21, 0],
    [5000, 11, 1],
    [2300, 5, 0],
    [6000, 12, 1],
    [3000, 4, 1],
    [4500, 21, 1],
    [2800, 20, 0],
    [5554, 16, 1]
])

labels = np.array([1,0,1,0,1,0,1,0,0,1])

X_train,X_test,y_train,y_test = train_test_split(fraud,labels,test_size=0.2,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train,y_train)

prediction = model.predict(X_test)
print(prediction)
accuracy = model.score(X_test,y_test)
accuracy

print("Confusion matrix:\n",confusion_matrix(y_test,prediction))

print("precision_score:\n",precision_score(y_test,prediction))
print("f1_score:\n",f1_score(y_test,prediction))
print("recall_score:\n",recall_score(y_test,prediction))

modell = DecisionTreeClassifier()
modell.fit(X_train,y_train)

prediction1 = modell.predict(X_test)
prediction1

accuracy = modell.score(X_test,y_test)
accuracy
print("Confusion matrix:\n",confusion_matrix(y_test,prediction1))
print("precision_score:\n",precision_score(y_test,prediction))
print("f1_score:\n",f1_score(y_test,prediction))
print("recall_score:\n",recall_score(y_test,prediction))