import pandas as pd
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv(r"C:\Users\INDRONIIL\Downloads\financial_fraud_detection_dataset.csv")
df.head()

df.info()

df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
df['timestamp'] = df['timestamp'].dt.strftime('%d:%m:%Y')

df['sender_account'] = pd.to_numeric(df['sender_account'],errors = 'coerce')
df['receiver_account'] = pd.to_numeric(df['receiver_account'],errors = 'coerce')

df['is_fraud'] = df['is_fraud'].map({True:1, False:0})

X = df.drop(['is_fraud','timestamp','transaction_id','transaction_type','merchant_category','location',
             'device_used','fraud_type','time_since_last_transaction','spending_deviation_score',
             'velocity_score','payment_channel','ip_address','device_hash'],axis=1)
y = df[['is_fraud']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)

rmodel = RandomForestClassifier(n_estimators=100)
rmodel.fit(X_train,y_train)

dmodel = DecisionTreeClassifier(max_depth=10)
dmodel.fit(X_train,y_train)

prediction1 = rmodel.predict(X_test)
prediction2 = dmodel.predict(X_test)

print("Accuracy of RandomForest:",accuracy_score(y_test,prediction1))
print("Accuracy of DecisionTree:",accuracy_score(y_test,prediction2))

print("Confusion matrix of RandomForest:\n",confusion_matrix(y_test,prediction1))
print("Confusion matrix of DecisionTree:\n",confusion_matrix(y_test,prediction2))

print("Classification report of RandomForest:\n",classification_report(y_test,prediction1))
print("Classification report of DecisionTree:\n",classification_report(y_test,prediction2))