import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report

# load dataset
df = pd.read_excel("indian_flight_delay_realistic.xlsx")

print(df.head())
print(df.info())

df_model = df[[
    'Airline',
    'Origin',
    'Destination',
    'Weather_Delay',
    'Reactionary_Delay',
    'ATC_Delay',
    'Slot_Delay',
    'Technical_Delay',
    'Crew_Delay',
    'Holding_Delay',
    'Turnaround_Time',
    'Cancelled',
    'Diverted',
    'Total_Delay'
]]

df_model = pd.get_dummies(df_model, columns=['Airline','Origin','Destination'])

X = df_model.drop("Total_Delay", axis=1)
y = df_model["Total_Delay"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model=RandomForestRegressor(n_estimators=200, random_state=42)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("R2 Score:", r2)


importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importance.head(10))

df["Delayed"]= (df["Total_Delay"]>30).astype(int)
print(df[["Total_Delay","Delayed"]].head())

df_class=df.drop(columns=[
    'Weather_Delay',
    'Reactionary_Delay',
    'ATC_Delay',
    'Slot_Delay',
    'Technical_Delay',
    'Crew_Delay',
    'Holding_Delay',
    'Total_Delay'
])

df_class = df_class.drop(columns=[
    'Aircraft_ID',
    'Flight_Number',
    'Scheduled_Departure',
    'Actual_Departure',
    'Actual_Arrival',
    'Diversion_Airport'
])

df_class=pd.get_dummies(df_class,columns=['Airline','Origin','Destination','Origin_Weather','Dest_Weather'])

X=df_class.drop(["Delayed"],axis=1)
y=df_class["Delayed"]

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_balanced, y_balanced = smote.fit_resample(X, y)

print("Before SMOTE:")
print(y.value_counts())

print("After SMOTE:")
print(y_balanced.value_counts())

X_train,X_test,y_train,y_test=train_test_split(X_balanced,y_balanced,test_size=0.2,random_state=42)

model_clf=RandomForestClassifier(n_estimators=200, random_state=42)

model_clf.fit(X_train,y_train)

y_pred=model_clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)



                ##############################################################

import joblib

joblib.dump(model_clf, "flight_delay_model.pkl")

joblib.dump(X.columns, "model_features.pkl")
