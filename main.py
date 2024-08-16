import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime


#%% IMPORT DATA
data = pd.read_excel(r"C:\Users\701540\VS_PY\FINAL_PROJ\FINAL.xlsx")
data.sample(5)
data.info()
data.isnull().sum()
data.columns

#%% DATA PREPROCESSING
data["DOB"]=pd.to_datetime(data["DOB"])

def calculate_age(dob):
    today = datetime.today()
    age = today.year - dob.year - ((today.month, today.day)>(dob.month, dob.day))
    return age

data["AGE"] = data["DOB"].apply(calculate_age)

cols_to_drop = ['YEAR', 'NUMBER','DATE', 'NAME',
       'FATHER/ HUSBAND NAME','PERMENANT ADDRESS',
       'PRESENT ADDRESS', 'PHONE NO','SECTION','DOB',
       'INTERVIEW BASIC', 'REPORTING DATE', 'SEL BRANCH', 'TRN BRANCH',
       'PHOTO','ORGANISATION', 'JOB DESCRIPTION',
       'REASON FOR LEAVING', 'REMARK', 'ECNO','DESIGNATION','LEFTOUT DATE', 'LEFTOUT REASON']

data = data.drop(cols_to_drop, axis=1)
#######################
data.columns = data.columns.str.replace(" ","_")
data["NOF_YEARS"] = data["NOF_YEARS"].replace("-","0").astype(float)
#######################
data["EXPERIENCE_FIELD"] = data["EXPERIENCE_FIELD"].replace("-","NO_EXPERIENCE")
data = data.rename(columns={"WRK._BRANCH": "BRANCH", "WRK.SECTION": "SECTION", "WRK.DESIGNATION":"DESIGNATION"})
data.shape
data.dtypes

print(data.sample(10))

cat_cols = data.select_dtypes(include=["object"]).columns

from sklearn.preprocessing import LabelEncoder
label_encoded = {}
for cols in cat_cols:
    le = LabelEncoder()
    data[cols]= le.fit_transform(data[cols])
    label_encoded = le

data.head()

df = data.copy()

#%% MODEL BUILDING
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

X = df.drop(["LEFTOUT"], axis=1)
X.dtypes
y = df["LEFTOUT"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
sns.displot(x=y_test)
plt.show()
#%% SUPPORT VECTOR MACHINE
svc_model = SVC()
model_svc = svc_model.fit(X_train,y_train)
y_pred_svc = svc_model.predict(X_test)

report_svc = classification_report(y_test,y_pred_svc)
print(report_svc)

#%% DECISION TREE CLASSIFIER
tree = DecisionTreeClassifier()
model_tree = tree.fit(X_train,y_train)
y_pred_tree = tree.predict(X_test)

tree_report = classification_report(y_test, y_pred_tree)
print(tree_report)

#%% RANDOM FOREST CLASSIFIER
forest = RandomForestClassifier(n_estimators=100)
model_forest = forest.fit(X_train,y_train)
y_pred_forest = forest.predict(X_test)

forest_report = classification_report(y_test,y_pred_forest)
print(forest_report)

#%% K NEAREST NEIGHBOUR
knn = KNeighborsClassifier()
model_knn = knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)

knn_report = classification_report(y_test, y_pred_knn)
print(knn_report)


#%% on building different model we came to an conclusion that, RANDOMFOREST CLASSIFIER has higher accuracy among the other so we save that model.
import joblib
joblib.dump(model_forest,"random_forest.joblib", compress=("zlib",3))
joblib.dump(model_knn, "KNN.joblib", compress=3)
data.dtypes

