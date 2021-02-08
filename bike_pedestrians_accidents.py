# Import required packages:


from matplotlib import pyplot as plt
from scipy.stats import binom
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression

from datetime import datetime as dt
import time
from sklearn.model_selection import train_test_split as split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer, OneHotEncoder, KBinsDiscretizer, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import math

warnings.filterwarnings('ignore')
%matplotlib inline

#Loading datast:

data = pd.read_csv("D:\\Analytics\\BikePedCrash.csv", sep= ',')

df = pd.DataFrame(data)

# Display the number of rows and columns
df.shape
type(df)

# Print some values and their frequencies
print(df['DrvrAlcDrg'].value_counts())
print(df['NumPedsAin'].value_counts())
print(df['NumPedsKil'].value_counts())
print(df['CrashType'].value_counts())
print(df['PedAlcDrg'].value_counts())

# Replace the (.) value with nan values
df['DrvrAlcDrg'].replace(['.'], np.nan, inplace=True)
df['PedAlcDrg'].replace(['.'], np.nan, inplace=True)

#Removing columns with more than half blank values:
df1 = df.drop(['NumPedsAin', 'NumPedsBin', 'NumPedsCin', 'NumPedsKil', 'NumPedsNoi', 
               'NumPedsTot', 'NumPedsUin', 'X', 'Y', 'OBJECTID', ], axis = 1)

print(df1.head())
df1.shape

# Checking nulls:
print(df1.isnull().sum())  

# converting columns to categorial values:

print(df1['DrvrAlcDrg'])
print(df1['PedAlcDrg'])

df1.DrvrAlcDrg = pd.Categorical(df1.DrvrAlcDrg)
df1.PedAlcDrg = pd.Categorical(df1.PedAlcDrg)

# Replacing NULL  values in catogricai Columns using mode

df1 = pd.DataFrame(df)

mode1= df1["DrvrAlcDrg"].mode().values[0]
mode2= df1["PedAlcDrg"].mode().values[0]

print(mode1)
print(mode2)

# replacing null values with mode

df1["DrvrAlcDrg"]= df1["DrvrAlcDrg"].replace(np.NaN,mode1)

df1["PedAlcDrg"]= df1["PedAlcDrg"].replace(np.NaN,mode1)


print(df1.isnull().sum())

# checking duplicates values:
    
duplicate = df1.duplicated()
print(duplicate.sum())
df1[duplicate]

# removing extra characters from some columns:
    # removing '+' character from columns
    
df1["PedAge"]= df1["PedAge"].replace('70+', 70)
df1["DrvrAge"]= df1["DrvrAge"].replace('70+', 70)
df1["DrvrAgeGrp"]= df1["DrvrAgeGrp"].replace('70+', 70)

# Converting columns to factors:

df1.DrvrVehTyp = pd.Categorical(df1.DrvrVehTyp)
df1.DrvrVehTyp = pd.Categorical(df1.CrashType)
df1.AmbulanceR = pd.Categorical(df1.AmbulanceR)
df1.CrashAlcoh = pd.Categorical(df1.CrashAlcoh)
df1.CrashDay = pd.Categorical(df1.CrashDay)
df1.CrashGrp = pd.Categorical(df1.CrashGrp)
df1.CrashLoc = pd.Categorical(df1.CrashLoc)
df1.CrashMonth = pd.Categorical(df1.CrashMonth)
df1.CrashSevr = pd.Categorical(df1.CrashSevr)
df1.Developmen = pd.Categorical(df1.Developmen)
df1.DrvrInjury = pd.Categorical(df1.DrvrInjury)
df1.DrvrRace = pd.Categorical(df1.DrvrRace)
df1.DrvrSex = pd.Categorical(df1.DrvrSex)
df1.HitRun = pd.Categorical(df1.HitRun)
df1.LightCond = pd.Categorical(df1.LightCond)
df1.Locality = pd.Categorical(df1.Locality)

df1['CrashType'].astype(str).astype(int)

print(df1.HitRun)
df1.dtypes

### Defining function for converting datatypes from obejct to categorical:###

for col_name in df1.columns:
    if(df1[col_name].dtype == 'object'):
        df1[col_name]= df1[col_name].astype('category')
        df1[col_name] = df1[col_name].cat.codes

## checking correlation graph

corrmat = df1.corr() 

f, ax = plt.subplots(figsize =(15, 14)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 

#Saving new dataframes after modifying and removing blanks:

sav_df = df1.to_csv ('D:\Analytics\BikePedCrash2.csv', 
                     index = False, header=True)

##################### Working on updated dataset:#########################

data1 = pd.read_csv("D:\\Analytics\\BikePedCrash2.csv")

df2 = pd.DataFrame(data1)

print(df2.isnull().sum())

print(df2.head())
df2.shape
df2.dtypes

# checking outliers:

df2.boxplot(column = ["CrashHour"])
plt.show

df2.boxplot(column = ["PedAge"])
plt.show

df2.boxplot(column = ["DrvrAge"])
plt.show

# removing outliers:
col = df2[columns]
    
def remove_outlier(col):
    
    
    sorted(col)
    Ql, Q3=col.quantile([0.25,0.75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR) 
    upper_range= Q3*(1.5 * IQR) 
    return lower_range,  upper_range

# Exploratory Data Analysis:
  
df2 ['City'].describe()
df2 ['CrashType'].describe()
df2 ['County'].describe()
df2 ['CrashDay'].describe()
df2 ['CrashMonth'].describe()
df2 ['Developmen'].describe()
df2 ['DrvrAgeGrp'].describe()
df2 ['DrvrSex'].describe()
df2 ['DrvrAge'].describe()   #found the 999 values which means data is unknown
df2 ['Workzone'].describe()  # no workzone has more frequency

df2.dtypes

# finding Correlation;

corr = df2.corr()
print(corr)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df2.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df2.columns)
ax.set_yticklabels(df2.columns)
plt.show()

# Correlation Matrix

corrmat = df2.corr() 

f, ax = plt.subplots(figsize =(15, 14)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1) 

# Grid Correlation Matrix

corrmat = df2.corr() 
  
cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1); 
plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 

# diffrent size of heat map;

plt.figure(figsize=(20,20))
sns.heatmap(df2.corr(),linewidths=.5,cmap="YlGnBu")
plt.show()

############################ making model #############################

#check whether the normalized data has a mean of zero and a standard deviation of one.

x = df2
np.mean(x),np.std(x)

# again converting some objects in factors

for col_name in x.columns:
    if(x[col_name].dtype == 'object'):
        x[col_name]= x[col_name].astype('category')
        x[col_name] = x[col_name].cat.codes

x.dtypes

print(x.CrashSevr)

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

normalised_df2 = pd.DataFrame(x,columns=x.columns)
normalised_df2.head()

scaler = MinMaxScaler(x)
print(scaler)

################# Logistic regression model
    
# Assign the data
lrdf = df2

# Set the target for the prediction
target='CrashSevr'

# Create arrays for the features and the response variable

# set X and y
y = lrdf[target]
X = lrdf.drop(target, axis=1)

# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)

# Logistic regression
lr = LogisticRegression(random_state=0)
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Initialize an empty list for the accuracy for each algorithm
accuracy_lst=[]

# Append to the accuracy list
accuracy_lst.append(acc)

print("[Logistic regression algorithm] accuracy_score: {:.3f}.".format(acc))


################################ Random Forest algorithm

#Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
accuracy_lst.append(acc)

# Model Accuracy, how often is the classifier correct?
print("[Randon forest algorithm] accuracy_score: {:.3f}.".format(acc))

feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features
k=15
sns.barplot(x=feature_imp[:15], y=feature_imp.index[:k])

# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# List top k important features
k=20

feature_imp.sort_values(ascending=False)[:k]



