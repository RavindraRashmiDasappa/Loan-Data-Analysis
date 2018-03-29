
# coding: utf-8

# <h2>Reading Data</h2>

# In[487]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

data = pd.read_csv("/Users/rahmi/Desktop/AnalyticsVidya/train.csv")
print(data.head(5))


# In[488]:


data.info()


# In[489]:


data.describe().T


# In[490]:


data.isnull().sum()


# In[491]:


data.shape


# In[492]:


data["Gender"] = data["Gender"].fillna("Other")


# In[493]:


data["Married"] = data["Married"].fillna("Other")


# In[494]:


#data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])


# In[495]:


data.loc[data['Dependents'] == '0', 'Dependents'] = '<=3'
data.loc[data['Dependents'] == '1', 'Dependents'] = '<=3'
data.loc[data['Dependents'] == '2', 'Dependents'] = '<=3'
data.loc[data['Dependents'] == '3+', 'Dependents'] = '>3'

data['Dependents'] = data['Dependents'].fillna(data['Dependents'].mode()[0])


# In[496]:


data['Self_Employed'] = data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])


# In[497]:


median_value= data['LoanAmount'].median()
data['LoanAmount']= data['LoanAmount'].fillna(median_value)


# In[498]:


data.loc[data['Loan_Amount_Term'] == 180, 'Loan_Amount_Term_Cat'] = '6months'
data.loc[data['Loan_Amount_Term'] == 240, 'Loan_Amount_Term_Cat'] = '8months'
data.loc[data['Loan_Amount_Term'] == 360, 'Loan_Amount_Term_Cat'] = '12months'

data['Loan_Amount_Term_Cat'] = data['Loan_Amount_Term_Cat'].fillna(data['Loan_Amount_Term_Cat'].mode()[0])


# In[499]:


data['Credit_History'] = data['Credit_History'].astype('category')


# In[500]:


data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])


# In[501]:


data.info()


# In[502]:


data.isnull().sum()


# In[503]:


data = data.drop(['Loan_Amount_Term'], axis = 1)


# In[504]:


data.skew()


# In[505]:


#,'LoanAmount'
skew_col = data.loc[:,['ApplicantIncome','CoapplicantIncome']]
skew_col = pd.DataFrame(skew_col)
skewness = skew_col.skew()


# In[506]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    data[feat] = boxcox1p(data[feat], lam)


# In[507]:


data.skew()


# In[508]:


#Normalizing the data using MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

#sc = StandardScaler()
#data.iloc[:,6:9] = sc.fit_transform(data.iloc[:,6:9])
#print(data.iloc[:,6:9])

#Normalizing the data using MinMaxScaler
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(data.iloc[:,6:9])
data.iloc[:,6:9] = pd.DataFrame(np_scaled, columns = data.iloc[:,6:9].columns)
data.shape


# In[509]:


data.head(5)


# In[510]:


#data = data.loc[(data['ApplicantIncome'] != 51763 ) & data['LoanAmount'] != 700.0] 
#LP001585 LP002101 LP002317 LP002949
#Deleting outliers
data = data.drop(data[(data['Loan_ID'] == 'LP001585' )].index)
print(data.shape)


# In[511]:


#Deleting outliers
data = data.drop(data[(data['Loan_ID'] == 'LP002101' )].index)
print(data.shape)


# In[512]:


#Deleting outliers
data = data.drop(data[(data['Loan_ID'] == 'LP002317' )].index)
print(data.shape)


# In[513]:



#Deleting outliers
data = data.drop(data[(data['Loan_ID'] == 'LP002949' )].index)
print(data.shape)


# In[514]:


fig, ax = plt.subplots()
ax.scatter(x = data['ApplicantIncome'], y = data['LoanAmount'], alpha = 0.5)
plt.ylabel('LoanAmount', fontsize=13)
plt.xlabel('ApplicantIncome', fontsize=13)
plt.show()


# In[515]:


#Deleting outliers
#data_1 = data.drop(data[(data['CoapplicantIncome'] == 33837)].index)

#data = data.drop(data[(data['CoapplicantIncome'] == 41667)].index)

fig, ax = plt.subplots()
ax.scatter(x = data['CoapplicantIncome'], y = data['LoanAmount'], alpha = 0.5)
plt.ylabel('LoanAmount', fontsize=13)
plt.xlabel('CoapplicantIncome', fontsize=13)
plt.show()


# In[516]:


data.shape


# In[517]:


data.isnull().sum()


# In[518]:


from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

sns.distplot(data['LoanAmount'], fit = norm);
fig = plt.figure()
res = stats.probplot(data['LoanAmount'], plot=plt)
plt.show()


# In[519]:


sns.distplot(data['ApplicantIncome'], fit = norm);
fig = plt.figure()
res = stats.probplot(data['ApplicantIncome'], plot=plt)
plt.show()


# In[520]:


sns.distplot(data['CoapplicantIncome'], fit = norm);
fig = plt.figure()
res = stats.probplot(data['CoapplicantIncome'], plot=plt)
plt.show()


# In[521]:


#correlation matrix
import matplotlib.pyplot as plt
import seaborn as sns
corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# In[522]:


target = data['Loan_Status']


# In[523]:


data['Loan_Status'].value_counts().plot(kind='bar',color=["black","gold"])
print(data['Loan_Status'].value_counts())
plt.xticks(rotation='horizontal')
plt.title("Loan Status")
plt.ylabel("Count for each Loan Status")
plt.xlabel("Loan Status");


# In[524]:


data = data.drop(['Loan_ID','Loan_Status'], axis = 1)


# In[525]:


data.columns


# In[526]:


#minmax_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount']
#minmax_data = data[minmax_features]

#category_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History',
#       'Property_Area', 'Loan_Amount_Term_Cat']
#category_data = data[category_features]


# In[527]:


data.isnull().sum()


# In[528]:


#df = pd.DataFrame(np.random.randn(100, 3))

# from SO answer by tanemaki
#from scipy import stats
#min_data = min_data[(np.abs(stats.zscore(min_data)) < 3).all(axis=1)]


# In[529]:


#min_data.shape


# In[530]:


#data['ApplicantIncome'] = min_data['ApplicantIncome']
#data['CoapplicantIncome'] = min_data['CoapplicantIncome']
#data['LoanAmount'] = min_data['LoanAmount']


# In[531]:


#data = data.drop(data[(data['ApplicantIncome']== 'NaN') & (data['CoapplicantIncome']== 'NaN') & (data['LoanAmount']== 'NaN')].index)
#data = data.dropna()


# In[532]:


#Getting dummy categorical features
data = pd.get_dummies(data)
print(data.shape)


# In[533]:


#data = data.drop(['Loan_Amount_Term'], axis = 1)


# In[534]:


data.describe()


# In[535]:


null_data = data[data.isnull().any(axis=1)]
print(null_data)


# In[536]:


target = np.where(target == 'Y', 1, 0)


# In[537]:


#features =['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Gender_Female',
#       'Gender_Male', 'Married_No','Married_Yes', 'Education_Graduate',
#       'Education_Not Graduate', 'Self_Employed_Yes', 'Credit_History_0.0',
#       'Credit_History_1.0', 'Property_Area_Rural', 'Property_Area_Semiurban', 
#           'Loan_Amount_Term_Cat_12months','Loan_Amount_Term_Cat_6months']

features =['ApplicantIncome','Credit_History_1.0','CoapplicantIncome', 'LoanAmount','Credit_History_0.0','Married_No',
'Property_Area_Semiurban','Property_Area_Rural','Married_Yes','Property_Area_Urban','Self_Employed_Yes',
'Gender_Female','Self_Employed_No','Education_Not Graduate','Education_Graduate','Gender_Other','Dependents_>3']



# In[538]:


data = data[features]


# In[539]:


#Dividing into train and test
import numpy as np
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data,
                                    target,test_size = 0.2, random_state = 42)


# In[352]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# Create the RFE object and rank each pixel
clf_rf_3 = RandomForestRegressor()      
rfe = RFE(estimator=clf_rf_3, n_features_to_select=12, step=1)
rfe = rfe.fit(x_train, y_train)


# In[295]:


print('Chosen best 10 feature by rfe:',x_train.columns[rfe.support_])
new_features = x_train.columns[rfe.support_]


# In[296]:


#Using Random Forest Algorithm to get the predictor variables
from sklearn.ensemble import RandomForestRegressor

clf_rf_5 = RandomForestRegressor()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf_5.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest

plt.figure(1, figsize=(14, 13))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.show()


# In[540]:


#Oversampling the data
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
import os

sm = SMOTE(random_state=42, ratio = 1.0)
x_train_bal, y_train_bal = sm.fit_sample(x_train, y_train)


# In[541]:


#Now we can see the instances are balanced we can proceed to modelling
print(x_train_bal.shape)
print(x_train_bal)
count_target_bal = pd.value_counts(y_train_bal)
print(count_target_bal)


# <h2>Logistic Regression</h2>

# In[542]:


# Grid Search for Logistic Regression Tuning
import numpy as np
from sklearn import datasets
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
# prepare a range of alpha values to test
alphas = np.array(np.power(10.0, np.arange(-10, 10)))
model = LogisticRegression()
grid = GridSearchCV(estimator=model, param_grid=dict(C=alphas))
grid.fit(x_train_bal, y_train_bal)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_.C)


# In[415]:


#Logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 0.01)
log_reg.fit(x_train_bal, y_train_bal)


# In[416]:


from sklearn.model_selection import cross_val_predict
y_pred_test = log_reg.predict(x_test)
y_pred_train = log_reg.predict(x_train_bal)


# In[417]:


print((accuracy_score(y_train_bal, y_pred_train))*100)
print((accuracy_score(y_test, y_pred_test))*100)


# <h2>SVM</h2>

# In[448]:


#Fine tuning the parameters for SVM
from sklearn import svm
from sklearn import svm, grid_search
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_


# In[449]:


svc_param_selection(x_train_bal,y_train_bal, 5)


# In[450]:


from sklearn import svm

svm_clf = svm.SVC(C = 10, gamma = 1)
svm_clf.fit(x_train_bal, y_train_bal)  


# In[451]:


#from sklearn.model_selection import cross_val_predict
y_train_pred = svm_clf.predict(x_train_bal)
y_test_pred = svm_clf.predict(x_test)


# In[452]:


print((accuracy_score(y_train_bal, y_train_pred))*100)
print((accuracy_score(y_test, y_test_pred))*100)


# <h2>Random Forest</h2>

# In[453]:


from sklearn.ensemble import RandomForestClassifier
#fine tuning random forest classifier using Grid search
clf = RandomForestClassifier(random_state =42, max_features = 'auto')
param_grid = {'max_depth' : [10,20,30],
              'n_estimators' :[100,200,300]}
## This line is throwing the error shown below
validator = GridSearchCV(clf, param_grid= param_grid) 
validator.fit(x_train_bal,y_train_bal)
print(validator.best_score_)
print(validator.best_estimator_.n_estimators)
print(validator.best_estimator_.max_depth)


# In[454]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42, max_depth = 20, n_estimators = 200)
forest_clf.fit(x_train_bal, y_train_bal)
y_train_pred = cross_val_predict(forest_clf, x_train_bal, y_train_bal, cv = 10)


# In[455]:


y_test_pred = forest_clf.predict(x_test)


# In[456]:


print((accuracy_score(y_train_bal, y_train_pred))*100)
print((accuracy_score(y_test, y_test_pred))*100)


# <h2>Decision Tree</h2>

# In[457]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators = 500, max_samples = 100,
                           bootstrap= True, n_jobs =-1)

bag_clf.fit(x_train_bal, y_train_bal)
y_pred = bag_clf.predict(x_test)
print(bag_clf.__class__.__name__,"\n Accuracy is :", (accuracy_score(y_test, y_pred))*100 , " \n Precision is:" ,(precision_score(y_test, y_pred))*100, "\n Recall is :", (recall_score(y_test,y_pred))*100)


# <h2>Neural Networks</h2>

# In[458]:


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(8, 2), random_state=1)

clf.fit(x_train_bal, y_train_bal)                         
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)


# In[459]:


y_train_pred = clf.predict(x_train_bal)
y_test_pred = clf.predict(x_test)


# In[460]:


print((accuracy_score(y_train_bal, y_train_pred))*100)
print((accuracy_score(y_test, y_test_pred))*100)


# <h2>Voting Classifier</h2>

# In[461]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


voting_clf = VotingClassifier(
             estimators = [('lr', log_reg),('rf',forest_clf),('svc', svm_clf)], voting = 'hard')
voting_clf.fit(x_train_bal,y_train_bal)


# In[462]:



for clf in (log_reg, forest_clf, svm_clf,bag_clf,clf, voting_clf):
    clf.fit(x_train_bal, y_train_bal)
    y_pred= clf.predict(x_test)
    print(clf.__class__.__name__,"\n Accuracy is :", (accuracy_score(y_test, y_pred))*100 , " \n Precision is:" ,(precision_score(y_test, y_pred))*100, "\n Recall is :", (recall_score(y_test,y_pred))*100)
    


# <h2>Test Data</h2>

# In[463]:


test_data = pd.read_csv("/Users/rahmi/Desktop/AnalyticsVidya/test.csv")
print(test_data.head(5))


# In[464]:


test_data.isnull().sum()


# In[465]:


test_data["Gender"] = test_data["Gender"].fillna("Other")


# In[466]:


#test_data['Dependents'] = test_data['Dependents'].fillna(test_data['Dependents'].mode()[0])


# In[467]:


test_data.loc[test_data['Dependents'] == '0', 'Dependents'] = '<=3'
test_data.loc[test_data['Dependents'] == '1', 'Dependents'] = '<=3'
test_data.loc[test_data['Dependents'] == '2', 'Dependents'] = '<=3'
test_data.loc[test_data['Dependents'] == '3+', 'Dependents'] = '>3'

test_data['Dependents'] = test_data['Dependents'].fillna(test_data['Dependents'].mode()[0])


# In[468]:


test_data['Self_Employed'] = test_data['Self_Employed'].fillna(test_data['Self_Employed'].mode()[0])


# In[469]:


median_value= test_data['LoanAmount'].median()
test_data['LoanAmount']= test_data['LoanAmount'].fillna(median_value)


# In[470]:


test_data.loc[test_data['Loan_Amount_Term'] == 180, 'Loan_Amount_Term_Cat'] = '6months'
test_data.loc[test_data['Loan_Amount_Term'] == 240, 'Loan_Amount_Term_Cat'] = '8months'
test_data.loc[test_data['Loan_Amount_Term'] == 360, 'Loan_Amount_Term_Cat'] = '12months'

test_data['Loan_Amount_Term_Cat'] = test_data['Loan_Amount_Term_Cat'].fillna(test_data['Loan_Amount_Term_Cat'].mode()[0])


# In[471]:


test_data['Credit_History'] = test_data['Credit_History'].astype('category')


# In[472]:


test_data['Credit_History'] = test_data['Credit_History'].fillna(test_data['Credit_History'].mode()[0])


# In[473]:


test_data.info()


# In[474]:


test_data.isnull().sum()


# In[475]:


test_data = test_data.drop(['Loan_Amount_Term'], axis = 1)


# In[476]:


test_data.skew()


# In[477]:


skew_col = test_data.loc[:,['ApplicantIncome','CoapplicantIncome']]
skew_col = pd.DataFrame(skew_col)
skewness = skew_col.skew()


# In[478]:


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    test_data[feat] = boxcox1p(test_data[feat], lam)


# In[479]:


test_data.skew()


# In[480]:


#Normalizing the data using MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

test_data.iloc[:,6:9] = min_max_scaler.transform(test_data.iloc[:,6:9])
#min_max_scaler = preprocessing.MinMaxScaler()
#np_scaled = min_max_scaler.fit_transform(data.iloc[:,5:8])
#test_data.iloc[:,6:9] = pd.DataFrame(scaler, columns = test_data.iloc[:,6:9].columns)
test_data.describe()


# In[481]:


loan_id = test_data['Loan_ID']


# In[482]:


test_data = test_data.drop(['Loan_ID'], axis = 1)


# In[483]:


#Getting dummy categorical features
test_data = pd.get_dummies(test_data)
print(test_data.shape)


# In[484]:


test_data.describe()


# In[485]:


test_features = ['ApplicantIncome','Credit_History_1.0','CoapplicantIncome', 'LoanAmount','Credit_History_0.0','Married_No',
'Property_Area_Semiurban','Property_Area_Rural','Married_Yes','Property_Area_Urban','Self_Employed_Yes',
'Gender_Female','Self_Employed_No','Education_Not Graduate','Education_Graduate','Loan_Amount_Term_Cat_6months']


# In[486]:


test_data = test_data[features]


# <h2>Logistic Regression Test</h2>

# In[442]:


#Logistic regression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(C = 10)
log_reg.fit(data, target)


# In[443]:


from sklearn.model_selection import cross_val_predict
y_pred_test = log_reg.predict(test_data)
y_pred_train = log_reg.predict(data)


# In[444]:


print((accuracy_score(target, y_pred_train))*100)
#print((accuracy_score(y_test, y_test_pred))*100)


# In[445]:


y_pred_test


# In[446]:


y_pred_test = np.where(y_pred_test == 1,'Y','N')


# In[447]:


sub = pd.DataFrame()
sub['Loan_ID'] = loan_id
sub['Loan_Status'] = y_pred_test
sub.to_csv('submission_stacked_log.csv',index=False)


# <h2>Random Forest</h2>

# In[175]:


from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42, max_depth = 30, n_estimators = 100)
forest_clf.fit(data, target)
y_train_pred = cross_val_predict(forest_clf, data, target, cv = 10)


# In[176]:


y_test_pred = forest_clf.predict(test_data)


# In[177]:


print((accuracy_score(target, y_train_pred))*100)


# <h2>Voting Classifier</h2>

# In[178]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression


voting_clf = VotingClassifier(
             estimators = [('lr', log_reg),('rf',forest_clf),('svc', svm_clf)], voting = 'hard')
voting_clf.fit(data,target)


# In[179]:


y_pred_test = voting_clf.predict(test_data)
y_pred_train = voting_clf.predict(data)


# In[180]:


print((accuracy_score(target, y_pred_train))*100)


# In[181]:


print(y_pred_test)


# In[182]:


y_pred_test = np.where(y_pred_test == 1,'Y','N')


# In[183]:


sub = pd.DataFrame()
sub['Loan_ID'] = loan_id
sub['Loan_Status'] = y_pred_test
sub.to_csv('submission_stacked.csv',index=False)

