#!/usr/bin/env python
# coding: utf-8

# In[35]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Importing Libraries

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import itertools
import plotly.graph_objects as go

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.simplefilter(action="ignore")


# In[37]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[137]:


df = pd.read_csv("/kaggle/input/credit-card-data-set/credit_card_approval.csv")


# In[39]:


df.head(10)


# In[40]:


df.tail(10)


# In[ ]:


# def check_df(dataframe, head=5):    
    # print("##################### Quantiles #####################")
    # print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)  
# check_df(df)
#sayısala dönüştürme sonrası bu işlem yapılacak


# In[41]:


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df)


# In[138]:


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Returns the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note Categorical variables include categorical variables with numeric appearance.

    Parameters
    ------
        dataframe: dataframe
                Variable names of the dataframe to be taken
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                List of cardinal variables with categorical appearance

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 return lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"] 

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car] 

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] 

    num_cols = [col for col in num_cols if col not in num_but_cat] 

    print(f"Observations: {dataframe.shape[0]}") 
    print(f"Variables: {dataframe.shape[1]}") 
    print(f'cat_cols: {len(cat_cols)}') 
    print(f'num_cols: {len(num_cols)}') 
    print(f'cat_but_car: {len(cat_but_car)}') 
    print(f'num_but_cat: {len(num_but_cat)}') 


    return cat_cols, num_cols, cat_but_car, num_but_cat


# In[139]:


cat_cols, num_cols, cat_but_car,  num_but_cat = grab_col_names(df)


# In[140]:


cat_cols


# In[141]:


num_cols


# In[142]:


cat_but_car


# In[143]:


num_but_cat


# # Analysis of Categorical Variables

# In[48]:


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


# In[49]:


# If there were more than one categorical variable, we would loop through all categorical variables one by one as follows to run the function.

for col in cat_cols:
    cat_summary(df, col, plot=True)


# * **FLAG_MOBIL** has a value and this is not logical for data analysis;therefore, this variable should delete from data.

# # Analysis of Numerical Variables

# In[50]:


def num_summary(dataframe, numerical_col, plot=False, color='blue'):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20,color='blue')
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# In[51]:


for col in num_cols:
    num_summary(df, col, plot=True, color='blue')


# * Analysis of Numerical Variables should be **change into** **true value** and ought to analysis.

# # Analysis of Categorical Variables by Target Variable

# In[52]:


def target_summary_with_cat(dataframe, target, categorical_col, plot=False):
    print(pd.DataFrame({'TARGET_MEAN': dataframe.groupby(categorical_col)[target].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=categorical_col, y=target, data=dataframe)
        plt.show(block=True)


# In[53]:


for col in cat_cols:
    target_summary_with_cat(df, "TARGET", col, plot=True)


# # Analysis of Numeric Variables by Target Variable

# In[54]:


def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    print(pd.DataFrame({numerical_col+'_mean': dataframe.groupby(target)[numerical_col].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=target, y=numerical_col, data=dataframe)
        plt.show(block=True)


# In[55]:


for col in num_cols:
    target_summary_with_num(df, "TARGET", col, plot=True)


# # Target Variable Log Process

# In[56]:


np.log1p(df["TARGET"]).hist(bins=50, color='red')
plt.show(block=True)


# # Correlation Analysis With Numarical Variables

# In[57]:


corr = df[num_cols].corr()


# In[58]:


corr


# In[59]:


# Correlation heatmap without using functions

sns.set(rc={"figure.figsize": (12, 12)})
corr_values = corr.round(2)
sns.heatmap(corr, cmap="RdBu", annot=corr_values)
plt.show(block=True)


# * **DAYS_EMPLOYED** and **DAYS_BIRTH** are positive and low correlation.(0.32)

# # Target Variable Analysis

# In[60]:


# Calculate the counts of each outcome
outcome_counts = df['TARGET'].value_counts()

# Calculate the total number of patients
total_patients = outcome_counts.sum()

# Calculate the percentages
percentages = outcome_counts / total_patients * 100

# Create labels with both quantity and percentage
labels = [f'0 - Non-Risk\n({outcome_counts[0]} / {percentages[0]:.1f}%)',
          f'1 - Risk\n({outcome_counts[1]} / {percentages[1]:.1f}%)']

# Plot the pie chart with labels and percentages
plt.figure(figsize=(8, 6))
plt.pie(outcome_counts, labels=labels, autopct='%1.1f%%', colors=['green', 'red'])
plt.title('Distribution of the Outcome Variable')
plt.show()


# * **%0.4** has a risk for credit card applicants.(1962 applicants or people/women/men)

# # FLAG_MOBIL Variable Process And Control

# In[144]:


col = 'FLAG_MOBIL'

print(f"'{col}' değişkeni için özet:")
print(f"- Boş değer sayısı       : {df[col].isnull().sum()}")
print(f"- Boş değer oranı (%)     : {100 * df[col].isnull().mean():.2f}")
print(f"- Eşsiz değer sayısı      : {df[col].nunique()}")
print(f"- Eşsiz değerler          : {df[col].unique()}")


# * FLAG_MOBILE has only a value and that does not mean to data and project; thus, this should delete from data and we should continue clear data.

# **Deleting Process**

# In[145]:


df.drop(columns=['FLAG_MOBIL'], inplace=True)


# **Control Process**

# In[146]:


'FLAG_MOBIL' in df.columns


# # Numerical Variables Visualisation

# In[64]:


sns.pairplot(data=df, vars=['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'BEGIN_MONTHS'], hue='TARGET', height=5)
plt.show(block=True)


# # Pairwise Cramér's V Heatmap (Kategorik Korelasyon Haritası)

# In[68]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

categorical_cols = df.select_dtypes(include='object').columns.tolist()
cramers_results = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

for col1 in categorical_cols:
    for col2 in categorical_cols:
        if col1 == col2:
            cramers_results.loc[col1, col2] = 1.0
        else:
            cramers_results.loc[col1, col2] = cramers_v(df[col1], df[col2])

plt.figure(figsize=(10, 8))
sns.heatmap(cramers_results.astype(float), annot=True, cmap='Reds')
plt.title("Cramér's V Heatmap (Categorical Correlation)")
plt.show()


# In[72]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

cross_tab = pd.crosstab(df['NAME_EDUCATION_TYPE'], df['NAME_FAMILY_STATUS'])
plt.figure(figsize=(10,6))
sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
plt.title('Education Type vs Family Status')
plt.show()


# # Kategorik Değişkenlerin Heatmap İle (Çapraz Frekans Tablosu ile) Görselleştirilmesi

# In[73]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# Kullanılacak kategorik sütunlar
categorical_columns = [
    'CODE_GENDER',
    'FLAG_OWN_CAR',
    'FLAG_OWN_REALTY',
    'CNT_CHILDREN',
    'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',
    'JOB',
    'STATUS',
    'FLAG_WORK_PHONE',
    'FLAG_PHONE',
    'FLAG_EMAIL',
    'TARGET'
]

# İkili kombinasyonlarını oluştur
combinations_list = list(combinations(categorical_columns, 2))

# Her kombinasyon için heatmap çiz
for col1, col2 in combinations_list:
    try:
        cross_tab = pd.crosstab(df[col1], df[col2])
        plt.figure(figsize=(10, 6))
        sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{col1} vs {col2}')
        plt.xlabel(col2)
        plt.ylabel(col1)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error while plotting {col1} vs {col2}: {e}")


# # Binary Categorical Variables Visualisation

# In[ ]:


# Create combinations of binary categorical variables
#feature_combinations = list(itertools.combinations(['CODE_GENDER',
 #'FLAG_OWN_CAR',
 #'FLAG_OWN_REALTY',
 #'CNT_CHILDREN',
 #'NAME_EDUCATION_TYPE',
# 'NAME_FAMILY_STATUS','TARGET'], 2))

# Create a separate Bubble Chart for each binary categorical variable
#for i, (feature1, feature2) in enumerate(feature_combinations):
 #   fig = px.scatter(df, x=feature1, y=feature2, color='TARGET', size='TARGET',
  #                   title=f'{feature1} vs {feature2} Bubble Chart')

   # fig.show(block=True)


# In[ ]:


# Create combinations of binary categorical variables
 #feature_combinations = list(itertools.combinations(['NAME_HOUSING_TYPE',
 # 'JOB',
  #'STATUS',
  #'FLAG_WORK_PHONE',
  #'FLAG_PHONE',
  #'FLAG_EMAIL',
  #'TARGET'], 2))

# Create a separate Bubble Chart for each binary categorical variable
 #for i, (feature1, feature2) in enumerate(feature_combinations):
    # fig = px.scatter(df, x=feature1, y=feature2, color='TARGET', size='TARGET',
        #              title=f'{feature1} vs {feature2} Bubble Chart')

    # fig.show(block=True)


# * **Because of CPU.**

# In[147]:


# Hem eşsiz değer sayısı hem de değerleri göster
for col in cat_cols:
    if col != 'FLAG_MOBIL':
        print(f"\n{col} ({df[col].nunique()} eşsiz değer):")
        print(df[col].unique())


# In[148]:


for col in cat_cols:
    if col != 'FLAG_MOBIL' and df[col].nunique() == 2:
        print(f"\n{col} ({df[col].nunique()} eşsiz değer):")
        print(df[col].unique())


# In[149]:


binary_cols = [col for col in cat_cols if col in df.columns and df[col].nunique() == 2 and col != 'FLAG_MOBIL']
binary_cols


# In[150]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in binary_cols:
    df[col] = le.fit_transform(df[col])


# In[151]:


df.head()


# In[87]:


job_risk = df.groupby("JOB")["TARGET"].mean().sort_values()
job_risk


# In[88]:


print(f"JOB ({df['JOB'].nunique()} eşsiz değer):")
print(df['JOB'].unique())


# # 🔢 CNT_CHILDREN with Ordinal Encoding

# * Ekonomik yük açısından sıralanmalı (çocuk sayısı arttıkça yük artar)

# In[152]:


print(df['CNT_CHILDREN'].unique())


# In[153]:


print(df['CNT_CHILDREN'].nunique())


# In[154]:


df['CNT_CHILDREN'].head()


# In[155]:


children_map = {
    'No children': 0,
    '1 children': 1,
    '2+ children': 2
}
df['CNT_CHILDREN'] = df['CNT_CHILDREN'].map(children_map)


# In[156]:


df['CNT_CHILDREN'].head()


# # 🎓 NAME_EDUCATION_TYPE With Ordinal Encoding

# In[157]:


education_map = {
    'Lower secondary': 1,
    'Secondary / secondary special': 2,
    'Incomplete higher': 3,
    'Higher education': 4,
    'Academic degree': 5
}
df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(education_map)


# In[158]:


df['NAME_EDUCATION_TYPE'].head()


# # 👪 NAME_FAMILY_STATUS With Ordinal Encoding

# In[159]:


family_map = {
    'Married': 0,
    'Civil marriage': 1,
    'Single / not married': 2,
    'Widow': 3,
    'Separated': 4
}
df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].map(family_map)


# In[160]:


df['NAME_FAMILY_STATUS'].head()


# Aile durumu risk sıralamasına göre değerlendirilebilir. Genelde:
# 
# * Evli kişiler daha stabil olabilir.
# 
# * Bekâr, boşanmış ve dul kişiler daha yüksek riskli olabilir.

# # 🏠 NAME_HOUSING_TYPE With Ordinal Encoding

# In[161]:


housing_map = {
    'House / apartment': 0,           # Sahip olunan
    'Co-op apartment': 1,             # Ortak sahiplik
    'With parents': 2,                # Aile yanında
    'Municipal apartment': 3,         # Kamu konutu
    'Rented apartment': 4,            # Kira
    'Office apartment': 5             # İş yeri adresi, riskli
}
df['NAME_HOUSING_TYPE'] = df['NAME_HOUSING_TYPE'].map(housing_map)


# In[162]:


df['NAME_HOUSING_TYPE'].head()


# # 👷‍♂️ JOB With Ordinal Encoding

# Meslek grupları maaş, iş güvencesi ve toplumsal prestij açısından sıralanabilir. 
# 
# Aşağıdaki sıralama ekonomik güç ve eğitim seviyesi gözetilerek yapılmıştır.
# 
# (0 en yüksek, 17 en düşük):

# In[163]:


job_order = [
    'Managers', 'High skill tech staff', 'IT staff', 'Accountants',
    'Core staff', 'HR staff', 'Medicine staff', 'Sales staff',
    'Realty agents', 'Drivers', 'Security staff', 'Private service staff',
    'Cooking staff', 'Cleaning staff', 'Secretaries',
    'Waiters/barmen staff', 'Laborers', 'Low-skill Laborers'
]

job_map = {job: i for i, job in enumerate(job_order)}
df['JOB'] = df['JOB'].map(job_map)


# In[164]:


df['JOB'].head()


# # 📊 STATUS With Ordinal Encoding

# | Değer | Anlamı (muhtemel)                                                  |
# | ----- | ------------------------------------------------------------------ |
# | `'0'` | Ödeme zamanında                                                    |
# | `'1'` | 1 ay gecikme                                                       |
# | `'2'` | 2 ay gecikme                                                       |
# | `'3'` | 3 ay gecikme                                                       |
# | `'4'` | 4 ay gecikme                                                       |
# | `'5'` | 5+ ay gecikme                                                      |
# | `'C'` | **Closed** (hesap kapatıldı veya kredi ödendi) ✅                   |
# | `'X'` | **No loan / No history** (kredi yok, veri eksik veya bilinmeyen) ❓ |
# 

# 🧠 Yorum:
# C: Kredi tamamen ödenmiş olabilir (risksiz veya düşük riskli kabul edilir).
# 
# X: Hiç kredi almamış ya da verisi olmayan müşteri olabilir (riskli de olabilir, model belirsizliğe karşı dikkatli olmalı).

# In[165]:


status_order = ['0', '1', '2', '3', '4', '5', 'C', 'X']
status_map = {k: i for i, k in enumerate(status_order)}
df['STATUS'] = df['STATUS'].map(status_map)


# In[166]:


df['STATUS'].head()


# In[167]:


df.head()


# In[168]:


df.tail()


# # 📌 BEGIN_MONTHS With Positive Change For Model Evaluation

# In[169]:


df["BEGIN_MONTHS"] = -df["BEGIN_MONTHS"]


# In[170]:


df["BEGIN_MONTHS"].head()


# # 📌 DAYS_BIRTH With Positive And Real Age Change For Model Evaluation 

# In[171]:


df["DAYS_BIRTH"] = (-df["DAYS_BIRTH"]) // 365


# In[172]:


df["DAYS_BIRTH"].head()


# # 📌 DAYS_EMPLOYED With Positive And Real Day-Year Change For Model Evaluation

# In[173]:


df["DAYS_EMPLOYED"].value_counts().head()


# In[174]:


(365243 in df["DAYS_EMPLOYED"].values)


# In[175]:


df[df["DAYS_EMPLOYED"] == 365243].shape[0]


# In[176]:


df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
df["DAYS_EMPLOYED"] = (-df["DAYS_EMPLOYED"]) / 365


# In[177]:


df["DAYS_EMPLOYED"].head()


# In[178]:


df.head()


# In[179]:


df.tail()


# # Quantiles With All Numerrical Variables

# In[180]:


def check_df(dataframe, head=5):    
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)  

check_df(df)


# # Correlation Analysis With Trashold

# In[181]:


# Creation of correlation heat map using the function

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (12, 12)})
        corr_values = corr.round(2)
        sns.heatmap(corr, cmap="RdBu", annot=corr_values)
        plt.show(block=True)
    return drop_list


# In[182]:


high_correlated_cols(df, plot=True)


# 0.37 'FLAG_WORK_PHONE' - 'FLAG_PHONE'
# 
# 0.35 'CODE_GENDER' - 'FLAG_OWN_CAR'
# 
# 0.32 'DAYS_BIRTH' - 'DAYS_EMPLOYED' *
# 
# 0.22 'NAME_EDUCATION_TYPE' - 'AMT_INCOME_TOTAL'
# 
# 0.20 'CODE_GENDER' - 'JOB'
# 
# 0.19 'AMT_INCOME_TOTAL' - 'CODE_GENDER'
# 
# 0.10 'FLAG_WORK_PHONE' - 'ID' is not logical
# 
# 0.09 'FLAG_EMAIL' - 'NAME_EDUCATION_TYPE' 
# 
# 0.07 'FLAG_EMAIL' - 'AMT_INCOME_TOTAL'
# 
# 0.06 'FLAG_EMAIL' - 'FLAG_OWN_REALTY'
# 
# 0.05 'BEGIN_MONTHS' - 'DAYS_BIRTH' * 
# 
# 0.04 'BEGIN_MONTHS' - 'DAYS_EMPLOYED'
# 
# 0.03 'FLAG_PHONE' - 'DAYS_BIRTH', 'DAYS_EMPLOYED' *
# 
# 0.02 'FLAG_WORK_PHONE' - 'CODE_GENDER'
# 
# 0.01 'TARGET' - 'FLAG_OWN_REALTY'

# # After Changing Into Real Values, Analysis of Numerical Variables

# In[183]:


def num_summary(dataframe, numerical_col, plot=False, color='blue'):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20,color='blue')
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True, color='blue')


# # After Changing Into Real Values, Analysis of Numeric Variables by Target Variable

# In[184]:


def target_summary_with_num(dataframe, target, numerical_col, plot=False):
    print(pd.DataFrame({numerical_col+'_mean': dataframe.groupby(target)[numerical_col].mean()}), end='\n\n\n')
    if plot:
        sns.barplot(x=target, y=numerical_col, data=dataframe)
        plt.show(block=True)

for col in num_cols:
    target_summary_with_num(df, "TARGET", col, plot=True)


# * Numerical variables have a balanced at data and compared to target with these, model has a huge power on that selection.

# In[185]:


num_but_cat


# In[186]:


cat_cols


# In[187]:


num_cols


# # Veri odaklı risk bazlı encoding
# risk_order = df.groupby("CNT_CHILDREN")["TARGET"].mean().sort_values().index
# children_map = {k: i for i, k in enumerate(risk_order)}
# df["CNT_CHILDREN_ENCODED"] = df["CNT_CHILDREN"].map(children_map)
# # Veri odaklı risk bazlı encoding
# risk_order = df.groupby("CNT_CHILDREN")["TARGET"].mean().sort_values().index
# children_map = {k: i for i, k in enumerate(risk_order)}
# df["CNT_CHILDREN_ENCODED"] = df["CNT_CHILDREN"].map(children_map)
# 

# job_order = {
#     'Low-skill Laborers': 1,
#     'Cleaning staff': 2,
#     'Cooking staff': 3,
#     'Waiters/barmen staff': 4,
#     'Drivers': 5,
#     'Security staff': 6,
#     'Private service staff': 7,
#     'Core staff': 8,
#     'Sales staff': 8,
#     'Realty agents': 8,
#     'Secretaries': 8,
#     'Accountants': 9,
#     'HR staff': 9,
#     'IT staff': 9,
#     'Medicine staff': 9,
#     'High skill tech staff': 10,
#     'Managers': 10
# }
# 
# df['JOB'] = df['JOB'].map(job_order)

# In[188]:


df.head()


# In[189]:


df.tail()


# # Eksik (NaN) Değerleri Kontrol Et

# In[191]:


# Eksik değerlerin sayısı ve oranını göster
def missing_values(df):
    missing_count = df.isnull().sum()
    missing_ratio = 100 * missing_count / len(df)
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Ratio (%)': missing_ratio
    })
    return missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Ratio (%)', ascending=False)

missing_values(df)


# # Outliers Investigating

# In[192]:


# Aykırı değerleri IQR yöntemine göre bulan fonksiyon
def detect_outliers(df, num_cols):
    outlier_summary = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_summary[col] = {
            'Outlier Count': len(outliers),
            'Outlier Ratio (%)': round(len(outliers) * 100 / len(df), 2)
        }
    return pd.DataFrame(outlier_summary).T.sort_values(by='Outlier Count', ascending=False)

# Sayısal değişkenleri seç
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Aykırı değer kontrolü
detect_outliers(df, numeric_cols)


# # Outliers Visualisation 

# In[193]:


import matplotlib.pyplot as plt
import seaborn as sns

def check_df(dataframe, head=5):
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    num_cols = dataframe.select_dtypes(include=["number"]).columns

    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        sns.histplot(dataframe[col], bins=30, kde=True, ax=axes[0], color="skyblue")
        axes[0].set_title(f'Histogram of {col}')

        sns.boxplot(x=dataframe[col], ax=axes[1], color="lightgreen")
        axes[1].set_title(f'Boxplot of {col}')

        plt.tight_layout()
        plt.show()

# Kullanımı:
check_df(df)


# # 'NAME_HOUSING_TYPE', 'FLAG_EMAIL', 'NAME_FAMILY_STATUS', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL' Analysis For Outliers

# In[194]:


import matplotlib.pyplot as plt
import seaborn as sns

def check_selected_cols(dataframe, cols):
    print("##################### Quantiles #####################")
    print(dataframe[cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    for col in cols:
        if pd.api.types.is_numeric_dtype(dataframe[col]):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            sns.histplot(dataframe[col], bins=30, kde=True, ax=axes[0], color="skyblue")
            axes[0].set_title(f'Histogram of {col}')

            sns.boxplot(x=dataframe[col], ax=axes[1], color="lightgreen")
            axes[1].set_title(f'Boxplot of {col}')

            plt.tight_layout()
            plt.show()
        else:
            print(f"'{col}' sayısal değil, grafik çizilmiyor.")

# Kullanımı:
selected_columns = ['NAME_HOUSING_TYPE', 'FLAG_EMAIL', 'NAME_FAMILY_STATUS', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL']
check_selected_cols(df, selected_columns)


# In[195]:


import pandas as pd

def fill_outliers_and_missing(df):
    # Sayısal değişkenler için IQR yöntemi ile aykırı değerleri ortalama ile doldurma
    numeric_cols = ['DAYS_EMPLOYED', 'AMT_INCOME_TOTAL']
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Ortalama hesapla (aykırı değerler hariç)
        mean_val = df.loc[(df[col] >= lower_bound) & (df[col] <= upper_bound), col].mean()

        # Aykırı değerleri ortalama ile değiştir
        df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = mean_val

    # Kategorik değişkenler için mod ile doldurma
    cat_cols = ['NAME_HOUSING_TYPE', 'FLAG_EMAIL', 'NAME_FAMILY_STATUS']
    for col in cat_cols:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

    return df

# Kullanımı:
df = fill_outliers_and_missing(df)


# 🎯 Özetle:
# 
# * Sayısal	DAYS_EMPLOYED, AMT_INCOME_TOTAL	Aykırı değerleri ortalama ile değiştirir.
# 
# * Kategorik	NAME_HOUSING_TYPE, FLAG_EMAIL, NAME_FAMILY_STATUS	Eksik değerleri mod ile doldurur.

# # Outlier Control And Check

# In[196]:


def check_selected_cols(dataframe, cols):
    print("##################### Quantiles #####################")
    print(dataframe[cols].quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

selected_columns = ['NAME_HOUSING_TYPE', 'FLAG_EMAIL', 'NAME_FAMILY_STATUS', 'DAYS_EMPLOYED', 'AMT_INCOME_TOTAL']
check_selected_cols(df, selected_columns)


# In[ ]:


Sayısal verilerin aykırı değerleri ortalam ile dolduruldu.
Kategorik verilerin eksik değerleri mod ile dolduruldu ve aykırı değer araştırmak anlamsızdır. 


# # Negative Value Control

# In[197]:


def check_negative_values(df):
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        neg_count = (df[col] < 0).sum()
        if neg_count > 0:
            print(f"'{col}' sütununda {neg_count} adet negatif değer var.")
        else:
            print(f"'{col}' sütununda negatif değer yok.")

check_negative_values(df)


# # Feature Engineering With Making New Variables

# In[199]:


# Yaş ve çalışma süresi için pozitif değerler
df['AGE'] = abs(df['DAYS_BIRTH']) / 365
df['EMPLOYED_YEARS'] = abs(df['DAYS_EMPLOYED']) / 365

# Çalışma süresinin yaşa oranı
df['EMPLOYED_RATIO'] = df['DAYS_EMPLOYED'] / df['AGE']

# Telefon kullanım sayısı
df['PHONE_COUNT'] = df['FLAG_PHONE'] + df['FLAG_WORK_PHONE']

# Cinsiyet ve araç sahipliği etkileşimi (string olarak birleştirip kategorik feature)
#df['GENDER_OWN_CAR'] = df['CODE_GENDER'].astype(str) + "_" + df['FLAG_OWN_CAR'].astype(str)

# Eğitim seviyesine göre gelir ortalaması (gruplandırılmış gelir)
df['EDU_INCOME_MEAN'] = df.groupby('NAME_EDUCATION_TYPE')['AMT_INCOME_TOTAL'].transform('mean')

# Cinsiyet ve iş tipi etkileşimi
#df['GENDER_JOB'] = df['CODE_GENDER'].astype(str) + "_" + df['JOB'].astype(str)

# Cinsiyet bazında ortalama gelir
df['GENDER_INCOME_MEAN'] = df.groupby('CODE_GENDER')['AMT_INCOME_TOTAL'].transform('mean')

df.head()


# In[ ]:


#df.drop(columns=['GENDER_OWN_CAR', 'GENDER_JOB'], inplace=True)


# # Standardization Process With Numerical Variables

# In[200]:


num_cols


# "ID" variable is not important for this process because this does not have a meaningful for data and that has only identification for people who is someone. 

# In[201]:


from sklearn.preprocessing import RobustScaler

# ID hariç sayısal sütunlar
a = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'BEGIN_MONTHS']

scaler = RobustScaler()
df[a] = scaler.fit_transform(df[a])

df.head(10)


# In[202]:


df.shape


# # Model Building/Base Model Step

# In[203]:


# Creating the Dependent Variable.

y = df["TARGET"]

# Creating Independent Variables.

X = df.drop("TARGET", axis=1)

# Splitting the Data into Training and Test Sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=17)


# * Train Size = %65
# 
# * Test Size = %35

# In[204]:


X_train


# In[205]:


X_test


# In[206]:


y_train


# In[207]:


y_test


# # RandomForestClassifier

# In[208]:


import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Model Training with timing
start_train = time.time()
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

# Prediction with timing
start_pred = time.time()
y_pred = rf_model.predict(X_test)
end_pred = time.time()
prediction_time = end_pred - start_pred

# Evaluation metrics
print("RandomForestClassifier Results:")
print(f"Training Time: {round(training_time, 4)} seconds")
print(f"Prediction Time: {round(prediction_time, 4)} seconds")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
print(f"Recall: {round(recall_score(y_test, y_pred), 4)}")
print(f"Precision: {round(precision_score(y_test, y_pred), 4)}")
print(f"F1 Score: {round(f1_score(y_test, y_pred), 4)}")
print(f"AUC Score: {round(roc_auc_score(y_test, y_pred), 4)}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")


# In[211]:


from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score

# Tahmin olasılıkları (pozitif sınıf = 1)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# ROC eğrisi için değerler
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

# ROC Curve çizimi
roc_display_rf = RocCurveDisplay(fpr=fpr_rf, tpr=tpr_rf,
                                 roc_auc=roc_auc_score(y_test, rf_probs),
                                 estimator_name='Random Forest')
roc_display_rf.plot(linewidth=2, color='green')


# In[212]:


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title(f'Feature Importance - RandomForestClassifier')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
plot_importance(rf_model, X)


# # Logistic Regression

# In[213]:


import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# Model training + süre ölçümü
start_train = time.time()
lr_model = LogisticRegression(random_state=46, solver='lbfgs', max_iter=1000).fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

# Tahmin + süre ölçümü
start_pred = time.time()
lr_pred = lr_model.predict(X_test)
end_pred = time.time()
prediction_time = end_pred - start_pred

# Değerlendirme metrikleri
print("Logistic Regression Results:")
print(f"Training Time   : {training_time:.4f} seconds")
print(f"Prediction Time : {prediction_time:.4f} seconds")
print(f"Accuracy        : {accuracy_score(y_test, lr_pred):.4f}")
print(f"Recall          : {recall_score(y_test, lr_pred):.4f}")
print(f"Precision       : {precision_score(y_test, lr_pred):.4f}")
print(f"F1 Score        : {f1_score(y_test, lr_pred):.4f}")
print(f"AUC Score       : {roc_auc_score(y_test, lr_pred):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, lr_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")


# In[214]:


from sklearn.metrics import roc_curve, RocCurveDisplay

# Tahmin olasılıkları (pozitif sınıf = 1)
lr_probs = lr_model.predict_proba(X_test)[:, 1]

# ROC eğrisi için değerler
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)

# ROC Curve çizimi
roc_display_lr = RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr,
                                 roc_auc=roc_auc_score(y_test, lr_probs),
                                 estimator_name='Logistic Regression')
roc_display_lr.plot(linewidth=2, color='navy')


# # K-Nearest Neighbors (KNN)

# In[215]:


import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# Modeli eğit ve süreyi ölç
start_train = time.time()
knn_model = KNeighborsClassifier().fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

# Tahmin yap ve süreyi ölç
start_pred = time.time()
knn_pred = knn_model.predict(X_test)
end_pred = time.time()
prediction_time = end_pred - start_pred

# Performans metrikleri
print("K-Nearest Neighbors (KNN) Results:")
print(f"Training Time   : {training_time:.4f} seconds")
print(f"Prediction Time : {prediction_time:.4f} seconds")
print(f"Accuracy        : {accuracy_score(y_test, knn_pred):.4f}")
print(f"Recall          : {recall_score(y_test, knn_pred):.4f}")
print(f"Precision       : {precision_score(y_test, knn_pred):.4f}")
print(f"F1 Score        : {f1_score(y_test, knn_pred):.4f}")
print(f"AUC Score       : {roc_auc_score(y_test, knn_pred):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, knn_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")


# In[216]:


from sklearn.metrics import roc_curve, RocCurveDisplay

# Eğer predict_proba destekleniyorsa kullan (KNN destekliyor)
y_prob = knn_model.predict_proba(X_test)[:, 1]  # Pozitif sınıfın olasılığı

# ROC eğrisi hesapla
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# ROC eğrisini çiz
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_score(y_test, y_prob), estimator_name='KNN')
roc_display.plot(linewidth=2, color='darkorange')


# # Support Vector Classifier (SVC)

# In[217]:


import time
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# Modeli eğit ve süreyi ölç
start_train = time.time()
svc_model = SVC(random_state=46, probability=True).fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

# Tahmin yap ve süreyi ölç
start_pred = time.time()
svc_pred = svc_model.predict(X_test)
end_pred = time.time()
prediction_time = end_pred - start_pred

# Performans metrikleri
print("Support Vector Classifier (SVC) Results:")
print(f"Training Time   : {training_time:.4f} seconds")
print(f"Prediction Time : {prediction_time:.4f} seconds")
print(f"Accuracy        : {accuracy_score(y_test, svc_pred):.4f}")
print(f"Recall          : {recall_score(y_test, svc_pred):.4f}")
print(f"Precision       : {precision_score(y_test, svc_pred):.4f}")
print(f"F1 Score        : {f1_score(y_test, svc_pred):.4f}")
print(f"AUC Score       : {roc_auc_score(y_test, svc_pred):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, svc_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")


# In[218]:


from sklearn.metrics import roc_curve, RocCurveDisplay

# Pozitif sınıf için tahmin olasılıkları
svc_probs = svc_model.predict_proba(X_test)[:, 1]

# FPR, TPR değerlerini hesapla
fpr_svc, tpr_svc, _ = roc_curve(y_test, svc_probs)

# ROC eğrisi çizimi
roc_display_svc = RocCurveDisplay(fpr=fpr_svc, tpr=tpr_svc,
                                  roc_auc=roc_auc_score(y_test, svc_probs),
                                  estimator_name='SVC')
roc_display_svc.plot(linewidth=2, color='darkorange')


# # Decision Tree Classifier

# In[219]:


import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# Model eğitimi
start_train = time.time()
dt_model = DecisionTreeClassifier(random_state=46).fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

# Tahmin
start_pred = time.time()
dt_pred = dt_model.predict(X_test)
end_pred = time.time()
prediction_time = end_pred - start_pred

# Performans çıktıları
print("Decision Tree Classifier Results:")
print(f"Training Time   : {training_time:.4f} seconds")
print(f"Prediction Time : {prediction_time:.4f} seconds")
print(f"Accuracy        : {accuracy_score(y_test, dt_pred):.4f}")
print(f"Recall          : {recall_score(y_test, dt_pred):.4f}")
print(f"Precision       : {precision_score(y_test, dt_pred):.4f}")
print(f"F1 Score        : {f1_score(y_test, dt_pred):.4f}")
print(f"AUC Score       : {roc_auc_score(y_test, dt_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, dt_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")


# In[220]:


from sklearn.metrics import roc_curve, RocCurveDisplay

# Decision Tree için olasılık tahminleri
dt_probs = dt_model.predict_proba(X_test)[:, 1]

# ROC eğrisi için FPR, TPR hesapla
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)

# ROC görselleştirme
roc_display_dt = RocCurveDisplay(fpr=fpr_dt, tpr=tpr_dt,
                                 roc_auc=roc_auc_score(y_test, dt_probs),
                                 estimator_name='Decision Tree')
roc_display_dt.plot(linewidth=2, color='teal')


# In[221]:


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title(f'Feature Importance - {model.__class__.__name__}')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
plot_importance(dt_model, X)


# # AdaBoost Classifier

# In[222]:


import time
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# Model eğitimi
start_train = time.time()
ada_model = AdaBoostClassifier(random_state=46).fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

# Tahmin
start_pred = time.time()
ada_pred = ada_model.predict(X_test)
end_pred = time.time()
prediction_time = end_pred - start_pred

# Performans çıktıları
print("AdaBoost Classifier Results:")
print(f"Training Time   : {training_time:.4f} seconds")
print(f"Prediction Time : {prediction_time:.4f} seconds")
print(f"Accuracy        : {accuracy_score(y_test, ada_pred):.4f}")
print(f"Recall          : {recall_score(y_test, ada_pred):.4f}")
print(f"Precision       : {precision_score(y_test, ada_pred):.4f}")
print(f"F1 Score        : {f1_score(y_test, ada_pred):.4f}")
print(f"AUC Score       : {roc_auc_score(y_test, ada_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, ada_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")


# In[223]:


from sklearn.metrics import roc_curve, RocCurveDisplay

# Olasılık tahmini (pozitif sınıf için)
ada_probs = ada_model.predict_proba(X_test)[:, 1]

# FPR, TPR değerlerini hesapla
fpr_ada, tpr_ada, _ = roc_curve(y_test, ada_probs)

# ROC eğrisi çizimi
roc_display_ada = RocCurveDisplay(fpr=fpr_ada, tpr=tpr_ada,
                                  roc_auc=roc_auc_score(y_test, ada_probs),
                                  estimator_name='AdaBoost')
roc_display_ada.plot(linewidth=2, color='darkorange')


# # Gradient Boosting Classifier

# In[224]:


import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Eğitim süresi ölçümü
start_train = time.time()
gb_model = GradientBoostingClassifier(random_state=46).fit(X_train, y_train)
end_train = time.time()

# Tahmin süresi ölçümü
start_pred = time.time()
gb_pred = gb_model.predict(X_test)
end_pred = time.time()

# Skorlar
print("Gradient Boosting Classifier:")
print(f"Training Time: {round(end_train - start_train, 4)} seconds")
print(f"Prediction Time: {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy: {round(accuracy_score(gb_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(gb_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(gb_pred, y_test), 4)}")
print(f"F1 Score: {round(f1_score(gb_pred, y_test), 4)}")
print(f"AUC: {round(roc_auc_score(gb_pred, y_test), 4)}")

# Confusion matrix görselleştirme
cm = confusion_matrix(y_test, gb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')


# In[225]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Olasılık tahminleri
gb_probs = gb_model.predict_proba(X_test)[:, 1]

# ROC eğrisi için FPR, TPR değerleri
fpr, tpr, thresholds = roc_curve(y_test, gb_probs)
roc_auc = auc(fpr, tpr)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Gradient Boosting Classifier')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# # XGBoost Classifier

# In[226]:


import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Eğitim süresi ölçümü
start_train = time.time()
xgb_model = XGBClassifier(random_state=46).fit(X_train, y_train)
end_train = time.time()

# Tahmin süresi ölçümü
start_pred = time.time()
xgb_pred = xgb_model.predict(X_test)
end_pred = time.time()

# Performans metrikleri
print("XGBoost Classifier:")
print(f"Training Time: {round(end_train - start_train, 4)} seconds")
print(f"Prediction Time: {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy: {round(accuracy_score(xgb_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(xgb_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(xgb_pred, y_test), 4)}")
print(f"F1 Score: {round(f1_score(xgb_pred, y_test), 4)}")
print(f"AUC: {round(roc_auc_score(xgb_pred, y_test), 4)}")

# Confusion matrix görselleştirme
cm = confusion_matrix(y_test, xgb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')


# In[227]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Olasılık tahminleri (pozitif sınıf için)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# ROC eğrisi için FPR, TPR ve threshold hesaplama
fpr, tpr, thresholds = roc_curve(y_test, xgb_probs)
roc_auc = auc(fpr, tpr)

# ROC eğrisi çizimi
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost Classifier')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# In[228]:


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title(f'Feature Importance - {model.__class__.__name__}')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')
plot_importance(xgb_model, X)


# # LightGBM Classifier

# In[229]:


import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Eğitim süresi ölçümü
start_train = time.time()
lgbm_model = LGBMClassifier(random_state=46).fit(X_train, y_train)
end_train = time.time()

# Tahmin süresi ölçümü
start_pred = time.time()
lgbm_pred = lgbm_model.predict(X_test)
end_pred = time.time()

# Performans metrikleri
print("LightGBM Classifier:")
print(f"Training Time: {round(end_train - start_train, 4)} seconds")
print(f"Prediction Time: {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy: {round(accuracy_score(lgbm_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(lgbm_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(lgbm_pred, y_test), 4)}")
print(f"F1 Score: {round(f1_score(lgbm_pred, y_test), 4)}")
print(f"AUC: {round(roc_auc_score(lgbm_pred, y_test), 4)}")

# Confusion matrix görselleştirme
cm = confusion_matrix(y_test, lgbm_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')


# In[230]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# LightGBM için pozitif sınıf olasılık tahminleri
lgbm_probs = lgbm_model.predict_proba(X_test)[:, 1]

# ROC eğrisi için FPR, TPR ve eşik değerler
fpr, tpr, thresholds = roc_curve(y_test, lgbm_probs)
roc_auc = auc(fpr, tpr)

# ROC eğrisini çiz
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LightGBM Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# # CatBoostClassifier

# In[231]:


import time
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

# Model eğitimi zamanı
start_train = time.time()
cat_model = CatBoostClassifier(random_state=46, verbose=False).fit(X_train, y_train)
end_train = time.time()

# Tahmin zamanı
start_pred = time.time()
cat_pred = cat_model.predict(X_test)
end_pred = time.time()

# Performans metrikleri yazdırma
print("CatBoost Classifier:")
print(f"Training Time: {round(end_train - start_train, 4)} seconds")
print(f"Prediction Time: {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy: {round(accuracy_score(cat_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(cat_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(cat_pred, y_test), 4)}")
print(f"F1 Score: {round(f1_score(cat_pred, y_test), 4)}")
print(f"AUC: {round(roc_auc_score(cat_pred, y_test), 4)}")

# Confusion matrix görselleştirme
cm = confusion_matrix(y_test, cat_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

# ROC eğrisi
cat_probs = cat_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, cat_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - CatBoost Classifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# # HistGradientBoostingClassifier

# In[232]:


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

# Model eğitimi ve zaman ölçümü
start_train = time.time()
hgb_model = HistGradientBoostingClassifier(random_state=46).fit(X_train, y_train)
end_train = time.time()

# Tahmin ve zaman ölçümü
start_pred = time.time()
hgb_pred = hgb_model.predict(X_test)
hgb_pred_proba = hgb_model.predict_proba(X_test)[:, 1]
end_pred = time.time()

# Performans metrikleri
print("HistGradientBoostingClassifier:")
print(f"Training Time: {round(end_train - start_train, 4)} seconds")
print(f"Prediction Time: {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy: {round(accuracy_score(y_test, hgb_pred), 4)}")
print(f"Recall: {round(recall_score(y_test, hgb_pred), 4)}")
print(f"Precision: {round(precision_score(y_test, hgb_pred), 4)}")
print(f"F1 Score: {round(f1_score(y_test, hgb_pred), 4)}")
print(f"AUC: {round(roc_auc_score(y_test, hgb_pred_proba), 4)}")

# Confusion Matrix
cm = confusion_matrix(y_test, hgb_pred)
print("Confusion Matrix:")
print(cm)

cm = confusion_matrix(y_test, cat_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')

# ROC Curve çizimi
fpr, tpr, thresholds = roc_curve(y_test, hgb_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - HistGradientBoostingClassifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# | Model                     | Training Time (s) |
# | ------------------------- | ----------------- |
# | K-Nearest Neighbors (KNN) | 0.0548            |
# | Decision Tree Classifier  | 0.5491            |
# | Logistic Regression       | 1.6832            |
# | XGBoost Classifier        | 1.9941            |
# | LightGBM Classifier       | 2.7916            |
# | HistGradientBoosting      | 3.9917            |
# | AdaBoost Classifier       | 21.8488           |
# | CatBoost Classifier       | 45.1831           |
# | Gradient Boosting         | 45.892            |
# | Support Vector Classifier | 555.9797          |
# 

# | Model                     | Prediction Time (s) |
# | ------------------------- | ------------------- |
# | Logistic Regression       | 0.0209              |
# | Decision Tree Classifier  | 0.0281              |
# | XGBoost Classifier        | 0.0846              |
# | CatBoost Classifier       | 0.1711              |
# | Gradient Boosting         | 0.3101              |
# | LightGBM Classifier       | 0.3134              |
# | HistGradientBoosting      | 0.9813              |
# | AdaBoost Classifier       | 1.3673              |
# | Support Vector Classifier | 31.0293             |
# | K-Nearest Neighbors (KNN) | 156.8432            |

# | Model                     | Accuracy |
# | ------------------------- | -------- |
# | Decision Tree Classifier  | 1.0000   |
# | AdaBoost Classifier       | 1.0000   |
# | Gradient Boosting         | 1.0000   |
# | XGBoost Classifier        | 1.0000   |
# | LightGBM Classifier       | 1.0000   |
# | CatBoost Classifier       | 1.0000   |
# | HistGradientBoosting      | 1.0000   |
# | K-Nearest Neighbors (KNN) | 0.9988   |
# | Logistic Regression       | 0.9963   |
# | Support Vector Classifier | 0.9963   |
# 

# | Model                     | Recall |
# | ------------------------- | ------ |
# | Decision Tree Classifier  | 1.0000 |
# | AdaBoost Classifier       | 1.0000 |
# | Gradient Boosting         | 1.0000 |
# | XGBoost Classifier        | 1.0000 |
# | LightGBM Classifier       | 1.0000 |
# | CatBoost Classifier       | 1.0000 |
# | HistGradientBoosting      | 1.0000 |
# | K-Nearest Neighbors (KNN) | 0.6916 |
# | Logistic Regression       | 0.0000 |
# | Support Vector Classifier | 0.0000 |
# 

# | Model                     | Precision |
# | ------------------------- | --------- |
# | Decision Tree Classifier  | 1.0000    |
# | AdaBoost Classifier       | 1.0000    |
# | Gradient Boosting         | 1.0000    |
# | XGBoost Classifier        | 1.0000    |
# | LightGBM Classifier       | 1.0000    |
# | CatBoost Classifier       | 1.0000    |
# | HistGradientBoosting      | 1.0000    |
# | K-Nearest Neighbors (KNN) | 0.9581    |
# | Logistic Regression       | 0.0000    |
# | Support Vector Classifier | 0.0000    |
# 

# | Model                     | F1 Score |
# | ------------------------- | -------- |
# | Decision Tree Classifier  | 1.0000   |
# | AdaBoost Classifier       | 1.0000   |
# | Gradient Boosting         | 1.0000   |
# | XGBoost Classifier        | 1.0000   |
# | LightGBM Classifier       | 1.0000   |
# | CatBoost Classifier       | 1.0000   |
# | HistGradientBoosting      | 1.0000   |
# | K-Nearest Neighbors (KNN) | 0.8033   |
# | Logistic Regression       | 0.0000   |
# | Support Vector Classifier | 0.0000   |
# 

# | Model                     | AUC Score |
# | ------------------------- | --------- |
# | Decision Tree Classifier  | 1.0000    |
# | AdaBoost Classifier       | 1.0000    |
# | Gradient Boosting         | 1.0000    |
# | XGBoost Classifier        | 1.0000    |
# | LightGBM Classifier       | 1.0000    |
# | CatBoost Classifier       | 1.0000    |
# | HistGradientBoosting      | 1.0000    |
# | K-Nearest Neighbors (KNN) | 0.8458    |
# | Logistic Regression       | 0.5000    |
# | Support Vector Classifier | 0.5000    |
# 

# **Final Model Decision Analysis**
# 
# *Model	Training Time (s)*
# 
# K-Nearest Neighbors (KNN)	0.0548
# 
# Decision Tree Classifier	0.5491
# 
# Logistic Regression	1.6832
# 
# XGBoost Classifier	1.9941
# 
# LightGBM Classifier	2.7916
# 
# HistGradientBoosting	3.9917
# 
# *Model	Prediction Time (s)*
# 
# Logistic Regression	0.0209
# 
# Decision Tree Classifier	0.0281
# 
# XGBoost Classifier	0.0846
# 
# CatBoost Classifier	0.1711
# 
# Gradient Boosting	0.3101
# 
# LightGBM Classifier	0.3134
# 
# HistGradientBoosting	0.9813
# 
# AdaBoost Classifier	1.3673
# 
# *Accuracy, Recall, F1 Score, AUC Score*
# 
# Decision Tree Classifier	1.0000
# 
# AdaBoost Classifier	1.0000
# 
# Gradient Boosting	1.0000
# 
# XGBoost Classifier	1.0000
# 
# LightGBM Classifier	1.0000
# 
# CatBoost Classifier	1.0000
# 
# HistGradientBoosting1.0000
# 
# **Final Models Decision**
# 
# **Decision Tree Classifier**, **XGBoost Classifier**, **LightGBM Classifier**, and **HistGradientBoosting** are the best models for every metrics.

# * **%0.4** is **1962 **customers but we apply for all process and model implementations. 
# * We decrease customers from 1962 to **694**. We get a lot of customers again thanks to models.(**1268 customers**).
# * Finally, **694** customers are risk for bank and this is %0.1414882772680938.(**%0.14**)

# # ROC Curve Analysis With All Models

# In[233]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

models = {
    "Logistic Regression": lr_model,
    "Random Forest": rf_model,
    "KNN": knn_model,
    "SVC": svc_model,
    "Decision Tree": dt_model,
    "AdaBoost": ada_model,
    "Gradient Boosting": gb_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "CatBoost": cat_model,
    "HistGradientBoosting": hgb_model
}

for name, model in models.items():
    # SVC gibi bazı modellerde predict_proba yoktur, decision_function kullanılır
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Classifiers')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# In[234]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

plt.figure(figsize=(10, 8))

models = {
    "Decision Tree": dt_model,
    "XGBoost": xgb_model,
    "LightGBM": lgbm_model,
    "HistGradientBoosting": hgb_model
}

for name, model in models.items():
    # SVC gibi bazı modellerde predict_proba yoktur, decision_function kullanılır
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison of Classifiers')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# # Shap Value Analysis

# In[235]:


import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# In[236]:


import shap

explainer = shap.TreeExplainer(dt_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# In[237]:


import shap

explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# In[238]:


import shap

explainer = shap.TreeExplainer(hgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# **Status** has a big role of this data becasue feature importance and shap value analysis show that how much important.  

# # Hiperparametre Ayarlaması - Decision Tree

# In[239]:


import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)

# Hiperparametre ayarları
dt_model = DecisionTreeClassifier(
    criterion='gini',        # 'entropy' de denenebilir
    max_depth=5,             # ağacın maksimum derinliği
    min_samples_split=10,    # düğüm bölünmeden önce en az 10 örnek olmalı
    min_samples_leaf=5,      # yaprakta en az 5 örnek olmalı
    max_features='sqrt',     # her bölünmede değerlendirilecek özellik sayısı
    class_weight='balanced', # dengesiz sınıflar varsa sınıf ağırlığını dengeler
    random_state=46
)

# Model eğitimi
start_train = time.time()
dt_model.fit(X_train, y_train)
end_train = time.time()
training_time = end_train - start_train

# Tahmin
start_pred = time.time()
dt_pred = dt_model.predict(X_test)
end_pred = time.time()
prediction_time = end_pred - start_pred

# Performans çıktıları
print("🌲 Decision Tree Classifier Results (Tuned):")
print(f"Training Time   : {training_time:.4f} seconds")
print(f"Prediction Time : {prediction_time:.4f} seconds")
print(f"Accuracy        : {accuracy_score(y_test, dt_pred):.4f}")
print(f"Recall          : {recall_score(y_test, dt_pred):.4f}")
print(f"Precision       : {precision_score(y_test, dt_pred):.4f}")
print(f"F1 Score        : {f1_score(y_test, dt_pred):.4f}")
print(f"AUC Score       : {roc_auc_score(y_test, dt_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, dt_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")


# # Hiperparametre Ayarlaması - XGBClassifier

# In[240]:


import time
import numpy as np
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# 1. Hiperparametre alanı tanımı
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

# 2. RandomizedSearchCV kurulumu
xgb = XGBClassifier(random_state=46, use_label_encoder=False, eval_metric='logloss')
random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=25,              # 25 farklı kombinasyon deneyecek
    scoring='roc_auc',      # Değerlendirme metriği
    cv=5,                   # 5-fold CV
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# 3. Model eğitimi
start_train = time.time()
random_search.fit(X_train, y_train)
end_train = time.time()

# En iyi modeli al
best_xgb = random_search.best_estimator_
print("🔧 En iyi hiperparametreler:")
print(random_search.best_params_)
print(f"GridSearch Training Time: {round(end_train - start_train, 2)} saniye\n")

# 4. Tahmin ve metrikler
start_pred = time.time()
xgb_pred = best_xgb.predict(X_test)
end_pred = time.time()

print("🌲 XGBoost Classifier with RandomizedSearchCV:")
print(f"Training Time   : {round(end_train - start_train, 4)} seconds")
print(f"Prediction Time : {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy        : {round(accuracy_score(y_test, xgb_pred), 4)}")
print(f"Recall          : {round(recall_score(y_test, xgb_pred), 4)}")
print(f"Precision       : {round(precision_score(y_test, xgb_pred), 4)}")
print(f"F1 Score        : {round(f1_score(y_test, xgb_pred), 4)}")
print(f"AUC Score       : {round(roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:,1]), 4)}")

# 5. Confusion Matrix
cm = confusion_matrix(y_test, xgb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')


# # Hiperparametre Ayarlaması - LGBMClassifier

# In[241]:


import time
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
param_dist = {
    'num_leaves': [20, 31, 50, 70],
    'max_depth': [-1, 5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'min_child_samples': [10, 20, 30, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

lgbm = LGBMClassifier(random_state=46)
random_search = RandomizedSearchCV(
    estimator=lgbm,
    param_distributions=param_dist,
    n_iter=25,                # 25 farklı kombinasyon denenir
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)
# Eğitim süresi
start_train = time.time()
random_search.fit(X_train, y_train)
end_train = time.time()

# En iyi modeli al
best_lgbm = random_search.best_estimator_
print("🔧 En iyi hiperparametreler:")
print(random_search.best_params_)
print(f"GridSearch Training Time: {round(end_train - start_train, 2)} saniye\n")

# Tahmin süresi
start_pred = time.time()
lgbm_pred = best_lgbm.predict(X_test)
end_pred = time.time()

# Performans çıktıları
print("🌿 LightGBM Classifier with RandomizedSearchCV:")
print(f"Training Time   : {round(end_train - start_train, 4)} seconds")
print(f"Prediction Time : {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy        : {round(accuracy_score(y_test, lgbm_pred), 4)}")
print(f"Recall          : {round(recall_score(y_test, lgbm_pred), 4)}")
print(f"Precision       : {round(precision_score(y_test, lgbm_pred), 4)}")
print(f"F1 Score        : {round(f1_score(y_test, lgbm_pred), 4)}")
print(f"AUC Score       : {round(roc_auc_score(y_test, best_lgbm.predict_proba(X_test)[:,1]), 4)}")
cm = confusion_matrix(y_test, lgbm_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')


# # Hiperparametre Ayarlaması - HistGradientBoostingClassifier

# In[242]:


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve, auc,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import time
import numpy as np
param_dist = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_iter': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 15],
    'min_samples_leaf': [10, 20, 30, 50],
    'l2_regularization': [0, 0.1, 1, 10],
    'max_bins': [255, 512, 1024]
}
hgb = HistGradientBoostingClassifier(random_state=46)

random_search = RandomizedSearchCV(
    estimator=hgb,
    param_distributions=param_dist,
    n_iter=25,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
# Eğitim süresi ölçümü
start_train = time.time()
random_search.fit(X_train, y_train)
end_train = time.time()

# En iyi model
best_hgb = random_search.best_estimator_
print("🔧 En iyi hiperparametreler:")
print(random_search.best_params_)
print(f"Training Time: {round(end_train - start_train, 4)} seconds\n")

# Tahmin süresi
start_pred = time.time()
hgb_pred = best_hgb.predict(X_test)
hgb_pred_proba = best_hgb.predict_proba(X_test)[:, 1]
end_pred = time.time()
print("HistGradientBoostingClassifier (with RandomizedSearchCV):")
print(f"Prediction Time: {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy        : {round(accuracy_score(y_test, hgb_pred), 4)}")
print(f"Recall          : {round(recall_score(y_test, hgb_pred), 4)}")
print(f"Precision       : {round(precision_score(y_test, hgb_pred), 4)}")
print(f"F1 Score        : {round(f1_score(y_test, hgb_pred), 4)}")
print(f"AUC Score       : {round(roc_auc_score(y_test, hgb_pred_proba), 4)}")
# Confusion Matrix
cm = confusion_matrix(y_test, hgb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - HistGradientBoostingClassifier")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, hgb_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0,1])
plt.ylim([0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - HistGradientBoostingClassifier')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# # HistGradientBoostingClassifier - Optuna ile Hiperparametre Optimizasyonu

# In[244]:


import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve, auc,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import time

def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "max_iter": trial.suggest_int("max_iter", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50),
        "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 10.0),
        "max_bins": trial.suggest_int("max_bins", 64, 255)  # ✅ sadece geçerli aralık
    }

    model = HistGradientBoostingClassifier(**params, random_state=46)
    return cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=3).mean()

study = optuna.create_study(direction="maximize")
start_train = time.time()
study.optimize(objective, n_trials=25, timeout=300)
end_train = time.time()

best_params = study.best_params
print("🔧 En iyi hiperparametreler (Optuna):")
print(best_params)
print(f"Training Time: {round(end_train - start_train, 4)} seconds\n")

best_hgb = HistGradientBoostingClassifier(**best_params, random_state=46)
best_hgb.fit(X_train, y_train)

start_pred = time.time()
hgb_pred = best_hgb.predict(X_test)
hgb_pred_proba = best_hgb.predict_proba(X_test)[:, 1]
end_pred = time.time()

print("HistGradientBoostingClassifier (Optuna):")
print(f"Prediction Time: {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy        : {round(accuracy_score(y_test, hgb_pred), 4)}")
print(f"Recall          : {round(recall_score(y_test, hgb_pred), 4)}")
print(f"Precision       : {round(precision_score(y_test, hgb_pred), 4)}")
print(f"F1 Score        : {round(f1_score(y_test, hgb_pred), 4)}")
print(f"AUC Score       : {round(roc_auc_score(y_test, hgb_pred_proba), 4)}")

cm = confusion_matrix(y_test, hgb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Optuna Tuned HGB")
plt.show()

fpr, tpr, _ = roc_curve(y_test, hgb_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Optuna Tuned HGB')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# * max_bins sınırlandırması HistGradientBoostingClassifier’a özgüdür, LightGBM gibi modellerde bu 1024'e kadar çıkabilir.
# 
# * Kaggle’da çalıştırırken n_trials değerini düşük tutarsan (örneğin 25 gibi), işlem süresi daha kısa olur. Daha yüksek doğruluk istiyorsan 50 veya 100'e kadar çıkarabilirsin.
# 
# * timeout=300 saniye ile sınırlandırılmış durumda; daha uzun süre tanımlayarak daha fazla deneme yapılmasını sağlayabilirsin.

# | Amaç                     | Optuna          | BayesSearchCV    |
# | ------------------------ | --------------- | ---------------- |
# | Daha akıllı öğrenme      | ✅               | ✅                |
# | Daha fazla esneklik      | ✅               | ❌ (sınırlı yapı) |
# | Daha kısa kod            | ❌               | ✅                |
# | Kaggle’da hızlı deneme   | ✅ (düşük trial) | ✅                |
# | GridSearch'e benzer yapı | ❌               | ✅                |
# 

# 💡 Tavsiye:
# Kaggle Notebook çalıştırırken Optuna / BayesSearchCV için:
# 
# n_iter=25 ideal.
# 
# Cross-validation katı cv=3 yeterlidir.
# 
# Daha uzun ve kapsamlı tuning için lokal sistemde eğitim, sonra Kaggle’a model yükleme daha verimlidir.

# | Özellik            | Optuna                                | BayesSearchCV (skopt)         |
# | ------------------ | ------------------------------------- | ----------------------------- |
# | Arama Stratejisi   | Akıllı (Tree-structured Parzen Est.)  | Gaussian Process / Surrogate  |
# | Kullanım Kolaylığı | Daha esnek (ama biraz daha fazla kod) | GridSearchCV benzeri, kolay   |
# | Kaggle Uyum        | Evet                                  | Evet                          |
# | GPU Desteği        | GPU destekli modellerde işe yarar     | Evet, ama model özelinde      |
# | Performans         | Daha iyi arama sonuçları              | İyi, ama daha sınırlı öğrenme |

# | Teknik                             | Açıklama                                      |
# | ---------------------------------- | --------------------------------------------- |
# | `n_trials=25` gibi düşük tut       | 25–50 arasında başla, sonra artırırsın        |
# | `timeout=300` (5 dakika sınırı)    | Zaman bazlı kısıtlama ile daha hızlı sonlanır |
# | `direction='maximize'`             | ROC-AUC gibi metriklerde kullanılmalı         |
# | `pruner=SuccessiveHalvingPruner()` | Kötü modeller erken durur, süreci hızlandırır |
# 

# # HistGradientBoostingClassifier - Bayesian Optimization (BayesSearchCV) ile Hiperparametre

# In[246]:


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, roc_auc_score, roc_curve, auc,
                             confusion_matrix, ConfusionMatrixDisplay)
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import matplotlib.pyplot as plt
import time

# Parametre aralıklarını düzeltilmiş şekilde tanımlıyoruz:
param_space = {
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'max_iter': Integer(100, 500),
    'max_depth': Integer(3, 15),
    'min_samples_leaf': Integer(10, 50),
    'l2_regularization': Real(0.0, 10.0),
    'max_bins': Integer(32, 255)  # ✅ 255 sınırına uyan aralık
}

hgb = HistGradientBoostingClassifier(random_state=46)

# BayesSearchCV tanımlama
opt = BayesSearchCV(
    estimator=hgb,
    search_spaces=param_space,
    scoring='roc_auc',
    n_iter=25,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

# Eğitim süresi ölçümü
start_train = time.time()
opt.fit(X_train, y_train)
end_train = time.time()

# En iyi model
best_hgb = opt.best_estimator_
print("🔧 En iyi hiperparametreler (BayesSearchCV):")
print(opt.best_params_)
print(f"Training Time: {round(end_train - start_train, 4)} seconds\n")

# Tahmin süresi
start_pred = time.time()
hgb_pred = best_hgb.predict(X_test)
hgb_pred_proba = best_hgb.predict_proba(X_test)[:, 1]
end_pred = time.time()

# Performans metrikleri
print("HistGradientBoostingClassifier (BayesSearchCV):")
print(f"Prediction Time: {round(end_pred - start_pred, 4)} seconds")
print(f"Accuracy        : {round(accuracy_score(y_test, hgb_pred), 4)}")
print(f"Recall          : {round(recall_score(y_test, hgb_pred), 4)}")
print(f"Precision       : {round(precision_score(y_test, hgb_pred), 4)}")
print(f"F1 Score        : {round(f1_score(y_test, hgb_pred), 4)}")
print(f"AUC Score       : {round(roc_auc_score(y_test, hgb_pred_proba), 4)}")

# Confusion Matrix
cm = confusion_matrix(y_test, hgb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - BayesSearchCV Tuned HGB")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, hgb_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Bayes Tuned HGB')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# 💡 Ek Öneriler:
# 
# * n_iter=25: Bayes optimizasyonun süresi ve performansı için kritik. Bunu 50–100’e çıkarırsan daha iyi sonuç alabilirsin ama süre uzar.
# 
# * cv=3: Kısa sürede sonuç almak için 3 fold kullandık. Daha doğru sonuç için cv=5 veya cv=10 denenebilir.
# 
# * verbose=0: Gereksiz çıktıyı engeller. Debug için verbose=1 yapabilirsin.

# **Final Değerlendirmesi, Sonuçlar Ve Tavsiyeler**
# 
# * %0.4 ile başlangıçta 1962 olan riskli müşteri sayısı 694'e düşürülmüştür.(%0.14)
# 
# * 1268 müşteri doğru yere konumlandırılmış ve bu müşterilere kredi verilmesi veri bilimi takımı tarafından ilgili bankacılık departmanına tavsiye edilmiştir.
# 
# * 537666 müşteri verisi olan bu projede 1268 müşteri kazanımı elde edilmiştir.(%0.00235834)(%0.0023)
# 
# * Yüzde olarak küçük gözükse de çalıştığımız veri seti banka ile ilgili bir veri olduğu için müşterilerin kazanımı ve doğru değerlendirilmesi oldukça önemlidir.Bir aylık veri seti olduğu varsaydığımızda her ay böyle proje çalışmaları yapıldığında 1 senede 1268*12=15.216 müşteri demektir.(+- %10 kayıp:13.694 kişi)
# 
# * Her projede aynı sayıları yakalamak ve tespit etmek mümkün değilse de ortalama bir senede 10.000 müşteri kazanımı çok anlamlı bir sayıdır.Bu müşteriler bize başvuran kredi başvurusu yapan ve sonuçlarını isteyen müşterilerdir ve biz bu müşterileri kazanmak için bir çalışma gerçekleştirdik.
# 
# * Özellik mühendisliği ve hiperparametre ayarlamaları sonuç vermemiştir.
