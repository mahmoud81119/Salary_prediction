import pandas as pd
import numpy as np
import os

# Sklearn --> Preprocessing 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler , OrdinalEncoder , OneHotEncoder
from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

## Reading data
df = pd.read_csv('Salary_Data.csv')


# Rename columns
df.columns = ['Age', 'Gender', 'Education', 'Title', 'Experience', 'Salary']


# Function to Make Grouping Job Titles
def categorize_jop_title(value):
    value = str(value).lower()
    if 'software' in value or 'developer' in value:
        return 'Software/Developer'
    elif 'data' in value or 'analyst' in value or 'scientist' in value:
        return 'Data Analyst/scientist'
    elif 'manager' in value or 'director' in value or 'vp' in value:
        return 'Manager/Director/VP'
    elif 'sales' in value or 'representative' in value:
        return 'Sales'
    elif 'marketing' in value or 'social media' in value:
        return 'Marketing/Social Media'
    elif 'product' in value or 'designer' in value:
        return 'Product/Designer'
    elif 'hr' in value or 'human resources' in value:
        return 'HR/Human Resources'
    elif 'financial' in value or 'accountant' in value:
        return 'Financial/Accountant'
    elif 'project manager' in value :
        return 'Project Manager'
    elif 'it' in value or 'support' in value :
        return 'IT/Technical Support'
    elif 'operations' in value or 'supply chain' in value:
        return 'Operations/Supply Chain'
    elif 'customer service' in value or 'receptionist':
        return 'Customer Service/Receptionist'
    else:
        return 'Other'

df['Title'] = df['Title'].apply(categorize_jop_title)


# Function to make Grouping Education Level
def group_education(value):
    value = str(value).lower()
    if 'high school' in value:
        return 'High School'
    elif 'bachelor\'s' in value or 'bachelor\'s degree' in value:
        return 'Bachelors'
    elif 'master\'s' in value or 'master\'s degree'  in value:
        return 'Masters'
    elif 'phd' in value :
        return 'PHD'
    
df['Education'] = df['Education'].apply(group_education)


# Extract featuer called Positions based on Years of Experience
def positions(value):
    try:
        if value <= 5:
            return  "Junior"
        elif value <= 10:
            return "Senior"
        elif value <= 15:
            return  "Team Leader"
        else:
            return 'Manager'
    except:
        np.nan
df['Positions']  = df['Experience'].apply(positions)


# Split to features and target 
X = df.drop(columns = 'Salary' , axis = 1)
y = df['Salary']


# Split to train and test 
X_train , X_test , y_train , y_test =  train_test_split(X , y , test_size = 0.2 , shuffle = True , random_state = 45 )

## Slice the Lists
normal_cat = ['Gender', 'Education']
special_ord = ['Positions']
numerical = ['Age', 'Experience']
one_hot = ['Title']





#Numerical --> Impute median, RobustScaler
numerical_pip = Pipeline(steps=[
    ('Selector', DataFrameSelector(numerical)),
    ('imputer', SimpleImputer(strategy='median')),
    ('Scaling', RobustScaler())
])

# Category OrdinalEncoder -- > Impute most_frequent, encoding
normal_category_pip = Pipeline(steps=[
    ('Selector', DataFrameSelector(normal_cat)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoding', OrdinalEncoder())
])

Positions_lis = [['Junior', 'Senior', 'Team Leader', 'Manager']]
special_ord_pip = Pipeline(steps=[
    ('Selector', DataFrameSelector(special_ord)),
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoding', OrdinalEncoder(categories=Positions_lis))
])

# Category OneHotEncoder -- > Impute most_frequent, encoding
one_hot_pip = Pipeline(steps=[
                        ('Selector' , DataFrameSelector(one_hot)),
                       ('imputer' , SimpleImputer(strategy='most_frequent')),
                       ('incodeing' , OneHotEncoder(drop = 'first' , sparse=False) )
])  

all_pip = FeatureUnion(transformer_list=[
    ('numerical_pip', numerical_pip),
    ('normal_category_pip', normal_category_pip),
    ('special_ord_pip', special_ord_pip),
    ('one_hot_pip', one_hot_pip)
])

all_pip.fit(X_train)




# The function to process new instances

def process(X_new):

    # Convert X_new to Dataframe
    df_new = pd.DataFrame([X_new])

    df_new.columns = X_train.columns

    # Adjust the dtypes of features
    df_new['Age'] =  df_new['Age'].astype(float)
    df_new['Gender'] =  df_new['Gender'].astype(str)
    df_new['Education'] =  df_new['Education'].astype(str)
    df_new['Title'] =  df_new['Title'].astype(str)
    df_new['Experience'] =  df_new['Experience'].astype(float)

    # Feature engineering
    df_new['Positions'] = df_new['Positions'].astype(str)

    X_processed =  all_pip.transform(df_new)

    return X_processed
