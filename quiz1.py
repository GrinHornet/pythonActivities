# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 12:31:35 2024

@author: Henureh
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset
dataset = pd.read_csv('C:\\Users\\User\\Documents\\python_test\\banana_disease_data_numerical.csv')

# Separate the dataset into train data and test data for Rainfall_mm
train_data_cultivar = dataset.dropna(subset=['Cultivar Susceptibility']) # dri, gilahi natu ang Rainfall nga variable nga naay nan. Meaning, naa dri ang mga tanang nan sa Rainfall
test_data_cultivar = dataset[dataset['Cultivar Susceptibility'].isnull()] #dri, ang mga dili nan value sa Rainfall.


#--------------------------------------------------------------------------------------------------------Cultivar Susceptibility

# Function to replace missing values of Cultivar Susceptibility using linear regression
def replace_missing_cultivar(train_data, test_data): #Kibali, mao ang method sa pag separate ug pag select sa xTrain ug sa independent or atung y.
    # Separate features and target variable
    X_train = train_data[['Environmental Conditions', 'Geographical Location','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']] #Mao ning 3 atung mga dependent variables since wala silay nan values.
    y_train = train_data['Cultivar Susceptibility'] # mao ni atung first independent variable nga eFreedik nga mga nan values.
    
    # Train linear regression model
    lr = LinearRegression() #So mao ni atung LinearRegression to train our data.
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Environmental Conditions', 'Geographical Location','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Cultivar Susceptibility'] = y_pred
    return test_data

# Replace missing values of Cultivar Susceptibility
test_data_cultivar = replace_missing_cultivar(train_data_cultivar, test_data_cultivar)

# Merge the modified test data for Cultivar Susceptibility back into the original dataset using the index
dataset_imputed_cultivar = dataset.copy()
dataset_imputed_cultivar.loc[test_data_cultivar.index, 'Cultivar Susceptibility'] = test_data_cultivar['Cultivar Susceptibility']

# Print the dataset after replacing missing values of Cultivar Susceptibility
print("Dataset after replacing missing values of Cultivar:")
print(dataset_imputed_cultivar)


#-----------------------------------------------------------------------------------------------------Crop Management Practices

# Separate the dataset into train data and test data for Crop Management Practices
train_data_practices = dataset_imputed_cultivar.dropna(subset=['Crop Management Practices'])
test_data_practices = dataset_imputed_cultivar[dataset_imputed_cultivar['Crop Management Practices'].isnull()]

# Function to replace missing values of Crop Management Practices using linear regression
def replace_missing_practices(train_data, test_data):
    # Separate features and target variable
    X_train = train_data[['Environmental Conditions', 'Geographical Location','Cultivar Susceptibility','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_train = train_data['Crop Management Practices']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Environmental Conditions', 'Geographical Location','Cultivar Susceptibility','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Crop Management Practices'] = y_pred
    return test_data

# Replace missing values of Crop Management Practices
test_data_practices = replace_missing_practices(train_data_practices, test_data_practices)

# Merge the modified test data for Crop Management Practices back into the original dataset using the index
dataset_imputed_practices = dataset_imputed_cultivar.copy()
dataset_imputed_practices.loc[test_data_practices.index, 'Crop Management Practices'] = test_data_practices['Crop Management Practices']

# Print the dataset after replacing missing values of Crop Management Practices
print("\nDataset after replacing missing values of practices:")
print(dataset_imputed_practices)

#-------------------------------------------------------------------------------------------Pest Infestations

# Separate the dataset into train data and test data for Pest Infestations
train_data_infestations = dataset_imputed_practices.dropna(subset=['Pest Infestations'])
test_data_infestations = dataset_imputed_practices[dataset_imputed_practices['Pest Infestations'].isnull()]

# Function to replace missing values of Pest Infestations using linear regression
def replace_missing_infestations(train_data, test_data):
    # Separate features and target variable
    X_train = train_data[['Environmental Conditions', 'Geographical Location','Cultivar Susceptibility','Crop Management Practices','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_train = train_data['Pest Infestations']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Environmental Conditions', 'Geographical Location','Cultivar Susceptibility','Crop Management Practices','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Pest Infestations'] = y_pred
    return test_data

# Replace missing values of Pest Infestations
test_data_infestations = replace_missing_infestations(train_data_infestations, test_data_infestations)

# Merge the modified test data for Pest Infestations back into the original dataset using the index
dataset_imputed_infestations = dataset_imputed_practices.copy()
dataset_imputed_infestations.loc[test_data_infestations.index, 'Pest Infestations'] = test_data_infestations['Pest Infestations']

# Print the dataset after replacing missing values of Pest Infestations
print("\nDataset after replacing missing values of infestations:")
print(dataset_imputed_infestations)

#---------------------------------------------------------------------------------------Soil Health
#Planting Material Quality
# Separate the dataset into train data and test data for Irrigation_Frequency_per_week
train_data_sHealth = dataset_imputed_infestations.dropna(subset=['Soil Health'])
test_data_sHealth = dataset_imputed_infestations[dataset_imputed_infestations['Soil Health'].isnull()]

# Function to replace missing values of Irrigation_Frequency_per_week using linear regression
def replace_missing_sHealth(train_data, test_data):
    # Separate features and target variable
    X_train = train_data[['Environmental Conditions', 'Geographical Location','Cultivar Susceptibility','Crop Management Practices','Pest Infestations','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_train = train_data['Soil Health']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Environmental Conditions', 'Geographical Location','Cultivar Susceptibility','Crop Management Practices','Pest Infestations','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Soil Health'] = y_pred
    return test_data

# Replace missing values of Soil Health
test_data_sHealth = replace_missing_sHealth(train_data_sHealth, test_data_sHealth)

# Merge the modified test data for Soil Health back into the original dataset using the index
dataset_imputed_sHealth = dataset_imputed_infestations.copy()
dataset_imputed_sHealth.loc[test_data_sHealth.index, 'Soil Health'] = test_data_sHealth['Soil Health']

# Print the dataset after replacing missing values of Soil Health
print("\nDataset after replacing missing values of Soil Health:")
print(dataset_imputed_sHealth)

#---------------------------------------------------------------------------------------Planting Material Quality

# Separate the dataset into train data and test data for Planting Material Quality
train_data_quality = dataset_imputed_sHealth.dropna(subset=['Planting Material Quality'])
test_data_quality = dataset_imputed_sHealth[dataset_imputed_sHealth['Planting Material Quality'].isnull()]

# Function to replace missing values of Planting Material Quality using linear regression
def replace_missing_quality(train_data, test_data):
    # Separate features and target variable
    X_train = train_data[['Environmental Conditions', 'Geographical Location','Cultivar Susceptibility','Crop Management Practices','Pest Infestations','Soil Health','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_train = train_data['Planting Material Quality']
    
    # Train linear regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    # Predict missing values in the test data
    X_test = test_data[['Environmental Conditions', 'Geographical Location','Cultivar Susceptibility','Crop Management Practices','Pest Infestations','Soil Health','Cultural Practices', 'Disease History','Disease Surveillance and Monitoring']]
    y_pred = lr.predict(X_test)
    
    # Replace missing values with predicted values
    test_data['Planting Material Quality'] = y_pred
    return test_data

# Replace missing values of Planting Material Quality
test_data_quality = replace_missing_quality(train_data_quality, test_data_quality)

# Merge the modified test data for Planting Material Quality back into the original dataset using the index
dataset_imputed_quality = dataset_imputed_sHealth.copy()
dataset_imputed_quality.loc[test_data_quality.index, 'Planting Material Quality'] = test_data_quality['Planting Material Quality']

# Print the dataset after replacing missing values of Planting Material Quality
print("\nDataset after replacing missing values of Planting Material Quality:")
print(dataset_imputed_quality)

# Save the processed dataset to CSV
dataset_imputed_quality.to_csv('quiz1.csv', index=False)

