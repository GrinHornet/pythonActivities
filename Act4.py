#Activity 4
#1.) 
#2.) 
#3.) 
#4.) 
#5.) 
#6.) 
#7.) 
#8.) 
#9.) 
#10.) 
#11.) 
#12.) 

# Importing libraries
import numpy as np
import pandas as pd


#load dataset
dataset = pd.read_csv('rice_yield.csv')
print(dataset.describe())

#====================For Rainfall_mm
#1.) Separate the null values
# separate the NULL values from 'Rainfall_mm' feature

    # Step 1: Filter the dataset to only include rows where "Rainfall_mm" is null
null_rainfall_data = dataset[dataset['Rainfall_mm'].isnull()]
    
    # Step 2: Select the specified columns
test_data1 = null_rainfall_data[['Average_Temperature_C', 'Rainfall_mm', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]
    
    # Optionally, you can create a new dataset with these filtered columns
    #test_data.to_csv('test_data.csv', index=False)  # Save the test data to a CSV file

#2.) Drop NULL values from 'Rainfall_mm' feature
    # Step 1: Filter out the rows where "Rainfall_mm" is null
filtered_data = dataset.dropna(subset=['Rainfall_mm'])

    # Step 2: Select the specified columns
training_data1 = filtered_data[['Average_Temperature_C', 'Rainfall_mm', 'Sunlight_Exposure_hours_per_day', 'Pest_Infestation_Severity']]

#3.) Create X_train and Y_train from the train data
   
     # for Rainfall_mm
    
    # Extract the target variable 'Rainfall_mm' as y_train
y_train_rainfall = training_data1['Rainfall_mm']
    
    # Extract the features 'Average_Temperature_C', 'Pest_Infestation_Severity', 'Sunlight_Exposure_hours_per_day' as x_train
x_train_rainfall = training_data1[['Average_Temperature_C','Sunlight_Exposure_hours_per_day','Pest_Infestation_Severity']]

#4.) Build the Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#train the model on train data set
lr.fit(x_train_rainfall,y_train_rainfall)

#5.) create the x_test from the test_data
    #for Rainfall_mm
    
    # Extract the target variable 'Rainfall_mm' as y_test
y_test_rainfall = test_data1['Rainfall_mm']
    
    # Extract the features 'Average_Temperature_C', 'Pest_Infestation_Severity', 'Sunlight_Exposure_hours_per_day' as x_test
x_test_rainfall = test_data1[['Average_Temperature_C','Sunlight_Exposure_hours_per_day','Pest_Infestation_Severity']]
    
#6.) Apply the model on x_test_rainfall and predicting missing values for age
y_predicted_rainfall = lr.predict(x_test_rainfall)
y_predicted_rainfall


#7.) Replace the Missing values with predicted values
test_data1.loc[test_data1.Rainfall_mm.isnull(), 'Rainfall_mm'] = y_predicted_rainfall

#8.) Merge the modified test_data1 (with imputed Rainfall_mm values) back into the original dataset
#datasetNew = pd.concat([dataset, test_data1], ignore_index=False)

#9.) Merge test_data1 with the original dataset on their indices
merged_dataset = pd.merge(dataset, test_data1, how='left', left_index=True, right_index=True, suffixes=('_original', '_imputed'))

#10.) Update the 'Rainfall_mm' column in the original dataset with the imputed values
merged_dataset['Rainfall_mm_original'].fillna(merged_dataset['Rainfall_mm_imputed'], inplace=True)

#11.) Drop the 'Rainfall_mm_imputed' column as it's no longer needed
merged_dataset.drop(columns=['Rainfall_mm_imputed','Average_Temperature_C_imputed','Sunlight_Exposure_hours_per_day_imputed','Pest_Infestation_Severity_imputed'], inplace=True)

#merged_dataset.to_csv('rice_yield2.csv', index=False)

#====================For Soil_pH
#1.) Separate the null values
# separate the NULL values from 'Soil_pH' feature

    # Step 1: Filter the dataset to only include rows where "Rainfall_mm" is null
null_soilPH_data = merged_dataset[merged_dataset['Soil_pH'].isnull()]
    
    # Step 2: Select the specified columns
test_data2 = null_soilPH_data[['Average_Temperature_C_original', 'Rainfall_mm_original', 'Soil_pH','Sunlight_Exposure_hours_per_day_original', 'Pest_Infestation_Severity_original']]
    
#2.) Drop NULL values from 'Soil_pH' feature
    # Step 1: Filter out the rows where "Soil_pH" is null
filtered_data = merged_dataset.dropna(subset=['Soil_pH'])
    
    # Step 2: Select the specified columns
training_data2 = filtered_data[['Average_Temperature_C_original', 'Rainfall_mm_original', 'Soil_pH', 'Sunlight_Exposure_hours_per_day_original', 'Pest_Infestation_Severity_original']]
   
    # Extract the target variable 'Soil_pH' as y_train
y_train_soilPH = training_data2['Soil_pH']
    
    # Extract the features 'Average_Temperature_C', 'Pest_Infestation_Severity', 'Sunlight_Exposure_hours_per_day' as x_train
x_train_soilPH = training_data2[['Average_Temperature_C_original','Rainfall_mm_original','Sunlight_Exposure_hours_per_day_original','Pest_Infestation_Severity_original']]

#4.) Build the Model
lr2 = LinearRegression()
#train the model on train data set
lr2.fit(x_train_soilPH,y_train_soilPH)

 # Extract the features 'Average_Temperature_C', 'Pest_Infestation_Severity', 'Sunlight_Exposure_hours_per_day' as x_test
x_test_soilPH = test_data2[['Average_Temperature_C_original','Rainfall_mm_original','Sunlight_Exposure_hours_per_day_original','Pest_Infestation_Severity_original']]
 
 #6.) Apply the model on x_test_rainfall and predicting missing values for age
y_predicted_soilPH = lr2.predict(x_test_soilPH)
y_predicted_soilPH
 
 #7.) Replace the Missing values with predicted values
test_data2.loc[test_data2.Soil_pH.isnull(), 'Soil_pH'] = y_predicted_soilPH
 
 #9.) Merge test_data1 with the original dataset on their indices
merged_dataset2 = pd.merge(merged_dataset, test_data2, how='left', left_index=True, right_index=True, suffixes=('_or', '_imp'))
 
 #10.) Update the 'Rainfall_mm' column in the original dataset with the imputed values
merged_dataset2['Soil_pH_or'].fillna(merged_dataset2['Soil_pH_imp'], inplace=True)
 
 #11.) Drop the 'Rainfall_mm_imputed' column as it's no longer needed
merged_dataset2.drop(columns=['Average_Temperature_C_original_imp','Rainfall_mm_original_imp','Soil_pH_imp','Sunlight_Exposure_hours_per_day_original_imp','Pest_Infestation_Severity_original_imp'], inplace=True)