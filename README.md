# Heart Disease Analysis

## Table of Contents
- [Project Overview](#project-overview)
- [Problem Motivation](#problem-motivation)
- [Approach](#approach)
- [Approach Details](#approach-details)
- [Key Metrics Analyzed](#key-metrics-analyzed)
- [Data Sources](#data-sources)
- [Tools](#tools)
- [Data Cleaning](#data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Analysis ](#data-analysis)
- [Results](#results)


### Project Overview
Heart diseases, which are heart conditions such as diseased vessels, structural problems, and blood clots, are one of the leading causes of death worldwide. Early detection and prevention is therefore important. The task is to determine the most impactful factors that lead to heart disease.

### Problem Motivation
Finding the biggest causes of heart disease puts us a step ahead in many ways:

Prevention focus: Knowing the primary causes allows for targeted public health initiatives to educate people on reducing their risk of developing heart disease

Policy changes: Research on heart disease causes can direct policy decisions to promote healthier lifestyles, regulate products like tobacco, and more

Early detection and treatment: By understanding the major factors that lead to heart disease, healthcare professionals would be able to identify high risk individuals and guide them towards lifestyle changes or medication to prevent heart disease

### Approach 
I used a Random Forest model because it’s a powerful machine learning method that combines multiple decision trees to make more accurate predictions, especially if our data does not follow a linear trend. One of its strengths is that it reduces the risk of overfitting by averaging the results of all the trees. This makes it a reliable option for handling complex data sets. Random Forests shows which variables are the most important in determining the outcome. For example, our model ranked predictors like age and cholesterol at the top. 

### Approach Details
For the Random Forest, I used 100 trees with the default settings for other parameters like tree depth. This allowed me to capture important patterns in the data without overcomplicating the model. 

Our dataset contained roughly 4000 individual rows, and I split the dataset into 70% for training the data and 30% for the actual testing.  This ensured that the model was trained on a substantial portion of the data but still tested on unseen data to validate its performance.


### Key Metrics Analyzed:
Body Mass Index (BMI): A measure of body fat based on height and weight.  
Total Cholesterol (totChol): The total amount of cholesterol in the blood, including HDL and LDL.  
Blood Glucose (glucose): The concentration of sugar in the blood, used to assess diabetes risk.  
Heart Rate (heartRate): The number of heartbeats per minute, indicating cardiovascular health.  
Cigarettes Per Day (cigsPerDay): The average number of cigarettes smoked daily.  
Smoking Status (is_smoking): Indicates whether an individual is currently smoking.  
Diabetes Diagnosis (diabetes): Identifies whether a person has been diagnosed with diabetes.  

### Data Sources
Cardiovascular Risk Data: The primary dataset used for this analysis is the "data_cardiovascular_risk.csv" file, containing biometric data of many patients. 

### Tools
- Excel - Data cleaning
- Python - Analysis

### Data Cleaning
In the initial data preparation phase, I performed the following tasks:

1. Data loading and inspection.  
2. Handling missing values.  
3. Data cleaning and formatting.

### Exploratory Data Analysis

EDA involved exploring the data_cardiovascular_risk dataset to answer a key question:  

How do biometric measurements correlate with the risk of heart disease progression?  


### Data Analysis

``` Python
import kagglehub
import os  # Import the os module
import pandas as pd


# Download latest version
path = kagglehub.dataset_download("mamta1999/cardiovascular-risk-data")

print("Path to dataset files:", path)

# Load and display the CSV file
csv_file = os.path.join(path, "data_cardiovascular_risk.csv")
df = pd.read_csv(csv_file)
df  # Display the first few rows

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Preprocessing
# Drop unwanted columns: 'Prevalent Stroke', 'id', 'Sys BP', 'Dia BP', and the target variable
X = df.drop(columns=["prevalentStroke", "id", "sysBP", "diaBP", "TenYearCHD", "age"])

# Encode categorical variables
categorical_columns = ["sex", "is_smoking", "education"]
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Target variable
y = df["TenYearCHD"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Feature Importance
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({"Predictor": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Predictor"], importance_df["Importance"], color="skyblue")
plt.xlabel("Predictor Importance")
plt.ylabel("Predictors")
plt.title("Predictor Importance for Predicting Heart Disease (Random Forest) ")
plt.gca().invert_yaxis()
plt.show()

# Display top predictors
print(importance_df.head())

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure sysBP is standardized
sns.set(style="whitegrid")

# Boxplot to show distribution of sysBP by TenYearCHD
plt.figure(figsize=(10, 6))
sns.boxplot(x='TenYearCHD', y='BMI', data=df, palette='coolwarm')
plt.title('Distribution of BMI by Ten-Year CHD Risk')
plt.xlabel('Ten-Year CHD Risk (0 = No, 1 = Yes)')
plt.ylabel('BMI')
plt.show()

# Ensure BMI is standardized
sns.set(style="whitegrid")

# Boxplot to show distribution of BMI by TenYearCHD
plt.figure(figsize=(10, 6))
sns.boxplot(x='TenYearCHD', y='totChol', data=df, palette='coolwarm')
plt.title('Distribution of Total Cholesterol by Ten-Year CHD Risk')
plt.xlabel('Ten-Year CHD Risk (0 = No, 1 = Yes)')
plt.ylabel('Total Cholsterol in mg/dL')
plt.show()

# Ensure totChol is standardized
sns.set(style="whitegrid")

# Boxplot to show distribution of totChol by TenYearCHD
plt.figure(figsize=(10, 6))
sns.boxplot(x='TenYearCHD', y='glucose', data=df, palette='coolwarm')
plt.title('Distribution of Glucose by Ten-Year CHD Risk')
plt.xlabel('Ten-Year CHD Risk (0 = No, 1 = Yes)')
plt.ylabel('Glucose Level mg/dL')
plt.show()
```

### Results
In conclusion, I demonstrated that machine learning algorithms such as random forests can predict heart disease well, with BMI and cholesterol being the most important factors. BMI (Body Mass Index) is directly related to your health in general, which is why it makes sense that a higher BMI directly correlates to heart disease. High cholesterol blocks your arteries leading to reduced blood flow to the heart, explaining why it is also a good predictor of heart disease.

These insights can help doctors make more informed decisions and improve patient care specifically with heart disease. 












