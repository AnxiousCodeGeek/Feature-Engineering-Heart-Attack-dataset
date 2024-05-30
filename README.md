# Heart Attack Prediction Project: Feature-Engineering
This project focuses on analyzing a heart attack dataset, performing feature selection, cleaning the data, and enriching it with additional features. A RandomForestClassifier is trained to predict the likelihood of a heart attack based on the available features. The project includes tasks such as feature selection, feature engineering, outlier detection, and data transformation, accompanied by visualizations to aid in understanding the data.

## Software Requirements
 - Python version 3.7 or higher
 - ```Pandas``` library
 - ```Scikit-learn``` library for machine learning models and statistical modelling
 - ```Matplotlib library``` for data visualizations
 - ```Seaborn``` library for statistical graphs
 - ```Numpy``` library for dealing with arrays mathematically

## Description
Performed the following tasks on the heartattack dataset:

### Feature Selection:

 - **Evaluating the importance of features using techniques like feature importance scores or feature ranking:** We used feature importance score by using a Random Forest Classifier. Though it was not a big help in our dataset.
```python
X = df[['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure']]
y = df['Result']

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
```

 - **Remove irrelevant or redundant features to simplify models and reduce overfitting:** We used z-scores to handle and remove outliers.
```python
z_scores = (df - df.mean()) / df.std()
outliers = (np.abs(z_scores) > 3).any(axis=1)
cleaned_df = df[~outliers]
```

### Feature Creation:
 - We generated random synthetic data for ```height``` and ```weight``` to include in our dataset for experimenting with features.
```python
np.random.seed(0)
random_weight = np.random.normal(loc=70, scale=10, size=len(cleaned_df))
random_height = np.random.normal(loc=1.7, scale=0.1, size=len(cleaned_df))

cleaned_df['weight'] = random_weight
cleaned_df['height'] = random_height
```

 - **Engineer new features by combining existing variables or applying domain-specific knowledge:** We engineered new features and generated ```BMI```, ```pulse_pressure``` and ```BP_based_Condition``` categorizing conditions based on Blood pressure (normal, pre-hypertension and hypertension) by applying domain-specific knowledge and also visually explored the new features and data using data visualization.

```python
cleaned_df['BMI'] = cleaned_df['weight'] / (cleaned_df['height'] ** 2)
cleaned_df['pulse_pressure'] = cleaned_df['Systolic blood pressure'] - cleaned_df['Diastolic blood pressure']

def categorize_blood_pressure(row):
    if row['Systolic blood pressure'] < 120 and row['Diastolic blood pressure'] < 80:
        return 'Normal'
    elif 120 <= row['Systolic blood pressure'] < 130 or 80 <= row['Diastolic blood pressure'] < 85:
        return 'Pre-Hypertension'
    else:
        return 'Hypertension'

cleaned_df['BP_based_Condition'] = cleaned_df.apply(categorize_blood_pressure, axis=1)
```

### Data Transformation:
 - **Handle categorical variables through one-hot encoding, label encoding, or embedding:** We have a column ```BP_based_condition``` that has categorical values like ```normal```, ```hypertension```, and ```prehypertension ```, therefore we performed one-hot encoding to convert these categories into binary variables using:

```python
cleaned_df_encoded = pd.get_dummies(cleaned_df, columns=['BP_based_Condition'])
```
## Results
The processed dataset is ready for further analysis or model training. Key insights and visualizations are included in the script.
