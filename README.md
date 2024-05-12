# Feature-Engineering-Heart-Attack-dataset

Performed the following tasks on the heartattack dataset:

## Feature Selection:

### Evaluating the importance of features using techniques like feature importance scores or feature ranking:
We used feature importance score by using a Random Forest Classifier. Though it was not a big help in our dataset.
### Remove irrelevant or redundant features to simplify models and reduce overfitting:
We used z-scores to handle and remove outliers.
  

## Feature Creation:
We generated random **height** and **weight** values to include in our dataset for experimenting with features.
### Engineer new features by combining existing variables or applying domain-specific knowledge:
We engineered new features and generated : BMI, pulse pressure and categorizing conditions based on Blood pressure (normal, pre-hypertension and hypertension) by applying domain-specific knowledge and also visually explored the new features and data using data visualization.

## Data Transformation:
### Handle categorical variables through one-hot encoding, label encoding, or embedding: 
We have a column **BP_based_condition** that has categorical values like 'normal', 'hypertension', and 'prehypertension', therefore we performed one-hot encoding to convert these categories into binary variables using:
```python
pd.dummies()
```
```python
cleaned_df_encoded = pd.get_dummies(cleaned_df, columns=['BP_based_Condition'])
```
 
