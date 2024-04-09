# Credit-Card-Fraud-Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

data = pd.read_csv('creditcard.csv')

if data.isnull().values.any():
    data.dropna(inplace=True)

scaler = StandardScaler()
data[['Amount']] = scaler.fit_transform(data[['Amount']])

class_distribution = data['Class'].value_counts()

oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(data.drop('Class', axis=1), data['Class'])

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Plot class distribution before oversampling
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(class_distribution.index, class_distribution.values, color=['blue', 'red'])
plt.title('Class Distribution Before Oversampling')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(class_distribution.index, ['Non-Fraudulent', 'Fraudulent'])
plt.grid(True)

# Plot class distribution after oversampling
plt.subplot(1, 2, 2)
resampled_distribution = pd.Series(y_resampled).value_counts()
plt.bar(resampled_distribution.index, resampled_distribution.values, color=['blue', 'red'])
plt.title('Class Distribution After Oversampling (SMOTE)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(resampled_distribution.index, ['Non-Fraudulent', 'Fraudulent'])
plt.grid(True)

plt.tight_layout()
plt.show()
