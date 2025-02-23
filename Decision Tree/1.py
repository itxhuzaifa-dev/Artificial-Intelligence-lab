# Import necessary libraries
import pandas as pd

# Define the column names manually
column_names = ['Alternate', 'Bar', 'Fri/Sat', 'Hungry', 'Patrons', 'Price','Raining', 'Reservation', 'Type', 'WaitEstimate', 'Wait']

# Load the dataset without a header
data = pd.read_csv('restaurant.csv', header=None, names=column_names)

# Print the first few rows of the dataset to ensure it's loaded correctly
print("First few rows of the dataset:")
print(data.head())
# Split the data into features (X) and target (y)
X = data.drop('Wait', axis=1)
y = data['Wait']

# Convert categorical variables to numerical using get_dummies
X = pd.get_dummies(X)

from sklearn.tree import DecisionTreeClassifier

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# Predict the test set results
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Generate a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Plot the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=True, filled=True)
plt.show()
# Get the feature importances
feature_importances = clf.feature_importances_

# Create a DataFrame for better visualization of feature importances
features_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance
features_df = features_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.gca().invert_yaxis()  # To display the most important feature at the top
plt.show()