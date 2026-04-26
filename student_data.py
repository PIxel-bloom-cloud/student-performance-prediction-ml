import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[['attendance', 'marks', 'study_hours']]
y = data['result']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=100)

# Train models
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)

# Accuracy
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

# Plot
sns.heatmap(cm, annot=True)
plt.title("Confusion Matrix - Random Forest")
plt.show()