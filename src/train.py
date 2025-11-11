import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

print("Loading Iris dataset...")
# Load the famous Iris flower dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Labels: 0=setosa, 1=versicolor, 2=virginica

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Flower types: {list(iris.target_names)}")

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Create and train the Decision Tree classifier
print("Training Decision Tree model...")
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model training completed!")

# Make predictions on the test set
print("Making predictions on test data...")
y_pred = model.predict(X_test)

# Calculate and display accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%} ({accuracy:.3f})")

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)
print("Created outputs/ directory")

# Generate and save confusion matrix visualization
print("Creating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Iris Flower Classification - Confusion Matrix', fontsize=16, pad=20)
plt.ylabel('Actual Flower Type', fontsize=12)
plt.xlabel('Predicted Flower Type', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("Confusion matrix saved as 'outputs/confusion_matrix.png'")

# Save the trained model for future use
print("Saving trained model...")
joblib.dump(model, 'outputs/model.joblib')
print("Model saved as 'outputs/model.joblib'")

# Display feature importance
feature_importance = model.feature_importances_
feature_names = iris.feature_names

print("\nFeature Importance (which measurements matter most):")
for name, importance in zip(feature_names, feature_importance):
    print(f"   {name}: {importance:.3f}")

print("\n🎉PROJECT COMPLETED SUCCESSFULLY!")
print("Check the 'outputs/' folder for your model and visualization")