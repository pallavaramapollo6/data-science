from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

"""
Load the Iris dataset
The dataset contains measurements of iris flowers:

Sepal length
Sepal width
Petal length
Petal width
"""
iris = load_iris()

"""
Separate features and target
X contains features of flower, the four measurements:
Sepal length
Sepal width
Petal length
Petal width

y contains the class of flower:
0 → setosa
1 → versicolor
2 → virginica
"""
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

"""
Create and train the Decision Tree
"""
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# --------------------------------
# Give NEW data for prediction
# --------------------------------

new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("Predicted class:", prediction[0])
print("Predicted flower:", iris.target_names[prediction[0]])
