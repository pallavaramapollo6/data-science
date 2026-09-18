from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

model = DecisionTreeClassifier(random_state=42)
iris = load_iris()
X = iris.data
y = iris.target
# cv=5 means 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(scores)
print(scores.mean())
