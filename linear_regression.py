import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

"""
Create dataset
Our target is exam_score.
"""

data = {
    "study_hours": [2, 3, 4, 5, 6, 7, 8, 1, 4, 6],
    "attendance": [65, 70, 75, 80, 85, 90, 92, 60, 82, 88],
    "previous_score": [55, 60, 65, 70, 75, 80, 85, 50, 68, 78],
    "assignments": [60, 65, 70, 75, 80, 85, 90, 55, 78, 82],
    "exam_score": [58, 63, 68, 74, 81, 87, 92, 52, 72, 84]
}

df = pd.DataFrame(data)

"""
Features (X)

Study hours
Attendance
Previous score
Assignment completion

Target (y)

Exam score

Study habits + attendance + previous performance → Exam score
"""
X = df[
    [
        "study_hours",
        "attendance",
        "previous_score",
        "assignments"
    ]
]

y = df["exam_score"]


"""
We don't want to train and test the model using exactly the same data.
80% → Training
20% → Testing

The training data teaches the model.

The testing data checks whether the model can make predictions on data it hasn't seen before.
"""

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


"""
Create and train model using Linear regression model
"""

model = LinearRegression()

"""
The model will learn relationships between using model.fit()

Study hours
      ↓
Attendance
      ↓
Previous score
      ↓
Assignments
      ↓
Exam score

"""
model.fit(X_train, y_train)


"""
Test the model to predict with 20% of test data
"""

y_pred = model.predict(X_test)

print("Actual scores:   ", y_test.values)
print("Predicted scores:", y_pred)


"""
Evaluate the model performance with Mean Absolute Error (MAE) and 
"""

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("MAE:", mae)
print("R²:", r2)

"""
Predict a new student
"""

new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [85],
    "previous_score": [75],
    "assignments": [80]
})

predicted_score = model.predict(new_student)

print("\nNew Student")
print("Predicted Exam Score:",
      predicted_score[0])
