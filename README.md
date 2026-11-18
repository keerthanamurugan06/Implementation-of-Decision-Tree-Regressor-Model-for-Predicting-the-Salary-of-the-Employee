# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required Python libraries and load the Salary.csv dataset.



2. Separate the input feature (Level) and target variable (Salary).



3. Split the dataset into training and testing data, and create the Decision Tree Regressor model.



4. Train the model, predict employee salaries, calculate the R² score and Mean Absolute Error, and visualize the Decision Tree.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: KEERTHANA M
RegisterNumber:  212225040185
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error

# Load the dataset
data = pd.read_csv("Salary.csv")

# Separate input and output features
X = data[['Level']]
y = data['Salary']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict salary
y_pred = model.predict(X_test)

# Display actual and predicted salary
print("Actual Salary\tPredicted Salary")

for actual, predicted in zip(y_test, y_pred):
    print(actual, "\t\t", predicted)

# Evaluate the model
print("\nR2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Visualize Decision Tree
plt.figure(figsize=(18, 10))

plot_tree(
    model,
    feature_names=['Level'],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree Regressor for Employee Salary Prediction")
plt.savefig("sam.png", dpi=200, bbox_inches='tight')
plt.show()

```

## Output:

<img width="610" height="161" alt="image" src="https://github.com/user-attachments/assets/b0403f49-b372-48a8-ab3a-46d5ec762cee" />
<img width="2830" height="1621" alt="image" src="https://github.com/user-attachments/assets/8a92791f-a4ff-4301-b380-398d2f61e056" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
