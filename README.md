## Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.  6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AHAMADH SULAIMAN M
RegisterNumber: 212224230009

        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Load dataset
        df = pd.read_csv("student_scores.csv")
        
        # Splitting into features (Hours) and target (Scores)
        X = df[['Hours']]
        Y = df['Scores']
        
        # Splitting the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
        
        # Creating and training the Linear Regression model
        regressor = LinearRegression()
        regressor.fit(X_train, Y_train)
        
        # Predicting test set results
        Y_pred = regressor.predict(X_test)
        
        # Visualizing the Training set
        plt.scatter(X_train, Y_train, color="blue")
        plt.plot(X_train, regressor.predict(X_train), color="black")
        plt.title("Hours vs Scores (Training Set)")
        plt.xlabel("Hours Studied")
        plt.ylabel("Marks Scored")
        plt.show()
        
        # Visualizing the Test set
        plt.scatter(X_test, Y_test, color="yellow")
        plt.plot(X_test, Y_pred, color="black")
        plt.title("Hours vs Scores (Test Set)")
        plt.xlabel("Hours Studied")
        plt.ylabel("Marks Scored")
        plt.show()
        
        # Model Evaluation
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        
        print(f"MSE  = {mse:.2f}")
        print(f"MAE  = {mae:.2f}")
        print(f"RMSE = {rmse:.2f}")

```

## Output:

<img width="836" height="577" alt="image" src="https://github.com/user-attachments/assets/67a0b9c5-44cc-4638-b239-3f09376c762b" />

<img width="842" height="670" alt="image" src="https://github.com/user-attachments/assets/89eb6fc1-8654-4fde-9695-25820161fc1c" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
