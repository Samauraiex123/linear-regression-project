import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = 'Bike Sharing Demand.csv'
df = pd.read_csv(file_path)
train_data = df[['datetime', 'temp', 'humidity', 'count']]

# Prepare data using pandas
X_df = train_data[['humidity', 'temp']]
X_df.insert(0, 'intercept', 1) # Add column of ones to account for the intercept
Z_df = train_data[['count']]

# Convert pandas DataFrames to NumPy arrays for calculations
X = X_df.values # X is the feature matrix (includes intercept, humidity, and temp)
Z = Z_df.values # Z is the target variable (count)

theta = np.array([[1], [1], [1]])
# This is what theta looks like right now
#[[1], -> intercept
# [1], -> humidity coefficient
# [1]] -> temp coefficient


#CURRENT VALUES TESTED at [[1],[1],[1]] -> Starting value of cost: 23041.98657808194
#0.00001 -> Value of parameters at convergence: [1.16901991 0.62637615 6.60536712], Final value of the cost: 14939.80240926716
#0.00005 -> Value of parameters at convergence: [1.13119328 0.97945845 5.44019571], Final value of the cost: 15679.67629591595
#0.0001 -> Value of parameters at convergence: [1.21993855 0.20671463 7.98995719], Final value of the cost: 14242.117449911146
#0.0004 -> Value of parameters at convergence: [ 1.57060929 -0.99830747 11.95564545], Final value of the cost: 13333.516723892735
#0.0005 -> Value of parameters at convergence: [-1.84158620e+09 -1.23915753e+11 -3.74904480e+10], Final value of the cost: 3.836483438684461e+25

#CURRENT VALUES TESTED at [[0.5],[0.5],[0.5]] -> Starting value of cost: 27983.72337656164
#0,00001 -> Value of parameters at convergence: [0.55242395 1.99623133 2.04740155], Final value of the cost: 18638.733580379518
#0.0001 -> Value of parameters at convergence: [0.73235807 0.24955988 7.8722931], Final value of the cost: 14302.411215015945
#0.0004 -> Value of parameters at convergence: [ 1.08570334 -0.99173342 11.95773574], Final value of the cost: 13338.330639082656
#0.0005 -> Value of parameters at convergence: [-2.75725640e+09 -1.85528922e+11 -5.61313816e+10], Final value of the cost: 8.600100613725901e+25

#Observation #1: Somewhere between learning rate of 0.0004 -> 0.0005, it seems that the function diverges.
#Observation #2: Decreasing the parameters seems to increase the cost



alpha = 0.0005
m = len(Z)
iterations = 100

def compute_cost(X, Z, theta):
    predictions = X.dot(theta)
    errors = predictions - Z
    return np.sum(errors**2) / (2 * m)

def gradient_descent(X, Z, theta, alpha, iterations):
    cost_history = []
    theta_history = []

    # theta.flatten() converts parameter vector into 1D array
    print("Starting values of parameters:", theta.flatten())
    print("Starting value of learning rate (alpha):", alpha)
    print("Starting value of cost:", compute_cost(X, Z, theta))

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - Z
        gradient = X.T.dot(errors) / m #get direction of steepest increase for cost function
        #Borrowed from https://medium.com/@ilyasaoudata/you-should-understand-backpropagation-in-regression-before-diving-into-neural-networks-5e08d48d69e6)
        theta = theta - alpha * gradient

        cost = compute_cost(X, Z, theta)
        cost_history.append(cost)
        theta_history.append(theta.flatten())

        if i == 0 or i == 24 or i == iterations - 1:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            #It takes too much ram to graph more so just a random sample size of 100
            sample_idx = np.random.choice(len(X), size=100, replace=False)
            ax.scatter(X[sample_idx, 1], X[sample_idx, 2], Z[sample_idx], label='Actual', alpha=0.5)
            ax.scatter(X[sample_idx, 1], X[sample_idx, 2], predictions[sample_idx], color='red', alpha=0.5, label='Predicted', s = 50)



            ax.set_xlabel('Humidity')
            ax.set_ylabel('Temperature')
            ax.set_zlabel('Count')
            ax.set_title(f'Iteration {i+1}')
            ax.legend()

            plt.show()
            plt.close(fig)


    print("\nValue of parameters at convergence:", theta.flatten())
    print("Final value of the cost:", compute_cost(X, Z, theta))

    return theta, cost_history, theta_history

theta, cost_history, theta_history = gradient_descent(X, Z, theta, alpha, iterations)

plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()