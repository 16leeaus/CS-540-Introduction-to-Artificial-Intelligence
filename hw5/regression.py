import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.

def get_dataset(filename):

    dataset = genfromtxt(filename, delimiter=',', skip_header = 1, usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16))
    return dataset

dataset = get_dataset('bodyfat.csv')

def print_stats(dataset, col):
    col = col
    sum = 0
    distSum = 0
    StdDev = 0

    # Get length of dataset, this is the num of pts.
    numDataPts = len(dataset)
    print(numDataPts)

    # Get mean of a certain column here:
    for row in dataset:
        sum = sum + row[col]
    mean = sum / numDataPts
    print('{:.2f}'.format(mean))

    # Calculate Std. Dev of column here:
    for row in dataset:
       distSqr = np.square(row[col] - mean)
       distSum = distSum + distSqr
    
    StdDev = np.sqrt(distSum/numDataPts)
    print('{:.2f}'.format(StdDev))

    pass

def regression(dataset, cols, betas):

    # Initalize empty variables:
    predictedy = 0
    totalError = 0
    
    # Loop through dataset and access elements at specified columns:
    for i in range(len(dataset)):
        for j in range(len(cols)):
            # Calculate a predicted y value:
            predictedy += (dataset[i][cols[j]] * betas[j+1])
        predictedy += betas[0]

        # Calculate the error, for the actual value:
        error = ((dataset[i][0] - predictedy) ** 2)
        # Get the total error, to find the average:
        totalError = totalError + error

        # Reset the predicted value:
        predictedy = 0 

    # Find the average squared error, then return:
    mse = totalError/len(dataset)
    return mse

def gradient_descent(dataset, cols, betas):

    # Get the length of the dataset:
    n = len(dataset)

    # Get the first column of the dataset, for the y vals:
    y = dataset[:,0]

    # Iterate through the dataset and get the necessary columns:
    X = dataset[:,cols]

    # Append a column of ones to the necessary columns:
    X = np.hstack((np.ones((n, 1)),X))

    # Find gradients then return as output of function:
    grads = 2*np.sum((X @ betas - y).reshape((-1,1)) * X, axis = 0) / n
    return np.array(grads)

def iterate_gradient(dataset, cols, betas, T, eta):

    # Loop through the update times:
    for i in range(0, T):
        # Call gradient descent to get the partial derivs:
        grads = gradient_descent(dataset, cols, betas)

        # Loop through betas:
        for j in range(len(betas)):
            betas[j] -= eta * grads[j]
            
        # Call regression to get the MSE:
        mse = regression(dataset, cols, betas)

        # Gather final result:
        result = ["%.2f" % elem for elem in betas]
        print('{:d} {:.2f}'.format(i + 1, mse), *result)
    pass

def compute_betas(dataset, cols):

    # Gather y values and desired cols from dataset, then append a column of ones:
    y = dataset[:, 0]
    x = dataset[:, cols]
    ones = [[1]] * len(dataset)
    x = np.concatenate((ones, x), 1)

    # Preform the required matrix operations:
    x_transpose = np.transpose(x)
    x_dot = np.dot(x_transpose, x)
    x_inverse = np.linalg.inv(x_dot)
    x = np.dot(x_inverse, x_transpose)

    # Get the dot product of the final matrix with the y values:
    betas = np.dot(x, y)
    # Use the calculated beta values to find the MSE and return:
    mse = regression(dataset, cols, betas)
    return (mse, *betas)

def predict(dataset, cols, features):

    # Call compute betas and gather the results:
    betas = compute_betas(dataset, cols)

    # Declare an array for the beta values:
    arr_betas = []

    # Loop through the temp beta values, and append them to the array:
    for i in range(2, len(betas)):
        arr_betas.append(betas[i])

    # Calculate the predicted body fat value and return:
    result = np.dot(features, arr_betas) + betas[1]
    return result

def synthetic_datasets(betas, alphas, X, sigma):
    # Generate a z_i value using sigma and a mean of 0:
    z = np.random.normal(0, sigma)
    
    # Create an array for the y values:
    y_beta = []
    y_alpha = []

    # Generate y values for the linear set using betas and z_i:
    for i in range(len(X)):
        y_linear = betas[0] + (betas[1] * X[i]) + z
        y_beta.append(y_linear)

    # Concatenate the linear dataset, with X:
    linear_dataset = np.array(y_beta)
    linear_dataset = np.concatenate((linear_dataset.reshape(-1, 1), X), 1)

    # Generate y values for the quadratic set using alphas and z_i:
    for j in range(len(X)):
        y_quadratic = alphas[0] + (alphas[1] * (X[j] ** 2) + z)
        y_alpha.append(y_quadratic)
   
    # Concatenate the quadratic dataset, with X:
    quadratic_dataset = np.array(y_alpha)
    quadratic_dataset = np.concatenate((quadratic_dataset.reshape(-1, 1), X), 1)

    # Return the generated datasets:
    return linear_dataset, quadratic_dataset

def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')

    # Import the random util. from numpy:
    from numpy.random import randint

    # Create arrays to hold our arrays of alpha and beta values:
    betas_arr = []
    alphas_arr = []
    cols = [2, 3]

    # Randomly generate 1000 datapoints, inbetween -100 and 100:
    values = randint(-100, 100, 1000)
    # Create a sigma array:
    sigmas = [10e-4, 10e-3, 10e-2, 10e-1, 10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5]
    # Loop through the sigma array and create a pair of alpha and beta values for each:
    for i in range(len(sigmas)):

        # Generate beta values and insert into array
        betas = np.array(randint(1, 10, 2))
        betas_arr.append(betas)

        # Generate alpha values and insert into array
        alphas = np.array(randint(1, 10, 2))
        alphas_arr.append(alphas)

        # For each setting of sigma generate two datasets:
        linear_dataset, quadratic_dataset = synthetic_datasets(betas_arr[i], alphas_arr[i], values, sigmas[i])
    
    # Calculate the betas for the dataset:
    linear_betas = compute_betas(linear_dataset, cols)
    quadratic_betas = compute_betas(quadratic_dataset, cols)

    # Find the MSE of the dataset with the appropriate beta values:
    linear_regression = regression(linear_dataset, cols, betas)
    quadratic_regression = regression(quadratic_dataset, cols, alphas)

if __name__ == '__main__':
    ### DO NOT CHANGE THIS SECTION ###
    plot_mse()
