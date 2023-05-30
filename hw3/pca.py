from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    dataset = np.load(filename)
    dataset = dataset - np.mean(dataset, axis = 0)
    return dataset

def get_covariance(dataset):
    S = (1 / (len(dataset) - 1)) * np.dot(np.transpose(dataset), dataset)
    m = len(S)
    return S

def get_eig(S, m):
    Lambda, U = eigh(S, eigvals = (len(S) - m, len(S) - 1))

    Lambda = np.diag(Lambda[::-1])
    U = np.flip(U, 1)

    return Lambda, U

def get_eig_perc(S, perc):
    arrayEigenVal = []
    arrayEigenVect = []

    eigneVal, eigenVect = eigh(S)
    sumEigneVal = np.sum(eigneVal)

    sumEigs = sum(eigneVal)
    limit = sumEigs * perc
    eigneVal, eigenVect = eigh(S, subset_by_value=(limit, np.inf))

    eigneVal = np.diag(eigneVal[::-1])
    eigenVect = np.flip(eigenVect, 1)

    return eigneVal, eigenVect

def project_image(img, U):
    proj = np.dot(U, np.dot(img, U))
    return proj

def display_image(orig, proj):

    orig = np.array([])
    proj = np.array([])

    np.reshape(orig,(32,32))
    np.reshape(proj,(32,32))

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1[0].set_title('Original')
    ax2[1].set_title('Projection')

    fig.colorbar(imshow(np.transpose(orig), aspect='equal'), ax=ax1)

    plt.show()