import numpy as np

"""
A2-Part-5: Compute the magnitude spectrum (Optional)

Write a function that computes the magnitude spectrum of an input sequence x of length N. The 
function should return an N point magnitude spectrum with frequency index ranging from 0 to N-1.

The input argument to the function is a numpy array x and the function should return a numpy array of the 
magnitude spectrum of x.

EXAMPLE: If you run your function using x = np.array([1, 2, 3, 4]), the function should return the following 
numpy array magX: [array([10.0, 2.82842712, 2.0, 2.82842712])
"""

def genComplexSine(k, N):
    """
    Inputs:
        k (integer) = frequency index of the complex sinusoid of the DFT
        N (integer) = length of complex sinusoid in samples
    Output:
        The function should return a numpy array
        cSine (numpy array) = The generated complex sinusoid (length N)
    """
    n = np.arange(0,N)
    return np.conjugate(np.exp(1j * 2. * np.pi * k * n/N))

def DFT(x):
    """
    Input:
        x (numpy array) = input sequence of length N
    Output:
        The function should return a numpy array of length N
        X (numpy array) = The N point DFT of the input sequence x
    """
    x = x.astype(complex)
    N = len(x)
    k = np.arange(N)
    k = k.reshape(N,1)
    X = np.sum(x*genComplexSine(k,N),axis=1)
    return X

def genMagSpec(x):
    """
    Input:
        x (numpy array) = input sequence of length N
    Output:
        The function should return a numpy array
        magX (numpy array) = The magnitude spectrum of the input sequence x
                             (length N)
    """
    return np.abs(DFT(x))

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4])
    output = genMagSpec(x)
    ground_truth = np.array([10.0, 2.82842712, 2.0, 2.82842712])
    if np.allclose(output,ground_truth):
        print('Test passed')
    else:
        print('my output: {0}'.format(output))
        print('ground truth: {0}'.format(ground_truth))