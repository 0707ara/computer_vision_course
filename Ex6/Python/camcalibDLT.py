import numpy as np


def camcalibDLT(x_world, x_im):
    """
    :param x_world: World coordinatesm with shape (point_id, coordinates)
    :param x_im: Image coordinates with shape (point_id, coordinates)
    :return P: Camera projection matrix with shape (3,4)
    """

    # Create the matrix A
    A = []
    Z = np.zeros(4)
    ##-your-code-starts-here-##
    for i in range(0,len(x_im)):
        Xi = x_world[i]

        row1 = np.hstack((Z, Xi, -x_im[i,1] * Xi))
        row2 = np.hstack((Xi, Z, -x_im[i,0] * Xi))
        A.append(row1)
        A.append(row2)

    A = np.array(A)

    ##-your-code-ends-here-##

    # Perform homogeneous least squares fitting.
    # The best solution is given by the eigenvector of
    # A.T*A with the smallest eigenvalue.
    ##-your-code-starts-here-##
    E, V = np.linalg.eig(np.dot(A.T, A))
    idxmin = np.argmin(E)
    ev = V[:, idxmin]
    ##-your-code-ends-here-##
    
    # Reshape the eigenvector into a projection matrix P
    P = np.reshape(ev, (3, 4))  # here ev is the eigenvector from above
    #P = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=float)  # remove this and uncomment the line above
    
    return P
