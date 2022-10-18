import numpy as np


def estimateF(x1, x2):
    """
    :param x1: Points from image 1, with shape (coordinates, point_id)
    :param x2: Points from image 2, with shape (coordinates, point_id)
    :return F: Estimated fundamental matrix
    """
    I = np.zeros((11, 9))

    # Use x1 and x2 to construct the equation for homogeneous linear system
    ##-your-code-starts-here-##
    for i in range(11):
        u = x1[0, i]
        v = x1[1, i]
        u2 = x2[0, i]
        v2 = x2[1, i]
        I[i] = [u2*u, u2*v, u2, v2*u, v2*v, v2, u, v, 1]


    ##-your-code-ends-here-##
    U, S, V = np.linalg.svd(I)
    v_min = V[np.argmin(S), :]
    # Use SVD to find the solution for this homogeneous linear system by
    # extracting the row from V corresponding to the smallest singular value.
    ##-your-code-starts-here-##

    ##-your-code-ends-here-##
    F = np.reshape(v_min, (3, 3))  # reshape to acquire Fundamental matrix F
    #F = np.ones((3, 3))  # remove me and uncomment the above

    # Enforce constraint that fundamental matrix has rank 2 by performing
    # SVD and then reconstructing with only the two largest singular values
    # Reconstruction is done with u @ s @ vh where s is the singular values
    # in a diagonal form.
    ##-your-code-starts-here-##
    U2, S2, V2 = np.linalg.svd(F)
    S2[2] = 0
    F = U2 @ np.diag(S2) @ V2

    ##-your-code-ends-here-##
    
    return F
