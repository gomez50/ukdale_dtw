
from numba import jit
import numpy as np

@jit(nopython=True)
def dtw(X, Y):

    
    ROWS = X.shape[0]
    COLUMNS = Y.shape[0]
    
    # Create an empty cost matrix filled with zeroes
    COST_MAT = np.empty((ROWS, COLUMNS))

    # Compute first left bottom cell of the cost matrix
    COST_MAT[0][0] = np.absolute(X[0] - Y[0])

    # Compute first COLUMN and ROW of the cost matrix
    for i in range(1, ROWS):
        COST_MAT[i][0] = COST_MAT[i - 1][0] + np.absolute(X[i] - Y[0])
    for i in range(1, COLUMNS):
        COST_MAT[0][i] = COST_MAT[0][i - 1] + np.absolute(X[0] - Y[i])

    for i in range(1, ROWS):
        for j in range(1, COLUMNS):
            abs_diff = np.absolute(X[i] - Y[j])

            val_1 = COST_MAT[i - 1][j]
            val_2 = COST_MAT[i - 1][j - 1]
            val_3 = COST_MAT[i][j - 1]

            if val_1 <= val_2 and val_1 <= val_3:
                COST_MAT[i][j] = val_1 + abs_diff
            elif val_2 <= val_1 and val_2 <= val_3:
                COST_MAT[i][j] = val_2 + abs_diff
            else:
                COST_MAT[i][j] = val_3 + abs_diff

    return COST_MAT[-1][-1]
