import numpy as np


def my_transpose(output):
    if output.ndim == 1:
        # If it's a 1D vector, reshape it to (1, dim_output) to make it 2D and then transpose
        output = output.reshape(1, -1).T
    elif output.ndim == 2:
        # If it's already 2D, simply transpose it
        output = output.T
    else:
        raise ValueError("Output should be a 1D or 2D array.")

    return output