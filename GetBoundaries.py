import sys
import numpy as np
import matplotlib.pyplot as plt


def check_left(matrix, i, j):
    if j == 0:
        return False
    elif matrix[i][j - 1] == 1:
        return True
    return False


def check_right(matrix, i, j):
    cols = matrix.shape[1]
    if j == cols - 1:
        return False
    elif matrix[i][j + 1] == 1:
        return True
    return False


def check_up(matrix, i, j):
    if i == 0:
        return False
    elif matrix[i - 1][j] == 1:
        return True
    return False


def check_down(matrix, i, j):
    rows = matrix.shape[0]
    if i == rows - 1:
        return False
    elif matrix[i + 1][j] == 1:
        return True
    return False


def is_inside_boundary(matrix, i, j):
    if matrix[i][j] == 1:
        if check_left(matrix, i, j) and check_up(matrix, i, j) and check_right(matrix, i, j) and check_down(matrix, i, j):
            return True
    return False


file_name = sys.argv[1]
b_matrix = np.loadtxt(file_name, dtype=int)

plt.imshow(b_matrix, cmap="Greys")
plt.title("Input Matrix")
plt.colorbar()
plt.show()

rows, cols = b_matrix.shape
boundary_matrix = np.ones((rows, cols), dtype=int)

for i in range(rows):
    for j in range(cols):
        if is_inside_boundary(b_matrix, i, j) or b_matrix[i][j] == 0:
            boundary_matrix[i][j] = 0

np.savetxt("boundaries_output.txt", boundary_matrix, delimiter=" ", fmt="%d")
plt.imshow(boundary_matrix, cmap="Greys")
plt.title("Output matrix")
plt.colorbar()
plt.show()