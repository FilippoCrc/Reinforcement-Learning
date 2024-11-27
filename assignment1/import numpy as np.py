import numpy as np

# Example matrix
state = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Extract a specific row
row_index = 1  # Change this to the desired row index
row = state[row_index, :]

# Extract a specific column
col_index = 2  # Change this to the desired column index
col = state[:, col_index]

print("Row:", row)
print("Column:", col)