import numpy as np
array1 = np.array([1, 2, 3, 4, 5])

array2 = np.array([6,7,8,9,10])

indices1 = np.array([0, 2, 4,6,8])
all_len = len(array1) + len(array2)
mask = ~np.isin(np.arange(all_len), indices1)
result = np.empty(all_len)

result[indices1] = array1
result[mask] = array2

print(result)