import numpy as np

# Create two different NumPy arrays of size 3x15 and 3x20
array1 = np.random.rand(3, 15)
array2 = np.random.rand(3, 20)

# Create a 1x2 object array containing the two arrays
object_array = np.array([array1, array2], dtype=object)

# Verify the contents
print(object_array)
print(object_array[0])
print(object_array[1])