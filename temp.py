# for i in range(1000):
#     print(i)
import numpy as np
a = np.array([[1,2],[1,2]])
a = np.concatenate((a,a), axis=1)
print(a)