import numpy as np

l0 = [0.33899, 0.34216, 0.34842, 0.34681, 0.344]
l1 = [0.69182, 0.70813, 0.71775, 0.69703, 0.69869]
l2 = [0.58697, 0.56632, 0.585,  0.58481,  0.57642] 


n0 = np.array(l0)
n1 = np.array(l1)
n2 = np.array(l2)

print(n0.mean(), n1.mean(), n2.mean())
print(n0.std(), n1.std(), n2.std())
