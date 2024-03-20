import numpy as np

a = np.array([
	[0.001,	0,	0,	0.45],
	[0,	0.001,	0,	0],
	[0,	0,	0.001,	0],
	[0,	0,	0,	0.001]
])

print(np.linalg.norm(a))