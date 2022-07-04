import numpy as np
import hilbert

locs = hilbert.decode(np.array([1,2,3]), 2, 3)

print(locs)
