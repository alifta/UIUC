import numpy as np

import matplotlib.pyplot as plt
from numpy.core.fromnumeric import trace

# List
# ----

# Save list to CSV file
x = np.arange(0.0, 5.0, 1.0)
np.savetxt('test.csv', x, delimiter=',', fmt='%s')

# Figure
# ------
plt.tight_layout()
plt.savefig(fname='figure.pdf', dpi=300, transparent=True,bbox_inches='tight')

