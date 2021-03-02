import numpy as np

# List
# ----

# Save list to CSV file
x = np.arange(0.0, 5.0, 1.0)
np.savetxt('test.csv', x, delimiter=',', fmt='%s')
