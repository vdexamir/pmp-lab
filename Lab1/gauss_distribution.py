from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

# Generate regularly spaced observations in the range (-5, 5)
x_data = np.arange(-5, 5, 0.001)

# Compute the probability density function
y_data = stats.norm.pdf(x_data, 0, 1)

# Generate the plot
plt.plot(x_data, y_data)

# Display the plot
plt.show()

# Test for Gaussian Distribution
np.random.seed(1)

# generating uni-variate data
data = 10 * np.random.randn(1000) + 100

plt.hist(data)
plt.show()

plt.hist(data, bins=100)
plt.show()
