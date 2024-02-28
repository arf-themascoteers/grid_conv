import numpy as np
from scipy.stats import pearsonr

# Example data
x = np.array([1, 2, 3, 4, 5])
y = np.array([10, 8, 12, 9, 11])

# Calculate Pearson correlation coefficient and p-value
correlation_coefficient, p_value = pearsonr(x, y)

# Display the results
print(f"Pearson Correlation Coefficient: {correlation_coefficient}")
print(f"P-value: {p_value}")
