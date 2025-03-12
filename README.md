# Alternative-Fitting-Methods-Comparison
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RANSACRegressor, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy import stats
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data with outliers
n_samples = 80
x = np.linspace(0, 10, n_samples)
# True relationship
y_true = 2 * x + 5
# Normal noise
noise = np.random.normal(0, 2, size=n_samples)
# Add some outliers
outlier_indices = np.random.choice(range(n_samples), size=5, replace=False)
noise[outlier_indices] = noise[outlier_indices] * 5
# Final noisy data
y = y_true + noise

# Reshape x for sklearn
X = x.reshape(-1, 1)

# Initialize different regression models
models = {
    'Ordinary Least Squares': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'LASSO Regression': Lasso(alpha=0.1),
    'RANSAC Regression': RANSACRegressor(random_state=42),
    'Huber Regression': HuberRegressor(epsilon=1.35),
    'Polynomial Regression': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
}

# Weighted Least Squares implementation
def weighted_least_squares(X, y, weights):
    # Calculate weighted means
    x_mean = np.average(X, weights=weights)
    y_mean = np.average(y, weights=weights)
    
    # Calculate weighted covariance and variance
    numerator = np.sum(weights * (X - x_mean) * (y - y_mean))
    denominator = np.sum(weights * (X - x_mean)**2)
    
    # Calculate slope and intercept
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept

# LAD (Least Absolute Deviations) implementation
def least_absolute_deviations(X, y):
    # Find the slope and intercept that minimize sum of absolute deviations
    def objective(params):
        slope, intercept = params
        return np.sum(np.abs(y - (slope * X.flatten() + intercept)))
    
    # Initial guess (using OLS)
    ols = LinearRegression().fit(X, y)
    initial_guess = [ols.coef_[0], ols.intercept_]
    
    # Optimize using scipy
    from scipy.optimize import minimize
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    
    return result.x[0], result.x[1]

# Train the models and predict
predictions = {}
for name, model in models.items():
    model.fit(X, y)
    predictions[name] = model.predict(X)

# For Weighted Least Squares
# Calculate weights inversely proportional to the square of x values (example weighting)
weights = 1 / (1 + x**2)
wls_slope, wls_intercept = weighted_least_squares(x, y, weights)
predictions['Weighted Least Squares'] = wls_slope * x + wls_intercept

# For LAD
lad_slope, lad_intercept = least_absolute_deviations(X, y)
predictions['Least Absolute Deviations'] = lad_slope * x + lad_intercept

# Calculate MSE for each model
mse = {}
for name, y_pred in predictions.items():
    mse[name] = mean_squared_error(y, y_pred)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Comparison of Different Fitting Methods', fontsize=16)

# Sort methods by MSE
sorted_methods = sorted(mse.items(), key=lambda x: x[1])

# Plot each method
for i, (name, _) in enumerate(sorted_methods):
    plt.subplot(3, 3, i+1)
    
    # Highlighting outliers with different color
    plt.scatter(x, y, color='blue', alpha=0.5, label='Regular data points')
    plt.scatter(x[outlier_indices], y[outlier_indices], color='red', label='Outliers')
    
    # Plot the true relationship
    plt.plot(x, y_true, 'g--', label='True relationship: y = 2x + 5')
    
    # Plot the predicted line
    plt.plot(x, predictions[name], 'r-', label=f'{name}\nMSE: {mse[name]:.2f}')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(name)
    plt.legend()
    plt.grid(True)

# Add a summary plot showing all regression lines
plt.subplot(3, 3, 9)
plt.scatter(x, y, color='blue', alpha=0.3, label='Data')
plt.scatter(x[outlier_indices], y[outlier_indices], color='red', label='Outliers')
plt.plot(x, y_true, 'g--', linewidth=2, label='True relationship')

# Plot all prediction lines
for name, y_pred in predictions.items():
    plt.plot(x, y_pred, label=f'{name}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('All Methods Comparison')
plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.grid(True)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('alternative_fitting_methods.png')
plt.show()

# Print summary of results
print("Performance Comparison (Mean Squared Error):")
print("-" * 50)
for name, error in sorted_methods:
    print(f"{name:30s}: {error:.4f}")

# Show how each method handles the outliers specifically
fig, ax = plt.figure(figsize=(12, 8)), plt.axes()
for i, (name, y_pred) in enumerate(predictions.items()):
    outlier_errors = np.abs(y[outlier_indices] - y_pred[outlier_indices])
    normal_errors = np.abs(np.delete(y, outlier_indices) - np.delete(y_pred, outlier_indices))
    
    ax.bar(i-0.2, np.mean(outlier_errors), width=0.4, color='red', label='Outlier Error' if i==0 else "")
    ax.bar(i+0.2, np.mean(normal_errors), width=0.4, color='blue', label='Normal Data Error' if i==0 else "")

ax.set_xticks(range(len(predictions)))
ax.set_xticklabels(predictions.keys(), rotation=45, ha='right')
ax.set_ylabel('Mean Absolute Error')
ax.set_title('Error Comparison: Normal Data vs. Outliers')
ax.legend()
plt.tight_layout()
plt.savefig('error_comparison.png')
plt.show()
