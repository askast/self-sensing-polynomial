import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
np.set_printoptions(threshold=sys.maxsize)

from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

filename = 'data3.csv'
reference_filename = 'reference_data.csv'
eta_mod_coefficient = 0.1

power_units = 'KW'  # {HP, kW, W}
match power_units:
    case 'HP':
        power_conversion_factor = 1  # 1 HP = 0.7457 kW
    case 'kW':
        power_conversion_factor = 1.34102  # 1 kW = 1.34102 HP
    case 'W':
        power_conversion_factor = 0.00134102  # 1 W = 0.00134102 HP
    case _:
        power_conversion_factor = 1  # Default to 1 if unknown unit

try:
    # Attempt to read as Excel file as per prompt
    data = pd.read_excel(filename)
except ValueError as e:
    # If 'filename' was 'data.csv' and it's a real CSV, pd.read_excel might fail.
    # Offer to try reading as CSV as a common fallback.
    if filename.lower().endswith('.csv'):
        print(f"Reading as Excel failed for {filename}: {e}. Trying to read as CSV.")
        data = pd.read_csv(filename)
    else:
        raise # Re-raise original error if not a .csv or if CSV read also fails

try:
    reference_data = pd.read_excel(reference_filename)
except ValueError as ref_e:
    if reference_filename.lower().endswith('.csv'):
        print(f"Reading as Excel failed for {reference_filename}: {ref_e}. Trying to read as CSV.")
        reference_data = pd.read_csv(reference_filename)
    else:
        raise
def create_off_speed_test_data(hz_data, flow_data, head_data, power_data):
    """
    Create a DataFrame with the given data.
    This function is a placeholder and can be modified to generate test data.
    """
    target_hz = [65, 60, 55, 50, 45, 40, 35, 33, 30]
    extrapolated_flow = []
    extrapolated_power = []
    extrapolated_head = []
    extrapolated_hz = []
    eta_data = flow_data * head_data / (3960 * power_data * power_conversion_factor)
    bep_eta = eta_data.max()  # Calculate the max efficiency
    for hz in target_hz:
        for hz_val, flow_val, head_val, power_val in zip(hz_data, flow_data, head_data, power_data):
            eta_ratio = (bep_eta**-1)-(((bep_eta**-1)-1)*((hz_val/hz)**eta_mod_coefficient))
            extrapolated_flow.append(flow_val * hz/hz_val)
            extrapolated_head.append(head_val * (hz/hz_val)**2)
            extrapolated_power.append((power_val * (hz/hz_val)**3)/ eta_ratio)
            extrapolated_hz.append(hz)

    return pd.DataFrame({
        'extrapolated_hz': extrapolated_hz,
        'extrapolated_power': extrapolated_power,
        'extrapolated_flow': extrapolated_flow,
        'extrapolated_head': extrapolated_head
    })

# Extract data columns. Ensure your file has 'x', 'y', 'z' columns.
hz_data = data['hz'].values
power_data = data['power'].values
flow_data = data['flow'].values
head_data = data.get('head', None)  # Optional, if 'head' column exists

extrapolated_data = create_off_speed_test_data(hz_data, flow_data, head_data, power_data)
# save extrapolated_data to a CSV file
extrapolated_data.to_csv("extrapolated_data.csv", index=False)

# Create the design matrix for a cubic polynomial z = f(x,y)
# The terms are: 1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3
A = np.vstack([
    np.ones_like(extrapolated_data['extrapolated_hz']),  # Constant term
    extrapolated_data['extrapolated_hz'],                # x
    extrapolated_data['extrapolated_power'],                # y
    extrapolated_data['extrapolated_hz']**2,             # x^2
    extrapolated_data['extrapolated_hz'] * extrapolated_data['extrapolated_power'],       # xy
    extrapolated_data['extrapolated_power']**2,             # y^2
    extrapolated_data['extrapolated_hz']**3,             # x^3
    extrapolated_data['extrapolated_hz']**2 * extrapolated_data['extrapolated_power'],    # x^2y
    extrapolated_data['extrapolated_hz'] * extrapolated_data['extrapolated_power']**2,# xy^2
    extrapolated_data['extrapolated_power']**3,             # y^3
    extrapolated_data['extrapolated_hz']**4,                # x^4
    extrapolated_data['extrapolated_hz']**3 * extrapolated_data['extrapolated_power'],         # x^3y
    extrapolated_data['extrapolated_hz']**2 * extrapolated_data['extrapolated_power']**2,      # x^2y^2
    extrapolated_data['extrapolated_hz'] * extrapolated_data['extrapolated_power']**3,         # xy^3
    extrapolated_data['extrapolated_power']**4              # y^4
]).T

# Perform least squares regression to find the polynomial coefficients
# np.linalg.lstsq returns a tuple; the first element contains the coefficients
coeffs, residuals, rank, s = np.linalg.lstsq(A, extrapolated_data['extrapolated_flow'], rcond=None)
# Fit a radial basis function interpolator to the data
rbf = Rbf(
    extrapolated_data['extrapolated_hz'],
    extrapolated_data['extrapolated_power'],
    extrapolated_data['extrapolated_flow'],
    function='thin_plate',  # You can choose other functions like 'linear', 'cubic', etc.
)

# Define the polynomial function using the calculated coefficients for predictions
def cubic_poly_surface(x, y, c):
    return (c[0] +
            c[1] * x + c[2] * y +
            c[3] * x**2 + c[4] * x * y + c[5] * y**2 +
            c[6] * x**3 + c[7] * x**2 * y + c[8] * x * y**2 + c[9] * y**3 
            + c[10] * x**4 + c[11] * x**3 * y + c[12] * x**2 * y**2 + c[13] * x * y**3 + c[14] * y**4
            )

#determining error in the fit
flow_estimated = cubic_poly_surface(reference_data['hz'], reference_data['power'], coeffs)
error = (flow_estimated - reference_data['flow'])/ reference_data['flow']
print(error*100)  # Print error as a percentage
for ref_data_point, flow_est_point, error in zip(reference_data.itertuples(), flow_estimated, error):
    print(f"Data Point: {ref_data_point.hz}, {ref_data_point.power}, {ref_data_point.flow}, {flow_est_point}| Error: {error*100:.2f}%")
# print the polynomial equation
print(f"""Polynomial Coefficients: 
        {coeffs[0]} +
        {coeffs[1]} * x + {coeffs[2]} * y +
        {coeffs[3]} * x**2 + {coeffs[4]} * x * y + {coeffs[5]} * y**2 +
        {coeffs[6]} * x**3 + {coeffs[7]} * x**2 * y + {coeffs[8]} * x * y**2 + {coeffs[9]} * y**3 
""")
poly_string= f"{coeffs[0]}+{coeffs[1]}*$B$98+{coeffs[2]}*C100+{coeffs[3]}*$B$98^2+{coeffs[4]}*$B$98*C100+{coeffs[5]}*C100^2+{coeffs[6]}*$B$98^3+{coeffs[7]}*$B$98^2*C100+{coeffs[8]}*$B$98*C100^2+{coeffs[9]}*C100^3+{coeffs[10]}*$B$98^4+{coeffs[11]}*$B$98^3*C100+{coeffs[12]}*$B$98^2*C100^2+{coeffs[13]}*$B$98*C100^3+{coeffs[14]}*C100^4"
poly_string = poly_string.replace('+-', '-')
print(f"Polynomial Coefficients:\n{poly_string}")
poly_string = poly_string.replace('$B$98', 'hz').replace('C100', 'power')
print(f"Polynomial Coefficients:\n{poly_string}")

# Calculate the error (difference) between estimated and actual flow values
flow_error = reference_data['flow'] - flow_estimated

# Create a grid of points to plot the fitted surface
# Generate a range of x and y values based on the data extents
x_surf_range = np.linspace(extrapolated_data['extrapolated_hz'].min(), extrapolated_data['extrapolated_hz'].max(), 300)  # 30 points for x-axis
y_surf_range = np.linspace(extrapolated_data['extrapolated_power'].min(), extrapolated_data['extrapolated_power'].max(), 300)  # 30 points for y-axis
X_surf, Y_surf = np.meshgrid(x_surf_range, y_surf_range)

# Calculate the Z values for the surface using the polynomial function
Z_surf = cubic_poly_surface(X_surf, Y_surf, coeffs)
# Z_surf = rbf(X_surf, Y_surf)  # Use the RBF interpolator to get Z values

# Matplotlib visualization
fig = plt.figure(figsize=(12, 8))  # Create a figure for the plot
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

# Scatter plot of the original data points
ax.scatter(extrapolated_data['extrapolated_hz'], extrapolated_data['extrapolated_power'], extrapolated_data['extrapolated_flow'], color='red', marker='o', s=50, label='Data Points')
# ax.scatter(reference_data['hz'], reference_data['power'], reference_data['flow'], color='red', marker='o', s=50, label='Data Points')
# max_flow = reference_data['flow'].max()*1.1
max_flow = extrapolated_data['extrapolated_flow'].max() * 1.1  # Set the maximum flow value for masking

# Create a mask to filter out Z values that are greater than the maximum flow value and less than or equal to the maximum flow value
mask = (Z_surf <= max_flow) & (Z_surf > 0)  # Mask for Z values within the range of interest

# Apply the mask
Z_masked = np.where(mask, Z_surf, np.nan)

# Plot the fitted 3D surface
# 'cmap' sets the color map (e.g., 'viridis', 'plasma', 'coolwarm')
# 'alpha' sets the transparency of the surface
surf = ax.plot_surface(X_surf, Y_surf, Z_masked, cmap='viridis', alpha=0.7, edgecolor='k', linewidth=0.2)

# Set labels for the axes and a title for the plot
ax.set_xlabel('Hz (Frequency)')
ax.set_ylabel('Power (HP)')
ax.set_zlabel('Flow (GPM)')
ax.set_title('3D Cubic Polynomial Surface Fit')

# Add a legend to identify the data points
ax.legend()

# Add a color bar to show the mapping of Z values to colors on the surface
fig.colorbar(surf, shrink=0.5, aspect=10, label='Fitted Z Value')

# Display the plot
plt.show()

