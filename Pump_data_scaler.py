import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
np.set_printoptions(threshold=sys.maxsize)

# Read data from XLSX file
filename = 'self-sensing-data.xlsx'
try:
    data = pd.read_excel(filename)
except ValueError as e:
    # If reading as Excel fails, try reading as CSV
    if filename.lower().endswith('.csv'):
        print(f"Reading as Excel failed for {filename}: {e}. Trying to read as CSV.")
        data = pd.read_csv(filename)
    else:
        raise

# For every combination of Pump, Trim extract gpm, ft, and HP at 60Hz, 50Hz, and 30Hz
def create_off_speed_test_data(hz_data, flow_data, head_data, power_data):
    off_speed_data = []
    for i in range(len(hz_data)):
        if hz_data[i] in [60, 50, 30]:
            off_speed_data.append({
                'Hz': hz_data[i],
                'GPM': flow_data[i],
                'FT': head_data[i],
                'HP': power_data[i]
            })
    return pd.DataFrame(off_speed_data)
