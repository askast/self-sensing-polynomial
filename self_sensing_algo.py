import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- 1. Simulate/Load Data from Excel ---
# In a real scenario, replace 'pump_data.xlsx' with your actual Excel file path.
# Make sure your Excel file has sheets or data with headers: Pump, Trim, Hz, ft, gpm, HP.
# For demonstration, we'll create a dummy DataFrame.
try:
    # Attempt to read from an Excel file if it exists
    # If you have a file named 'pump_data.xlsx' in the same directory, it will be used.
    # Otherwise, a dummy DataFrame will be created.
    df = pd.read_excel('self-sensing-data.xlsx')
except FileNotFoundError:
    # Create a dummy DataFrame if the Excel file is not found
    print("pump_data.xlsx not found. Using dummy data for demonstration.")
    data = {
        'Pump': ['Pump A', 'Pump A', 'Pump A', 'Pump A', 'Pump A', 'Pump A', 'Pump B', 'Pump B', 'Pump B', 'Pump C', 'Pump C'],
        'Trim': ['Trim 1', 'Trim 1', 'Trim 1', 'Trim 2', 'Trim 2', 'Trim 2', 'Trim X', 'Trim X', 'Trim X', 'Trim Z', 'Trim Z'],
        'Hz': [30, 45, 60, 30, 45, 60, 50, 60, 60, 30, 60],
        'ft': [100, 150, 200, 90, 140, 190, 120, 160, 150, 80, 180],
        'gpm': [100, 150, 200, 90, 140, 190, 110, 150, 140, 70, 170],
        'HP': [5, 10, 18, 4, 9, 16, 7, 12, 11, 3, 15]
    }
    df = pd.DataFrame(data)

# Ensure required columns exist
required_columns = ['Pump', 'Trim', 'Hz', 'ft', 'gpm', 'HP']
if not all(col in df.columns for col in required_columns):
    messagebox.showerror("Data Error", "Excel file must contain 'Pump', 'Trim', 'Hz', 'ft', 'gpm', 'HP' columns.")
    exit()

# --- 2. Tkinter Application Setup ---
class PumpPerformanceApp:
    def __init__(self, master):
        self.master = master
        master.title("Pump Performance Analyzer")
        master.geometry("1000x700") # Set initial window size
        master.configure(bg="#f0f0f0") # Light grey background

        # Bind the window closing event to the on_closing method
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- Styling ---
        style = ttk.Style()
        style.theme_use('clam') # Use a modern theme
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Inter", 12))
        style.configure("TButton", font=("Inter", 12, "bold"), background="#4CAF50", foreground="white", borderwidth=0, relief="raised")
        style.map("TButton", background=[('active', '#45a049')]) # Darker green on hover
        style.configure("TMenubutton", font=("Inter", 12), background="#ffffff", borderwidth=1, relief="solid") # Dropdown buttons
        style.configure("Dropdown.TCombobox", fieldbackground="#ffffff", background="#ffffff", selectbackground="#e0e0e0")


        # --- Variables for selections ---
        self.selected_pump = tk.StringVar(master)
        self.selected_trim = tk.StringVar(master)

        # Get unique pumps and set default selection
        self.pumps = sorted(df['Pump'].unique().tolist())
        if self.pumps:
            self.selected_pump.set(self.pumps[0]) # Set initial pump
        else:
            self.selected_pump.set("No Pumps Found")
            messagebox.showwarning("No Data", "No pump data found in the Excel file.")

        # --- UI Layout ---
        # Frame for controls
        self.control_frame = ttk.Frame(master, padding="15 15 15 15", relief="groove")
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)

        # Pump Selection
        ttk.Label(self.control_frame, text="Select Pump:").pack(side=tk.LEFT, padx=10, pady=5)
        self.pump_menu = ttk.OptionMenu(self.control_frame, self.selected_pump, self.selected_pump.get(), *self.pumps, command=self.update_trim_menu)
        self.pump_menu.pack(side=tk.LEFT, padx=10, pady=5)

        # Trim Selection (will be updated dynamically)
        ttk.Label(self.control_frame, text="Select Trim:").pack(side=tk.LEFT, padx=10, pady=5)
        # Initialize with dummy options, will be filled by update_trim_menu
        self.trim_options = []
        self.trim_menu = ttk.OptionMenu(self.control_frame, self.selected_trim, "Select a Trim", *self.trim_options)
        self.trim_menu.pack(side=tk.LEFT, padx=10, pady=5)

        # Plot Button
        self.plot_button = ttk.Button(self.control_frame, text="Plot Performance", command=self.plot_data)
        self.plot_button.pack(side=tk.LEFT, padx=20, pady=5)

        # --- Matplotlib Plot Area ---
        self.figure, self.ax = plt.subplots(figsize=(8, 5)) # Create a figure and an axes
        self.figure.patch.set_facecolor('#f0f0f0') # Match background
        self.canvas = FigureCanvasTkAgg(self.figure, master=master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.toolbar = NavigationToolbar2Tk(self.canvas, master)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Initial update of trim menu and plot
        if self.pumps:
            self.update_trim_menu(self.selected_pump.get())
            self.plot_data() # Plot initial data

    def on_closing(self):
        """Handles the window closing event to gracefully exit the application."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.master.destroy()

    def update_trim_menu(self, *args):
        """Updates the trim dropdown menu based on the selected pump."""
        selected_pump_name = self.selected_pump.get()
        if selected_pump_name == "No Pumps Found":
            self.trim_options = []
        else:
            # Filter DataFrame for the selected pump and get unique trims
            trims_for_pump = df[df['Pump'] == selected_pump_name]['Trim'].unique().tolist()
            self.trim_options = sorted(trims_for_pump)

        # Clear existing options in the trim menu
        self.trim_menu['menu'].delete(0, 'end')

        # Add new options
        if self.trim_options:
            for trim in self.trim_options:
                # Corrected line: Use lambda to set the StringVar
                self.trim_menu['menu'].add_command(label=trim, command=lambda value=trim: self.selected_trim.set(value))
            self.selected_trim.set(self.trim_options[0]) # Set default trim
        else:
            self.selected_trim.set("No Trims Available")

        # Update plot immediately after trim menu update, in case pump changed
        self.plot_data()

    def calculated_offspeed_power(self):
        """Calculates the off-speed power for the selected pump and trim."""
        pump_name = self.selected_pump.get()
        trim_name = float(self.selected_trim.get())

        if pump_name == "No Pumps Found" or trim_name == "No Trims Available":
            return None

        # Filter data for the selected pump and trim
        filtered_df = df[(df['Pump'] == pump_name) & (df['Trim'] == trim_name) & (df['Hz'] == 50)]
        # print(f"filtered_df: {filtered_df}")

        if filtered_df.empty:
            return None

        zero_flow_power_60Hz = df[(df['Pump'] == pump_name) & (df['Trim'] == trim_name) & (df['Hz'] == 60)].sort_values(by='gpm')['HP'].values[0]
        zero_flow_power_30Hz = df[(df['Pump'] == pump_name) & (df['Trim'] == trim_name) & (df['Hz'] == 30)].sort_values(by='gpm')['HP'].values[0]
        zero_flow_power_50Hz = df[(df['Pump'] == pump_name) & (df['Trim'] == trim_name) & (df['Hz'] == 50)].sort_values(by='gpm')['HP'].values[0]

        # curve fit a quadratic polynomial to the zero flow power values
        coeffs = np.polyfit([30, 50, 60], [zero_flow_power_30Hz-(zero_flow_power_50Hz*(30/50)**3), 0, zero_flow_power_60Hz-(zero_flow_power_50Hz*(60/50)**3)], 2)

        # Calculate off-speed power (example calculation)
        unique_frequencies = [20, 30, 40, 50, 55, 60, 65]

        calculated_offspeed_values = []
        for index, row in filtered_df.iterrows():
            # print(f"row: {row}")
            for hz in unique_frequencies:
                power_offset = np.polyval(coeffs, hz)
                data_point = {}
                data_point['Hz'] = hz
                data_point['gpm'] = row['gpm'] * (hz / 50)
                data_point['ft'] = row['ft'] * (hz / 50)**2
                data_point['HP'] = row['HP'] * (hz / 50)**3 + power_offset
                calculated_offspeed_values.append(data_point)
        
        calculated_offspeed_values_df = pd.DataFrame(calculated_offspeed_values, columns=['Hz', 'gpm', 'ft', 'HP'])
        # fit a cubic polynomial surface to the calculated HZ, gpm, HP values
        A = np.vstack([
            np.ones_like(calculated_offspeed_values_df['Hz']),  # Constant term
            calculated_offspeed_values_df['Hz'],                # x
            calculated_offspeed_values_df['HP'],                # y
            calculated_offspeed_values_df['Hz']**2,             # x^2
            calculated_offspeed_values_df['Hz'] * calculated_offspeed_values_df['HP'],       # xy
            calculated_offspeed_values_df['HP']**2,             # y^2
            calculated_offspeed_values_df['Hz']**3,             # x^3
            calculated_offspeed_values_df['Hz']**2 * calculated_offspeed_values_df['HP'],    # x^2y
            calculated_offspeed_values_df['Hz'] * calculated_offspeed_values_df['HP']**2,# xy^2
            calculated_offspeed_values_df['HP']**3,             # y^3
            calculated_offspeed_values_df['Hz']**4,                # x^4
            calculated_offspeed_values_df['Hz']**3 * calculated_offspeed_values_df['HP'],         # x^3y
            calculated_offspeed_values_df['Hz']**2 * calculated_offspeed_values_df['HP']**2,      # x^2y^2
            calculated_offspeed_values_df['Hz'] * calculated_offspeed_values_df['HP']**3,         # xy^3
            calculated_offspeed_values_df['HP']**4              # y^4
        ]).T

        # np.linalg.lstsq returns a tuple; the first element contains the coefficients
        coeffs, residuals, rank, s = np.linalg.lstsq(A, calculated_offspeed_values_df['gpm'], rcond=None)

        return (calculated_offspeed_values_df, coeffs)

    # Define the polynomial function using the calculated coefficients for predictions
    def poly_surface(self, x, y, c):
        return (c[0] +
                c[1] * x + c[2] * y +
                c[3] * x**2 + c[4] * x * y + c[5] * y**2 +
                c[6] * x**3 + c[7] * x**2 * y + c[8] * x * y**2 + c[9] * y**3 
                + c[10] * x**4 + c[11] * x**3 * y + c[12] * x**2 * y**2 + c[13] * x * y**3 + c[14] * y**4
                )
        
    
    def plot_data(self):
        """Plots gpm vs HP for the selected pump and trim, with a line for each Hz."""
        pump_name = self.selected_pump.get()
        trim_name = float(self.selected_trim.get())

        self.ax.clear() # Clear previous plot

        if pump_name == "No Pumps Found" or trim_name == "No Trims Available":
            self.ax.text(0.5, 0.5, "Please select a Pump and Trim to display data.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=14, color='gray')
            self.ax.set_title("Pump Performance (No Data Selected)")
            self.canvas.draw()
            return

        # Filter data based on selections
        filtered_df = df[(df['Pump'] == pump_name) & (df['Trim'] == trim_name)]

        if filtered_df.empty:
            self.ax.text(0.5, 0.5, f"No data found for {pump_name} - {trim_name}.",
                         horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes, fontsize=14, color='gray')
            self.ax.set_title(f"Pump Performance ({pump_name} - {trim_name}) - No Data")
            self.ax.set_xlabel("Flow (gpm)")
            self.ax.set_ylabel("Horsepower (HP)")
            self.canvas.draw()
            return

        calculated_offspeed_values, coeffs = self.calculated_offspeed_power()
        # print(f"calculated_offspeed_values: {calculated_offspeed_values}")

        # Plot for each unique Hz value
        unique_hz = sorted(filtered_df['Hz'].unique())

        # polyfit_flow = self.poly_surface(calculated_offspeed_values['Hz'], calculated_offspeed_values['HP'], coeffs)

        for hz in unique_hz:
            hz_data = filtered_df[filtered_df['Hz'] == hz].sort_values(by='gpm')
            offspeed_data = calculated_offspeed_values[calculated_offspeed_values['Hz'] == hz]
            hz_ones = np.empty(len(offspeed_data))
            hz_ones.fill(hz)
            polyfit_flow = self.poly_surface(hz_ones, offspeed_data['HP'], coeffs)
            if not hz_data.empty:
                self.ax.plot(hz_data['gpm'], hz_data['HP'], marker='o', label=f'{hz} Hz')
                self.ax.plot(offspeed_data['gpm'], offspeed_data['HP'], linestyle='--', label=f'{hz} Hz Calculated', alpha=0.7)
                self.ax.plot(polyfit_flow, offspeed_data['HP'], linestyle=':', label=f'{hz} Hz Poly Fit', alpha=0.7)


        self.ax.set_xlabel("Flow (gpm)", fontsize=12)
        self.ax.set_ylabel("Horsepower (HP)", fontsize=12)
        self.ax.set_title(f"Pump Performance: {pump_name} - {trim_name}", fontsize=14)
        self.ax.legend(title="Frequency", loc='best')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_facecolor('#ffffff') # White plot background

        # Draw the canvas
        self.canvas.draw()

    
# --- Main Application Loop ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PumpPerformanceApp(root)
    root.protocol("WM_DELETE_WINDOW", sys.exit)
    root.mainloop()
