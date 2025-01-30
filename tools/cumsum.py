import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# Read the GeoDataFrame
data = gpd.read_file(r'n:\isipd\projects\p_planetdw\data\dw_detection\aerial\results\driftwood_macs.gpkg')

# Extract the 'macs_area' field
macs_area = data['area_macs'] / 10000

# Sort the data
sorted_data = np.sort(macs_area)

# Accumulate the values
accumulated_data = np.cumsum(sorted_data)

# Plot the accumulation without points
plt.plot(sorted_data, accumulated_data, linestyle='-')

# Add individual data points as scatter plot (xs)
#plt.scatter(sorted_data, accumulated_data, s=10)

# Add a vertical line at macs_area = 12
#plt.axvline(x=0.0100, color='red', linestyle='--', marker='x', label="Threshold")

# Add text for the accumulated macs_area
#plt.text(0.01, accumulated_data[np.searchsorted(sorted_data, 0.01)], 
#         f'Area of deposits smaller than 100 m²: {accumulated_data[np.searchsorted(sorted_data, 0.01)]:.4f} ha', 
#         va='bottom', ha='right', fontsize=6, color='red')

# Customize plot labels and ticks
plt.xlabel('Area of Driftwood Deposits [ha]')
plt.ylabel('Accumulated Area of Driftwood Deposits [ha]')
plt.tick_params(axis='both', which='major', labelsize=6)

# Optional: Add a legend
#plt.legend()


# Optional: Pylustrator code for layout (if required)
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(16.260000/2.54, 8.000000/2.54, forward=True)
plt.figure(1).axes[0].set(position=[0.125, 0.1895, 0.775, 0.6905])
#plt.figure(1).axes[0].texts[0].set(position=(4.73, 102.8))
plt.figure(1).axes[0].get_xaxis().get_label().set(position=(0.5, 57.33), fontsize=6.)
plt.figure(1).axes[0].get_yaxis().get_label().set(position=(71.57, 0.5), fontsize=6.)
#% end: automatic generated code from pylustrator

# Show the plot
plt.show()
