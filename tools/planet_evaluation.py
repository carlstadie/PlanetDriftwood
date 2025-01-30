import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import seaborn as sns

# Load your data
df = gpd.read_file(r'/isipd/projects/p_planetdw/data/outputs/final/prob_0506/comp_0506.gpkg')

planet = 'perc_planet'

min_deposit_size = None

df = df[df['target_area'].isin(['Tuktoyaktuk Coast', 'Mackenzie Delta North', 'Kittigazuit North'])]


Yobs = df['area_macs_sum']
Ypred = df['area_sum']

r2_total = r2_score(Yobs, Ypred)
print(f'total R²: {r2_total}')



# Fit a linear regression model to calculate the slope
model_total = LinearRegression()
# Reshape Yobs for fitting since it needs to be 2D for sklearn
model_total.fit(Yobs.values.reshape(-1, 1), Ypred)
slope_total = model_total.coef_[0]  # Extract the slope
print(f'Total Slope: {slope_total}')

# Calculate bias (Total) using the modified values 
bias_total = ((Ypred - Yobs).sum() / Yobs.sum()) * 100
print(Ypred.sum())

print("Bias (Total):", bias_total)

# Extract unique target areas for creating subplots
target_areas = df['target_area'].unique()
num_areas = len(target_areas)

# Manually define a list of colors (can use any format like hex, RGB, or names)
manual_colors = ['#c2d8e9', '#8b86be', '#832187']  # Example colors, add more if needed

# Define the subplot grid dimensions
fig, axes = plt.subplots(nrows=num_areas, ncols=1, figsize=(6, 6 * num_areas))  # Larger figure size to accommodate square plots
axes = axes.flatten()  # Flatten the axes for easy iteration

# Loop through each target area and corresponding subplot axis
for i, target_area in enumerate(target_areas):
    ax = axes[i]
    area_data = df[df['target_area'] == target_area]

    color = manual_colors[i]  # Use the manually defined color for each plot

    # Add a density heatmap as the background
    x = area_data['perc_macs']
    y = area_data[planet]
    sns.kdeplot(
        x=x, y=y, ax=ax, cmap='Blues', fill=True, alpha=0.5, levels=100, thresh=0.05
    )
    
    # Scatter plot between perc_macs and perc_ps
    ax.scatter(area_data['perc_macs'], area_data[planet], color=color, edgecolor='white', s=30, linewidth=0.5)
    
    # Add line of perfect fit (y=x)
    max_val = max(area_data[['perc_macs', planet]].max())
    ax.plot([0, 30], [0, 30], 'k--', linewidth=0.8)

    # Fit a linear regression model
    X = area_data[['perc_macs']]
    y = area_data[planet]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Plot the regression line
    ax.plot(area_data['perc_macs'], y_pred, color=color, linewidth=1)

    # Calculate R-squared and RMSE
    r2 = r2_score(y, y_pred)
    rmse = mean_squared_error(y, y_pred, squared=False)
    n = area_data.shape[0]

    # Display R-squared and RMSE directly on the plot
    ax.text(0.05, 0.9, f'$r^2$ = {r2:.2f}', transform=ax.transAxes, fontsize=6, color="black")
    ax.text(0.05, 0.8, f'RMSE = {rmse:.2f} %', transform=ax.transAxes, fontsize=6, color="black")
    ax.text(0.05, 0.7, f'n = {n}', transform=ax.transAxes, fontsize=6, color="black")

    # Set titles in compliance with Nature Communications style
    ax.set_title(f'{target_area}', fontsize=8, fontweight='bold')
    
    # Add axis labels to all subplots
    ax.set_ylabel('Driftwood cover % (Planet)', fontsize=6)
    ax.tick_params(axis='both', labelsize=6)  # Set tick labels to size 6
    if i == (num_areas - 1):  # Bottom plot
        ax.set_xlabel('Driftwood cover % (MACS)', fontsize=6)

    # Set the aspect ratio to 'equal' to ensure the plot is square
    ax.set_aspect('equal')

    # Remove the legend
    ax.legend().set_visible(False)

# Adjust layout for clear spacing
plt.tight_layout(pad=1.0)

# Save the plot
plt.savefig(r'N:\isipd\projects\p_planetdw\data\outputs\final\prob_0506\plot.png', dpi=300, bbox_inches='tight')
plt.show()

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps

# Load the GeoPackage
file_path = r'N:\isipd\projects\p_planetdw\data\outputs\final\prob_0506\dw with distance to river mouth indv.gpkg'
# Optionally, save the plot to a file
output_path = r'N:\isipd\projects\p_planetdw\data\outputs\final\prob_0506\deposit_areaindv_by_distance.png'

data = gpd.read_file(file_path)

# Drop rows where 'area' or 'hubdistance' is NaN
data = data.dropna(subset=['area', 'HubDist'])

# Define the thresholds and corresponding labels for the new 'class'
thresholds = [100, 2000, 7000, 19000, 70000, 147674]
labels = ['100-2,000 m²', '2,000-7,000 m²', '7,000-19,000 m²', '19,000-70,000 m²', '> 70,000 m²']

# Create the 'class' field using pd.cut
data['class'] = pd.cut(data['area'], bins=thresholds, labels=labels, include_lowest=True)

# Rename 'class' to 'HubName' for further processing
data['HubName'] = data['class']

# Define the bin size for hubdistance (e.g., 10 units)
bin_size = 10

# Create a new column for binned hubdistance
data['HubDist_Binned'] = (data['HubDist'] // bin_size) * bin_size

# Group by binned hubdistance and HubName (class), summing area
grouped_data = data.groupby(['HubDist_Binned', 'HubName']).agg({'area': 'sum'}).reset_index()

# Divide area by 1,000,000 to convert to millions (if applicable)
grouped_data['area'] = grouped_data['area'] / 1_000_000

# Pivot the data for stacked area plot
pivoted_data = grouped_data.pivot(index='HubDist_Binned', columns='HubName', values='area').fillna(0)

# Sort by HubDist_Binned
pivoted_data = pivoted_data.sort_index()

# Use the BuPu color palette and exclude the lightest shade
cmap = colormaps["YlGn"]
colors = [cmap(i / (len(pivoted_data.columns) + 1)) for i in range(1, len(pivoted_data.columns) + 1)]  # Skip the lightest color

# Set font size globally to 6
plt.rcParams.update({'font.size': 6})

# Plot a stacked area chart
fig, ax = plt.subplots(figsize=(8 / 2.54, 4 / 2.54))  # Convert cm to inches (8x4 cm)

pivoted_data.plot.area(ax=ax, alpha=1, color=colors)

# Add axis labels
ax.set_xlabel('Distance to River (km)', fontsize=6)
ax.set_ylabel('Driftwood Area (km²)', fontsize=6)

# Move legend within the plot without a title or box
ax.legend(loc='upper right', frameon=False, bbox_to_anchor=(0.95, 0.95), fontsize=6)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig(output_path, dpi=300)

# Show the plot
plt.show()