# Creates a plot of area and percentage covered for detected polygons and their sites

import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pylustrator

#IMPORT DATA

joined_gdf = gpd.read_file('/media/rscph/274b2768-5356-4929-90a9-91f99c20da4c/dw_data/driftwood/results_aerial/driftwood_macs_with_site_info.gpkg')
#joined_gdf['polygon_area'] = joined_gdf['geometry'].area

#sns.displot(joined_gdf, x="polygon_area", hue="target_area", multiple="stack")
#plt.show()

to_plot = 'percentage' #define which field to plot, can be polygon_area or percentage

#PROCESS DATA

df = pd.DataFrame(joined_gdf.drop(columns='geometry'))
df['percentage'] = df['polygon_area'] / df['area'] * 100 #calculate percentage
df['polygon_area'] = df['polygon_area'] / 10000 # convert polygon area to hectare


#prepare df for plotting
df1 = df[df['region'] == 'North Alaska']
df1 = df1.groupby('target_area', as_index=False)[to_plot].sum()

df2 = df[df['region'] == 'West Alaska']
df2 = df2.groupby('target_area', as_index=False)[to_plot].sum()

df3 = df[df['region'] == 'North West Canada']
df3 = df3.groupby('target_area', as_index=False)[to_plot].sum()


#MAKE BARPLOT

#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
pylustrator.start()

fig, axs = plt.subplots(1, 3)

# Plot the first subplot
axs[0].bar(df1['target_area'], df1[to_plot], color='#fde725')
axs[0].set_title('North Alaska', fontsize=8)
axs[0].set_xticklabels(df1['target_area'], rotation=45, ha='right', fontsize=6)
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=6)
axs[0].set_ylim(0, max(df3[to_plot]))  # Set common y-axis scale

# Plot the second subplot
axs[1].bar(df2['target_area'], df2[to_plot], color='#21918c')
axs[1].set_title('West Alaska', fontsize=8)
axs[1].set_xticklabels(df2['target_area'], rotation=45, ha='right', fontsize=6)
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=6)
axs[1].set_ylim(0, max(df3[to_plot]))  # Set common y-axis scale

# Plot the third subplot
axs[2].bar(df3['target_area'], df3[to_plot], color='#440154')
axs[2].set_title('North West Canada', fontsize=8)
axs[2].set_xticklabels(df3['target_area'], rotation=45, ha='right', fontsize=6)
axs[2].set_yticklabels(axs[2].get_yticks(), fontsize=6)
axs[2].set_ylim(0, max(df3[to_plot]))  # Set common y-axis scale



# Add labels and title
for ax in axs:
    ax.set_ylabel('Deposited Driftwood [%]', fontsize=8)
    ax.set_xlabel('Target Site', fontsize=8)


# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
pylustrator.start()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(16.000000/2.54, 8.000000/2.54, forward=True)
plt.figure(1).axes[0].set(position=[0.0924, 0.291, 0.1573, 0.6437], xlabel='', ylim=(0., 1.4))
plt.figure(1).axes[0].get_xaxis().get_label().set(text='')
plt.figure(1).axes[1].set(position=[0.3212, 0.291, 0.4265, 0.6437], ylabel='', ylim=(0., 1.4))
plt.figure(1).axes[1].get_yaxis().get_label().set(text='')
plt.figure(1).axes[2].set(position=[0.8192, 0.291, 0.1573, 0.6437], xlabel='', ylabel='', yticklabels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4'], ylim=(0., 1.4))
plt.figure(1).axes[2].get_xaxis().get_label().set(text='')
plt.figure(1).axes[2].get_yaxis().get_label().set(text='')
#% end: automatic generated code from pylustrator

plt.show()

plt.savefig(f'/home/rscph/Desktop/plots_macs/figure_{to_plot}.png', dpi=400)

# Parameters relative numbers

plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(16.000000/2.54, 8.000000/2.54, forward=True)
plt.figure(1).axes[0].set(position=[0.09164, 0.291, 0.1573, 0.6437], xlabel='', ylim=(0., 1.4))
plt.figure(1).axes[0].get_xaxis().get_label().set(text='')
plt.figure(1).axes[1].set(position=[0.3212, 0.291, 0.4265, 0.6437], ylabel='', ylim=(0., 1.4))
plt.figure(1).axes[1].get_yaxis().get_label().set(text='')
plt.figure(1).axes[2].set(position=[0.8192, 0.291, 0.1573, 0.6437], xlabel='', ylabel='', ylim=(0., 1.4))
plt.figure(1).axes[2].get_xaxis().get_label().set(text='')
plt.figure(1).axes[2].get_yaxis().get_label().set(text='')