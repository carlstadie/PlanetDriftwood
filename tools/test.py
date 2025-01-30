import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from scipy.stats import norm

# Ensure the output directory exists
output_dir = "/isipd/projects/p_planetdw/data/outputs/final/prob_0506/tests/"
os.makedirs(output_dir, exist_ok=True)

# Load the dataset for area
gdf = gpd.read_file(r'/isipd/projects/p_planetdw/data/outputs/final/prob_0506/dw_final_cluster_info.gpkg')

# Separate data into two groups for the first test (area)
group_one = gdf[gdf['subcluster'] == 'delta']['area']
group_two = gdf[gdf['subcluster'] == 'coast']['area']

area_delta = group_one.values
area_coast = group_two.values

# Compute priors based on the combined dataset for the first test (area)
combined_area = np.concatenate([area_delta, area_coast])
mu_m_area = np.mean(combined_area)
mu_s_area = 2 * np.std(combined_area)

# Define priors and the Bayesian model with Student's t-distribution for the first test (area)
with pm.Model() as model1:
    # Priors for group means
    mu1 = pm.Normal('mu1', mu=np.mean(area_delta), sigma=np.std(area_delta))
    mu2 = pm.Normal('mu2', mu=np.mean(area_coast), sigma=np.std(area_coast))

    # Priors for standard deviations
    sigma1 = pm.HalfCauchy('sigma1', beta=1)
    sigma2 = pm.HalfCauchy('sigma2', beta=1)

    # Likelihoods
    obs1 = pm.Lognormal('obs1', mu=mu1, sigma=sigma1, observed=area_delta)
    obs2 = pm.Lognormal('obs2', mu=mu2, sigma=sigma2, observed=area_coast)

    # Difference of means
    delta = pm.Deterministic('delta', mu1 - mu2)

    # Sampling
    trace1 = pm.sample(2000, return_inferencedata=True, tune=1000, chains=4, cores=2)

# Plot posterior distribution of the difference in means for the first test (area)
delta_posterior = trace1.posterior['delta'].values.flatten()
az.plot_posterior(delta_posterior, hdi_prob=0.95)
plt.title('Posterior Distribution of Mean Difference (Test 1 - Area)')
plt.xlabel('Mean Difference (Delta)')
plt.savefig(f"{output_dir}Delta_Coast_area_posterior.png")
plt.close()

# Compute the Bayes Factor for the first test (area)
delta_posterior_density = az.kde(delta_posterior)
posterior_at_zero = delta_posterior_density[1][np.argmin(np.abs(delta_posterior_density[0]))]
prior_std_delta = np.sqrt(2) * mu_s_area
prior_at_zero = norm.pdf(0, loc=0, scale=prior_std_delta)

# Compute BF10 (evidence for H_A)
bayes_factor = posterior_at_zero / prior_at_zero
with open(f"{output_dir}Delta_Coast_area_bayes_factor.txt", "w") as f:
    f.write(f"Bayes Factor (BF10) in favor of the alternative hypothesis (delta != 0): {bayes_factor:.2f}\n")

if bayes_factor > 10:
    interpretation = "Strong evidence for the alternative hypothesis."
elif bayes_factor > 3:
    interpretation = "Moderate evidence for the alternative hypothesis."
elif bayes_factor > 1:
    interpretation = "Weak evidence for the alternative hypothesis."
else:
    interpretation = "Evidence supports the null hypothesis (no difference)."

with open(f"{output_dir}Delta_Coast_area_bayes_factor.txt", "a") as f:
    f.write(f"Interpretation for Test 1 (Area): {interpretation}\n")

# Separate data into two groups for the second test (area)
group_one_na = gdf[gdf['cluster'].isna()]['area']
group_two_no_na = gdf[gdf['cluster'].notna()]['area']

area_na = group_one_na.values
area_no_na = group_two_no_na.values

# Compute priors based on the combined dataset for the second test (area)
combined_area_na = np.concatenate([area_na, area_no_na])
mu_m_area_na = np.mean(combined_area_na)
mu_s_area_na = 2 * np.std(combined_area_na)

# Define priors and the Bayesian model with Student's t-distribution for the second test (area)
with pm.Model() as model2:
    # Priors for group means
    mu1 = pm.Normal('mu1', mu=np.mean(area_na), sigma=np.std(area_na))
    mu2 = pm.Normal('mu2', mu=np.mean(area_no_na), sigma=np.std(area_no_na))

    # Priors for standard deviations
    sigma1 = pm.HalfCauchy('sigma1', beta=1)
    sigma2 = pm.HalfCauchy('sigma2', beta=1)

    # Likelihoods
    obs1 = pm.Lognormal('obs1', mu=mu1, sigma=sigma1, observed=area_na)
    obs2 = pm.Lognormal('obs2', mu=mu2, sigma=sigma2, observed=area_no_na)

    # Difference of means
    delta = pm.Deterministic('delta', mu1 - mu2)

    # Sampling
    trace2 = pm.sample(2000, return_inferencedata=True, tune=1000, chains=4, cores=2)

# Plot posterior distribution of the difference in means for the second test (area)
delta_posterior2 = trace2.posterior['delta'].values.flatten()
az.plot_posterior(delta_posterior2, hdi_prob=0.95)
plt.title('Posterior Distribution of Mean Difference (Test 2 - Area)')
plt.xlabel('Mean Difference (Delta)')
plt.savefig(f"{output_dir}NA_NonNA_area_posterior.png")
plt.close()

# Compute the Bayes Factor for the second test (area)
delta_posterior_density2 = az.kde(delta_posterior2)
posterior_at_zero2 = delta_posterior_density2[1][np.argmin(np.abs(delta_posterior_density2[0]))]
prior_std_delta2 = np.sqrt(2) * mu_s_area_na
prior_at_zero2 = norm.pdf(0, loc=0, scale=prior_std_delta2)

# Compute BF10 (evidence for H_A)
bayes_factor2 = posterior_at_zero2 / prior_at_zero2
with open(f"{output_dir}NA_NonNA_area_bayes_factor.txt", "w") as f:
    f.write(f"Bayes Factor (BF10) in favor of the alternative hypothesis (delta != 0): {bayes_factor2:.2f}\n")

if bayes_factor2 > 10:
    interpretation2 = "Strong evidence for the alternative hypothesis."
elif bayes_factor2 > 3:
    interpretation2 = "Moderate evidence for the alternative hypothesis."
elif bayes_factor2 > 1:
    interpretation2 = "Weak evidence for the alternative hypothesis."
else:
    interpretation2 = "Evidence supports the null hypothesis (no difference)."

with open(f"{output_dir}NA_NonNA_area_bayes_factor.txt", "a") as f:
    f.write(f"Interpretation for Test 2 (Area): {interpretation2}\n")

# Load the dataset for density
gdf_density = gpd.read_file(r'/isipd/projects/p_planetdw/data/outputs/final/prob_0506/dw_final_gridded_clusters.gpkg')

# Separate data into two groups for the first test (density)
group_one_density = gdf_density[gdf_density['subcluster'] == 'delta']['density']
group_two_density = gdf_density[gdf_density['subcluster'] == 'coast']['density']

density_delta = group_one_density.values
density_coast = group_two_density.values

# Compute priors based on the combined dataset for the first test (density)
combined_density = np.concatenate([density_delta, density_coast])
mu_m_density = np.mean(combined_density)
mu_s_density = 2 * np.std(combined_density)

# Define priors and the Bayesian model with Student's t-distribution for the first test (density)
with pm.Model() as model3:
    # Priors for group means
    mu1 = pm.Normal('mu1', mu=np.mean(density_delta), sigma=np.std(density_delta))
    mu2 = pm.Normal('mu2', mu=np.mean(density_coast), sigma=np.std(density_coast))

    # Priors for standard deviations
    sigma1 = pm.HalfCauchy('sigma1', beta=1)
    sigma2 = pm.HalfCauchy('sigma2', beta=1)

    # Likelihoods
    obs1 = pm.Lognormal('obs1', mu=mu1, sigma=sigma1, observed=density_delta)
    obs2 = pm.Lognormal('obs2', mu=mu2, sigma=sigma2, observed=density_coast)

    # Difference of means
    delta = pm.Deterministic('delta', mu1 - mu2)

    # Sampling
    trace3 = pm.sample(2000, return_inferencedata=True, tune=1000, chains=4, cores=2)

# Plot posterior distribution of the difference in means for the first test (density)
delta_posterior3 = trace3.posterior['delta'].values.flatten()
az.plot_posterior(delta_posterior3, hdi_prob=0.95)
plt.title('Posterior Distribution of Mean Difference (Test 1 - Density)')
plt.xlabel('Mean Difference (Delta)')
plt.savefig(f"{output_dir}Delta_Coast_density_posterior.png")
plt.close()

# Compute the Bayes Factor for the first test (density)
delta_posterior_density3 = az.kde(delta_posterior3)
posterior_at_zero3 = delta_posterior_density3[1][np.argmin(np.abs(delta_posterior_density3[0]))]
prior_std_delta3 = np.sqrt(2) * mu_s_density
prior_at_zero3 = norm.pdf(0, loc=0, scale=prior_std_delta3)

# Compute BF10 (evidence for H_A)
bayes_factor3 = posterior_at_zero3 / prior_at_zero3
with open(f"{output_dir}Delta_Coast_density_bayes_factor.txt", "w") as f:
    f.write(f"Bayes Factor (BF10) in favor of the alternative hypothesis (delta != 0): {bayes_factor3:.2f}\n")

if bayes_factor3 > 10:
    interpretation3 = "Strong evidence for the alternative hypothesis."
elif bayes_factor3 > 3:
    interpretation3 = "Moderate evidence for the alternative hypothesis."
elif bayes_factor3 > 1:
    interpretation3 = "Weak evidence for the alternative hypothesis."
else:
    interpretation3 = "Evidence supports the null hypothesis (no difference)."

with open(f"{output_dir}Delta_Coast_density_bayes_factor.txt", "a") as f:
    f.write(f"Interpretation for Test 1 (Density): {interpretation3}\n")

# Separate data into two groups for the second test (density)
group_one_na_density = gdf_density[gdf_density['cluster'].isna()]['density']
group_two_no_na_density = gdf_density[gdf_density['cluster'].notna()]['density']

density_na = group_one_na_density.values
density_no_na = group_two_no_na_density.values
density_na = density_na/1000000
density_no_na = density_no_na/1000000

# Compute priors based on the combined dataset for the second test (density)
combined_density_na = np.concatenate([density_na, density_no_na])/1000000
mu_m_density_na = np.mean(combined_density_na)
mu_s_density_na = 2 * np.std(combined_density_na)

# Define priors and the Bayesian model with Student's t-distribution for the second test (density)
with pm.Model() as model4:
    # Priors for group means
    mu1 = pm.Lognormal('mu1', mu=np.mean(density_na), sigma=np.std(density_na))
    mu2 = pm.Lognormal('mu2', mu=np.mean(density_no_na), sigma=np.std(density_no_na))

    # Priors for standard deviations
    sigma1 = pm.Lognormal('sigma1', sigma=np.std(density_na))
    sigma2 = pm.Lognormal('sigma2', sigma=np.std(density_no_na))

    # Likelihoods
    obs1 = pm.Lognormal('obs1', mu=mu1, sigma=sigma1, observed=density_na)
    obs2 = pm.Lognormal('obs2', mu=mu2, sigma=sigma2, observed=density_no_na)

    # Difference of means
    delta = pm.Deterministic('delta', mu1 - mu2)

    # Sampling
    trace4 = pm.sample(2000, return_inferencedata=True, tune=1000, chains=4, cores=2)

# Plot posterior distribution of the difference in means for the second test (density)
delta_posterior4 = trace4.posterior['delta'].values.flatten()
az.plot_posterior(delta_posterior4, hdi_prob=0.95)
plt.title('Posterior Distribution of Mean Difference (Test 2 - Density)')
plt.xlabel('Mean Difference (Delta)')
plt.savefig(f"{output_dir}NA_NonNA_density_posterior.png")
plt.close()

# Compute the Bayes Factor for the second test (density)
delta_posterior_density4 = az.kde(delta_posterior4)
posterior_at_zero4 = delta_posterior_density4[1][np.argmin(np.abs(delta_posterior_density4[0]))]
prior_std_delta4 = np.sqrt(2) * mu_s_density_na
prior_at_zero4 = norm.pdf(0, loc=0, scale=prior_std_delta4)

# Compute BF10 (evidence for H_A)
bayes_factor4 = posterior_at_zero4 / prior_at_zero4
with open(f"{output_dir}NA_NonNA_density_bayes_factor.txt", "w") as f:
    f.write(f"Bayes Factor (BF10) in favor of the alternative hypothesis (delta != 0): {bayes_factor4:.2f}\n")

if bayes_factor4 > 10:
    interpretation4 = "Strong evidence for the alternative hypothesis."
elif bayes_factor4 > 3:
    interpretation4 = "Moderate evidence for the alternative hypothesis."
elif bayes_factor4 > 1:
    interpretation4 = "Weak evidence for the alternative hypothesis."
else:
    interpretation4 = "Evidence supports the null hypothesis (no difference)."

with open(f"{output_dir}NA_NonNA_density_bayes_factor.txt", "a") as f:
    f.write(f"Interpretation for Test 2 (Density): {interpretation4}\n")


from scipy.stats import mannwhitneyu
import math

# Perform Mann-Whitney U test for the first test (area, delta vs. coast)
stat_area_1, p_value_area_1 = mannwhitneyu(area_delta, area_coast, alternative='two-sided')

# Calculate z-score for the first test
n1_area = len(area_delta)
n2_area = len(area_coast)
mu_u_area = n1_area * n2_area / 2  # Mean of U
sigma_u_area = math.sqrt(n1_area * n2_area * (n1_area + n2_area + 1) / 12)  # Std dev of U
z_area_1 = (stat_area_1 - mu_u_area) / sigma_u_area  # z-score

with open(f"{output_dir}Delta_Coast_area_mannwhitney.txt", "w") as f:
    f.write(f"Mann-Whitney U Test Results (Test 1 - Area):\n")
    f.write(f"U Statistic: {stat_area_1}\n")
    f.write(f"Z-Score: {z_area_1:.4f}\n")
    f.write(f"P-Value: {p_value_area_1}\n")
    if p_value_area_1 < 0.05:
        f.write("Interpretation: Significant difference between groups.\n")
    else:
        f.write("Interpretation: No significant difference between groups.\n")

# Repeat for the second test (area, NA vs. Non-NA)
stat_area_2, p_value_area_2 = mannwhitneyu(area_na, area_no_na, alternative='two-sided')
n1_area_na = len(area_na)
n2_area_na = len(area_no_na)
mu_u_area_na = n1_area_na * n2_area_na / 2
sigma_u_area_na = math.sqrt(n1_area_na * n2_area_na * (n1_area_na + n2_area_na + 1) / 12)
z_area_2 = (stat_area_2 - mu_u_area_na) / sigma_u_area_na

with open(f"{output_dir}NA_NonNA_area_mannwhitney.txt", "w") as f:
    f.write(f"Mann-Whitney U Test Results (Test 2 - Area):\n")
    f.write(f"U Statistic: {stat_area_2}\n")
    f.write(f"Z-Score: {z_area_2:.4f}\n")
    f.write(f"P-Value: {p_value_area_2}\n")
    if p_value_area_2 < 0.05:
        f.write("Interpretation: Significant difference between groups.\n")
    else:
        f.write("Interpretation: No significant difference between groups.\n")

# Perform Mann-Whitney U test for the first test (density, delta vs. coast)
stat_density_1, p_value_density_1 = mannwhitneyu(density_delta, density_coast, alternative='two-sided')
n1_density = len(density_delta)
n2_density = len(density_coast)
mu_u_density = n1_density * n2_density / 2
sigma_u_density = math.sqrt(n1_density * n2_density * (n1_density + n2_density + 1) / 12)
z_density_1 = (stat_density_1 - mu_u_density) / sigma_u_density

with open(f"{output_dir}Delta_Coast_density_mannwhitney.txt", "w") as f:
    f.write(f"Mann-Whitney U Test Results (Test 1 - Density):\n")
    f.write(f"U Statistic: {stat_density_1}\n")
    f.write(f"Z-Score: {z_density_1:.4f}\n")
    f.write(f"P-Value: {p_value_density_1}\n")
    if p_value_density_1 < 0.05:
        f.write("Interpretation: Significant difference between groups.\n")
    else:
        f.write("Interpretation: No significant difference between groups.\n")

# Repeat for the second test (density, NA vs. Non-NA)
stat_density_2, p_value_density_2 = mannwhitneyu(density_na, density_no_na, alternative='two-sided')
n1_density_na = len(density_na)
n2_density_na = len(density_no_na)
mu_u_density_na = n1_density_na * n2_density_na / 2
sigma_u_density_na = math.sqrt(n1_density_na * n2_density_na * (n1_density_na + n2_density_na + 1) / 12)
z_density_2 = (stat_density_2 - mu_u_density_na) / sigma_u_density_na

with open(f"{output_dir}NA_NonNA_density_mannwhitney.txt", "w") as f:
    f.write(f"Mann-Whitney U Test Results (Test 2 - Density):\n")
    f.write(f"U Statistic: {stat_density_2}\n")
    f.write(f"Z-Score: {z_density_2:.4f}\n")
    f.write(f"P-Value: {p_value_density_2}\n")
    if p_value_density_2 < 0.05:
        f.write("Interpretation: Significant difference between groups.\n")
    else:
        f.write("Interpretation: No significant difference between groups.\n")


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import linregress

# Load the GeoPackage
file_path = r'/isipd/projects/p_planetdw/data/outputs/final/prob_0506/catchments.gpkg'
data = gpd.read_file(file_path)

# Extract variables
bin_midpoints = data['boreal_area']
area_by_bin = data['area_sum']

# Use Spearman correlation
spearman_corr_prop, spearman_p_prop = stats.spearmanr(bin_midpoints, area_by_bin)

# Use linear regression
slope_regression, intercept_regression, _, p_value_regression, _ = linregress(bin_midpoints, area_by_bin)

# Print results
print(f"Spearman correlation (perc_boreal vs area_sum): {spearman_corr_prop:.4f}")
print(f"Spearman p-value: {spearman_p_prop:.4e}")
print(f"Linear regression slope: {slope_regression:.4f}, intercept: {intercept_regression:.4f}")
print(f"Linear regression p-value: {p_value_regression:.4e}")

# Data preparation for Bayesian analysis
x = bin_midpoints
y = area_by_bin

# Standardize data
x_standardized = (x - np.mean(x)) / np.std(x)
y_standardized = (y - np.mean(y)) / np.std(y)

# Null Model (H0: No Correlation)
with pm.Model() as null_model:
    sigma_x = pm.HalfNormal("sigma_x", sigma=1)
    sigma_y = pm.HalfNormal("sigma_y", sigma=1)
    
    # Covariance matrix (no correlation, rho=0)
    cov = pm.math.stack([[sigma_x**2, 0],
                         [0, sigma_y**2]])
    
    observed = pm.MvNormal("observed", mu=[0, 0], cov=cov, observed=np.column_stack([x_standardized, y_standardized]))
    
    # Sample posterior
    trace_null = pm.sample(
        2000,
        tune=1000,
        return_inferencedata=True,
        target_accept=0.95,
        idata_kwargs={"log_likelihood": True}  # Include log-likelihood
    )

# Alternative Model (H1: Correlation Exists)
with pm.Model() as alt_model:
    rho = pm.Uniform("rho", -1, 1)  # Correlation coefficient
    sigma_x = pm.HalfNormal("sigma_x", sigma=1)
    sigma_y = pm.HalfNormal("sigma_y", sigma=1)
    
    # Covariance matrix (includes correlation)
    cov = pm.math.stack([[sigma_x**2, rho * sigma_x * sigma_y],
                         [rho * sigma_x * sigma_y, sigma_y**2]])
    
    # Student's t-likelihood for robustness
    observed = pm.MvNormal("observed", mu=[0, 0], cov=cov, observed=np.column_stack([x_standardized, y_standardized]))
    # Sample posterior
    trace_alt = pm.sample(
        2000,
        tune=1000,
        return_inferencedata=True,
        target_accept=0.95,
        idata_kwargs={"log_likelihood": True}  # Include log-likelihood
    )

# Compute marginal likelihoods using LOO
loo_null = az.loo(trace_null)
loo_alt = az.loo(trace_alt)

# Check Pareto k values
print("Pareto k values (Alternative):", loo_alt.pareto_k)

# Extract log marginal likelihoods
log_marginal_null = loo_null.elpd_loo
log_marginal_alt = loo_alt.elpd_loo

# Compute Bayes Factor
bf10 = np.exp(log_marginal_alt - log_marginal_null)

# Print Bayes Factor results
print(f"Log Marginal Likelihood (Null): {log_marginal_null:.2f}")
print(f"Log Marginal Likelihood (Alternative): {log_marginal_alt:.2f}")
print(f"Bayes Factor (BF10): {bf10:.2f}")

# Compute WAIC for both models
waic_null = az.waic(trace_null)
waic_alt = az.waic(trace_alt)

# Print WAIC results
try:
    print(f"WAIC Null: {waic_null.waic:.2f}, WAIC Alt: {waic_alt.waic:.2f}")
    print(f"WAIC p_waic (Null): {waic_null.p_waic:.2f}, WAIC p_waic (Alt): {waic_alt.p_waic:.2f}")
except AttributeError:
    print(f"WAIC Null:\n{waic_null}")
    print(f"WAIC Alt:\n{waic_alt}")

# Optional: Fallback to manual model comparison
print("\nConsider alternative metrics like Bayes Factor or posterior predictive checks if WAIC is unreliable.")

rho_summary = az.summary(trace_alt, var_names=["rho"], hdi_prob=0.95)
print("\nPosterior Summary for rho:")
print(rho_summary)

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import linregress


# Load the GeoPackage
file_path = r'/isipd/projects/p_planetdw/data/outputs/final/prob_0506/catchments.gpkg'
data = gpd.read_file(file_path)

# Remove rows with NaN or infinite values
data = data[~data[['HubDist', 'area_sum']].isna().any(axis=1)]
data = data[(~data['HubDist'].isin([float('inf'), -float('inf')])) & 
            (~data['area_sum'].isin([float('inf'), -float('inf')]))]

# Bin distances and calculate proportion of total area in each bin
bin_size = 10  # Define bin size for distances
data['distance_bin'] = pd.cut(data['HubDist'], bins=range(0, int(data['HubDist'].max()) + bin_size, bin_size))
area_by_bin = data.groupby('distance_bin')['area_sum'].sum()  # Sum of area in each bin
area_by_bin = area_by_bin / 1_000_000  # Convert to km²
area_by_bin_prop = area_by_bin / area_by_bin.sum()  # Proportion of total area
bin_midpoints = np.array([interval.mid for interval in area_by_bin.index])

# Adjust proportions to avoid zero values for log-transformation
epsilon = 1e-6  # Small adjustment value
area_by_bin_prop_adjusted = area_by_bin_prop + epsilon

# Use Spearman correlation
spearman_corr_prop, spearman_p_prop = stats.spearmanr(bin_midpoints, area_by_bin)

# Use linear regression
slope_regression, intercept_regression, _, p_value_regression, _ = linregress(bin_midpoints, area_by_bin)

# Print results
print(f"Spearman correlation (distance vs proportion of total area): {spearman_corr_prop:.4f}")
print(f"Spearman p-value: {spearman_p_prop:.4e}")
print(f"Linear regression slope: {slope_regression:.4f}, intercept: {intercept_regression:.4f}")
print(f"Linear regression p-value: {p_value_regression:.4e}")

# Data preparation for Bayesian analysis
x = bin_midpoints
y = area_by_bin

# Standardize data
x_standardized = (x - np.mean(x)) / np.std(x)
y_standardized = (y - np.mean(y)) / np.std(y)

# Null Model (H0: No Correlation)
with pm.Model() as null_model:
    sigma_x = pm.HalfNormal("sigma_x", sigma=1)
    sigma_y = pm.HalfNormal("sigma_y", sigma=1)
    
    # Covariance matrix (no correlation, rho=0)
    cov = pm.math.stack([[sigma_x**2, 0],
                         [0, sigma_y**2]])
    
    # Multivariate normal likelihood
    observed = pm.MvNormal("observed", mu=[0, 0], cov=cov, observed=np.column_stack([x_standardized, y_standardized]))
    
    # Sample posterior
    trace_null = pm.sample(
        2000,
        tune=1000,
        return_inferencedata=True,
        target_accept=0.95,
        idata_kwargs={"log_likelihood": True}  # Include log-likelihood
    )

# Alternative Model (H1: Correlation Exists)
with pm.Model() as alt_model:
    rho = pm.Uniform("rho", -1, 1)  # Correlation coefficient
    sigma_x = pm.HalfNormal("sigma_x", sigma=1)
    sigma_y = pm.HalfNormal("sigma_y", sigma=1)
    
    # Covariance matrix (includes correlation)
    cov = pm.math.stack([[sigma_x**2, rho * sigma_x * sigma_y],
                         [rho * sigma_x * sigma_y, sigma_y**2]])
    
    # Multivariate normal likelihood
    observed = pm.MvNormal("observed", mu=[0, 0], cov=cov, observed=np.column_stack([x_standardized, y_standardized]))
    
    # Sample posterior
    trace_alt = pm.sample(
        2000,
        tune=1000,
        return_inferencedata=True,
        target_accept=0.95,
        idata_kwargs={"log_likelihood": True}  # Include log-likelihood
    )

# Compute marginal likelihoods using LOO
loo_null = az.loo(trace_null)
loo_alt = az.loo(trace_alt)

# Extract log marginal likelihoods
log_marginal_null = loo_null.elpd_loo
log_marginal_alt = loo_alt.elpd_loo

# Compute Bayes Factor
bf10 = np.exp(log_marginal_alt - log_marginal_null)

# Print Bayes Factor results
print(f"Log Marginal Likelihood (Null): {log_marginal_null:.2f}")
print(f"Log Marginal Likelihood (Alternative): {log_marginal_alt:.2f}")
print(f"Bayes Factor (BF10): {bf10:.2f}")

# Summarize rho
rho_summary = az.summary(trace_alt, var_names=["rho"], hdi_prob=0.95)
print("\nPosterior Summary for rho:")
print(rho_summary)

# Visualize posterior distribution of rho
az.plot_posterior(trace_alt, var_names=["rho"], hdi_prob=0.95)
plt.show()


