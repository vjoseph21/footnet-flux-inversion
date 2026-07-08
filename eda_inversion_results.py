#!/usr/bin/env python
"""
EDA script for inversion results (inversion.nc)
Explores emissions changes, observation predictions, and background
"""

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path

# Find the latest inversion file
inversion_files = sorted(glob.glob("data/inversions/inversion_data_*.nc"))
if not inversion_files:
    print("No inversion files found!")
    exit(1)

latest_file = inversion_files[-1]
print(f"Loading: {latest_file}\n")

# Load NetCDF
ds = nc.Dataset(latest_file, "r")

print("=" * 70)
print("INVERSION RESULTS SUMMARY")
print("=" * 70)

# Dimensions
print("\nDIMENSIONS:")
print(f"  Time steps (weeks): {ds.dimensions['time'].size}")
print(f"  Grid cells (lat): {ds.dimensions['lat'].size}")
print(f"  Grid cells (lon): {ds.dimensions['lon'].size}")
print(f"  Train observations: {ds.dimensions['train_nobs'].size}")
print(f"  Test observations: {ds.dimensions['test_nobs'].size}")
print(f"  Background octants: {ds.dimensions['bkg_octants'].size}")
print(f"  Background dates: {ds.dimensions['bkg_dates'].size}")

# Dates
dates = ds.variables["date"][:]
print(f"\nINVERSION DATES (weekly Sundays):")
for i, d in enumerate(dates):
    print(f"  Week {i}: {int(d)}")

# Prior emissions
prior_fluxes = ds.variables["prior_fluxes"][:]
print(f"\nPRIOR EMISSIONS (EPA):")
print(f"  Shape: {prior_fluxes.shape}")
print(f"  Min: {np.nanmin(prior_fluxes):.4f} umol/m2/s")
print(f"  Mean: {np.nanmean(prior_fluxes):.4f} umol/m2/s")
print(f"  Max: {np.nanmax(prior_fluxes):.4f} umol/m2/s")
print(f"  Total domain: {np.nansum(prior_fluxes):.2e} umol/m2/s")

# Post (fitted) emissions
post_fluxes = ds.variables["post_fluxes"][:]
print(f"\nFITTED EMISSIONS (post-inversion):")
print(f"  Shape: {post_fluxes.shape} (time, lat, lon)")

for week_idx in range(post_fluxes.shape[0]):
    week_data = post_fluxes[week_idx, :, :]

    # FULL DOMAIN
    change_full = ((np.nansum(week_data) - np.nansum(prior_fluxes)) / np.nansum(prior_fluxes)) * 100
    print(f"  Week {week_idx} (date {int(dates[week_idx])}):")
    print(f"    FULL DOMAIN (240×240):")
    print(f"      Total flux: {np.nansum(week_data):.2e} umol/m2/s")
    print(f"      Change from prior: {change_full:+.1f}%")
    print(f"      Mean: {np.nanmean(week_data):.4f} umol/m2/s")

    # OBSERVATION DOMAIN (inner 40×40 with obs_indices_margin=100)
    obs_margin = 100  # from config
    obs_end = 240 - obs_margin  # 140
    obs_domain = week_data[obs_margin:obs_end, obs_margin:obs_end]
    obs_domain_prior = prior_fluxes[obs_margin:obs_end, obs_margin:obs_end]

    change_obs = ((np.nansum(obs_domain) - np.nansum(obs_domain_prior)) / np.nansum(obs_domain_prior)) * 100
    print(f"    OBSERVATION DOMAIN ONLY (40×40 inner region):")
    print(f"      Total flux: {np.nansum(obs_domain):.2e} umol/m2/s")
    print(f"      Change from prior: {change_obs:+.1f}%")
    print(f"      Mean: {np.nanmean(obs_domain):.4f} umol/m2/s")
    print(f"      Min/Max: {np.nanmin(obs_domain):.4f} / {np.nanmax(obs_domain):.4f}")

# Observation predictions
print(f"\nOBSERVATION PREDICTIONS (Train):")
y_actual_train = ds.variables["y_actual_train"][:]
ye_pred_train = ds.variables["ye_pred_train"][:]
ysim_pred_train = ds.variables["ysim_pred_train"][:]
ybkg_pred_train = ds.variables["ybkg_pred_train"][:]

print(f"  Actual observations: min={y_actual_train.min():.1f}, mean={y_actual_train.mean():.1f}, max={y_actual_train.max():.1f} ppb")
print(f"  Predicted (emissions only): mean={ye_pred_train.mean():.1f} ppb")
print(f"  Predicted (emissions + bkg): mean={ysim_pred_train.mean():.1f} ppb")
print(f"  Background only: mean={ybkg_pred_train.mean():.1f} ppb")

# RMSE
rmse_train = np.sqrt(np.mean((y_actual_train - ysim_pred_train.flatten()) ** 2))
print(f"  RMSE (train): {rmse_train:.1f} ppb")

print(f"\nOBSERVATION PREDICTIONS (Test):")
y_actual_test = ds.variables["y_actual_test"][:]
ysim_pred_test = ds.variables["ysim_pred_test"][:]

print(f"  Actual observations: min={y_actual_test.min():.1f}, mean={y_actual_test.mean():.1f}, max={y_actual_test.max():.1f} ppb")
print(f"  Predicted (emissions + bkg): mean={ysim_pred_test.mean():.1f} ppb")

rmse_test = np.sqrt(np.mean((y_actual_test - ysim_pred_test.flatten()) ** 2))
print(f"  RMSE (test): {rmse_test:.1f} ppb")

# Background
bkg_prior = ds.variables["bkg_prior"][:]
bkg_post = ds.variables["bkg_post"][:]

print(f"\nBACKGROUND (Upwind concentrations):")
print(f"  Prior shape: {bkg_prior.shape} (octants, dates)")
print(f"  Prior mean: {np.nanmean(bkg_prior):.1f} ppb")
print(f"  Prior by octant:")
for oct in range(8):
    print(f"    Octant {oct+1}: {np.nanmean(bkg_prior[oct, :]):.1f} ppb")
print(f"  Post mean: {np.nanmean(bkg_post):.1f} ppb")
print(f"  Change from prior: {((np.nanmean(bkg_post) - np.nanmean(bkg_prior)) / np.nanmean(bkg_prior)) * 100:+.1f}%")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Weekly emissions totals (observation domain only)
ax = axes[0, 0]
obs_margin = 100
obs_end = 240 - obs_margin

# Calculate for observation domain only
weekly_totals_prior_obs = [np.nansum(prior_fluxes[obs_margin:obs_end, obs_margin:obs_end]) for _ in range(post_fluxes.shape[0])]
weekly_totals_post_obs = [np.nansum(post_fluxes[i, obs_margin:obs_end, obs_margin:obs_end]) for i in range(post_fluxes.shape[0])]

x = np.arange(len(dates))
ax.bar(x - 0.2, weekly_totals_prior_obs, 0.4, label="Prior", alpha=0.7)
ax.bar(x + 0.2, weekly_totals_post_obs, 0.4, label="Fitted", alpha=0.7)
ax.set_xlabel("Week")
ax.set_ylabel("Total Flux (umol/m2/s)")
ax.set_title("Weekly Total Emissions: Prior vs Fitted\n(Observation Domain 40×40 Only)")
ax.set_xticks(x)
ax.set_xticklabels([int(d) for d in dates], rotation=45)
ax.legend()
ax.grid(alpha=0.3)

# 2. Observation fit (train)
ax = axes[0, 1]
ax.scatter(y_actual_train, ysim_pred_train.flatten(), alpha=0.6, s=30)
ax.plot([y_actual_train.min(), y_actual_train.max()],
        [y_actual_train.min(), y_actual_train.max()],
        "r--", label="Perfect fit")
ax.set_xlabel("Actual CH4 (ppb)")
ax.set_ylabel("Predicted CH4 (ppb)")
ax.set_title(f"Train Observations: Actual vs Predicted (RMSE={rmse_train:.1f} ppb)")
ax.legend()
ax.grid(alpha=0.3)

# 3. Observation fit (test)
ax = axes[1, 0]
ax.scatter(y_actual_test, ysim_pred_test.flatten(), alpha=0.6, s=30, color="orange")
ax.plot([y_actual_test.min(), y_actual_test.max()],
        [y_actual_test.min(), y_actual_test.max()],
        "r--", label="Perfect fit")
ax.set_xlabel("Actual CH4 (ppb)")
ax.set_ylabel("Predicted CH4 (ppb)")
ax.set_title(f"Test Observations: Actual vs Predicted (RMSE={rmse_test:.1f} ppb)")
ax.legend()
ax.grid(alpha=0.3)

# 4. Background by octant
ax = axes[1, 1]
octants = np.arange(1, 9)
prior_by_oct = [np.nanmean(bkg_prior[i, :]) for i in range(8)]
post_by_oct = [np.nanmean(bkg_post[i, :]) for i in range(8)]
x = np.arange(8)
ax.bar(x - 0.2, prior_by_oct, 0.4, label="Prior", alpha=0.7)
ax.bar(x + 0.2, post_by_oct, 0.4, label="Fitted", alpha=0.7)
ax.set_xlabel("Octant")
ax.set_ylabel("Background CH4 (ppb)")
ax.set_title("Background by Upwind Octant: Prior vs Fitted")
ax.set_xticks(x)
ax.set_xticklabels(octants)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("inversion_results_summary.png", dpi=150, bbox_inches="tight")
print(f"\n✓ Saved visualization to: inversion_results_summary.png")

ds.close()
