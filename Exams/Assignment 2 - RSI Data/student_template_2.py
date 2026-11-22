
"""
CULMINATING ASSIGNMENT 2: RSI

Create a program which will provide answers to the tasks posed in the assignment description.

For this assignment, you will start from scratch with this blank template.
It is your reponsibility to include all necessary program components in this file.
Your code needs to be successful as well as sufficiently commented/documented to receive full credit.

The relevant data file is named "all_participant_data_rsi.csv" and is located in data > drop-jump.

All file paths need to be relative and execute from this file's directory;
the Visual Studio Code setting "Python > Terminal: Execute in File Dir"
must be enabled in File > Preferences > Settings.

See Canvas for additional requirements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, norm, chisquare, t
from pathlib import Path


# Load the RSI dataset
file_path = Path(__file__).resolve().parent.parent.parent / "data" / "drop-jump" / "all_participant_data_rsi.csv"

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Remove rows with missing measurements to preserve matching pairs
data = data.dropna(subset=["force_plate_rsi", "accelerometer_rsi"])

force_plate = data["force_plate_rsi"]
accelerometer = data["accelerometer_rsi"]

alpha = 0.05

# Task 1: Paired t-test

print("\nTASK 1: PAIRED T-TEST")

# H0: Mean difference between measurement methods is zero
# Ha: Mean difference between measurement methods is not zero

t_stat, p_value = ttest_rel(force_plate, accelerometer)

print("Null hypothesis: Mean difference = 0")
print("Alternative hypothesis: Mean difference != 0")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")

if p_value < alpha:
    print("Reject the null hypothesis.")
    print("The measurement methods have significantly different means.")
else:
    print("Fail to reject the null hypothesis.")
    print("There is insufficient evidence of a difference in means.")



# Task 2: Normal distribution and Chi-square test


print("\nTASK 2: CHI-SQUARE GOODNESS OF FIT")

# Fit a normal distribution to the accelerometer values
mu, sigma = norm.fit(accelerometer)

print(f"Mean: {mu:.4f}")
print(f"Standard deviation: {sigma:.4f}")

# Create bins with equal expected probabilities
n = len(accelerometer)
num_bins = min(10, n // 5)

if num_bins < 4:
    raise ValueError("Not enough observations for this chi-square test.")

probabilities = np.linspace(0, 1, num_bins + 1)
bin_edges = norm.ppf(probabilities, loc=mu, scale=sigma)

# Calculate observed and expected frequencies
observed, _ = np.histogram(accelerometer, bins=bin_edges)
expected = np.full(num_bins, n / num_bins)

# Perform chi-square goodness-of-fit test
# Subtract 2 degrees of freedom for estimated mean and std
chi2_stat, chi2_p = chisquare(observed, expected, ddof=2)

print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"P-value: {chi2_p:.6f}")

if chi2_p < alpha:
    print("Reject the null hypothesis.")
    print("The normal distribution is not a good fit.")
else:
    print("Fail to reject the null hypothesis.")
    print("The normal distribution is a plausible fit.")


# Plot histogram and fitted normal distribution
plt.figure(figsize=(9, 6))

plt.hist(accelerometer, bins=10, density=True,
         alpha=0.6, edgecolor="black",
         label="Accelerometer RSI Data")

x = np.linspace(accelerometer.min(), accelerometer.max(), 200)
y = norm.pdf(x, mu, sigma)

plt.plot(x, y, color="red", linewidth=2,
         label="Fitted Normal Distribution")

plt.xlabel("Reactive Strength Index (RSI)")
plt.ylabel("Probability Density")
plt.title("Normal Distribution Fit of Accelerometer RSI")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Task 3: Participant F confidence intervals

print("\nTASK 3: PARTICIPANT F")

# Select only trials belonging to Participant F
participant_f = data[
    data["trial"].astype(str).str.startswith("F_trial_")
].copy()

print("\nParticipant F Data:")
print(participant_f.to_string(index=False))


# Calculate the 95% confidence interval
def confidence_interval(values):
    values = values.dropna()
    n = len(values)

    if n < 2:
        raise ValueError("At least two trials are required.")

    mean = values.mean()
    std = values.std(ddof=1)

    # Find the critical t-value
    t_critical = t.ppf(0.975, n - 1)

    # Calculate the margin of error
    margin_error = t_critical * std / np.sqrt(n)

    # Calculate the confidence interval bounds
    lower = mean - margin_error
    upper = mean + margin_error

    return mean, lower, upper


# Force plate confidence interval
fp_mean, fp_lower, fp_upper = confidence_interval(
    participant_f["force_plate_rsi"]
)

# Accelerometer confidence interval
acc_mean, acc_lower, acc_upper = confidence_interval(
    participant_f["accelerometer_rsi"]
)

print("\nForce Plate Results:")
print(f"Mean RSI: {fp_mean:.4f}")
print(f"95% CI: ({fp_lower:.4f}, {fp_upper:.4f})")

print("\nAccelerometer Results:")
print(f"Mean RSI: {acc_mean:.4f}")
print(f"95% CI: ({acc_lower:.4f}, {acc_upper:.4f})")

print("\nComparison:")

if (fp_lower <= acc_upper) and (acc_lower <= fp_upper):
    print("The confidence intervals overlap.")
else:
    print("The confidence intervals do not overlap.")

fp_width = fp_upper - fp_lower
acc_width = acc_upper - acc_lower

if fp_width < acc_width:
    print("The force plate has the narrower confidence interval.")
elif acc_width < fp_width:
    print("The accelerometer has the narrower confidence interval.")
else:
    print("Both confidence intervals have the same width.")
