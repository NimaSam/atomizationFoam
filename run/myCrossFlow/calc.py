import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#path to data file
path = "postProcessing/dropletCloud/0/fz1_cloudData.dat"

# Read the data file
data = []
with open(path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('//') or not line:
            continue  # Skip comments and empty lines
        parts = line.split('\t')
        time = float(parts[0])
        diameter = float(parts[1])
        n_particle = float(parts[3])
        data.append({'time': time, 'diameter': diameter, 'nParticle': n_particle})

df = pd.DataFrame(data)

# Convert diameter from meters to microns
df['diameter_micron'] = df['diameter'] * 1e6

# Define histogram bins: 5 bins based on the range of diameters
min_diameter = df['diameter_micron'].min()
max_diameter = df['diameter_micron'].max()
bins = np.linspace(min_diameter, max_diameter, 6)  # 5 bins

# Function to compute SMD and histogram
def compute_smd_and_histogram(data_subset):
    # Calculate Sauter Mean Diameter (SMD)
    sum_d3 = np.sum(data_subset['nParticle'] * (data_subset['diameter'] ** 3))
    sum_d2 = np.sum(data_subset['nParticle'] * (data_subset['diameter'] ** 2))

    smd_micron = (sum_d3 / sum_d2) * 1e6 if sum_d2 != 0 else 0.0

    # Calculate histogram
    hist_counts, _ = np.histogram(
        data_subset['diameter_micron'], bins=bins, weights=data_subset['nParticle']
    )

    return smd_micron, hist_counts

# Process each unique time step
unique_times = np.sort(df['time'].unique())

# Store SMD results
smd_results = []
all_hist_counts = np.zeros(len(bins) - 1)

for t in unique_times:
    subset = df[df['time'] == t]
    smd, hist_counts = compute_smd_and_histogram(subset)
    smd_results.append({'time': t, 'smd': smd})
    all_hist_counts += hist_counts

# Save SMD to CSV
smd_df = pd.DataFrame(smd_results)
smd_df.to_csv('smd_results.csv', index=False)

# Plot SMD vs Time
plt.figure(figsize=(10, 5))
plt.plot(smd_df['time'], smd_df['smd'], marker='o')
plt.xlabel('Time (s)')
plt.ylabel('Sauter Mean Diameter (microns)')
plt.title('SMD vs Time')
plt.grid()
plt.savefig('smd_vs_time.png')
plt.show()

# Plot Histogram
bin_labels = [f'{int(bins[i])}-{int(bins[i+1])} μm' for i in range(len(bins)-1)]
plt.figure(figsize=(10, 5))
plt.bar(bin_labels, all_hist_counts, width=0.6)
plt.xlabel('Diameter Range (μm)')
plt.ylabel('Particle Count')
plt.title('Particle Distribution Across Diameter Bins')
plt.grid(axis='y')
plt.savefig('particle_histogram.png')
plt.show()

