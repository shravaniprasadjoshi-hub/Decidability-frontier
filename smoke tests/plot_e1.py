import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the CSV
df = pd.read_csv('b8a_coverage_CA_0_m1_R1000.csv')

# 2. Filter for N=20000 and gamma=0.05 (pick a representative gamma)
df = df[(df['N'] == 20000) & (df['gamma'] == 0.05)]

# 3. Group by 'ng' (the sample size bin) to get the mean coverage
grouped = df.groupby('ng')[['cL', 'cL_nb']].mean().reset_index()

# 4. Plotting
plt.figure(figsize=(10, 6))
x = np.arange(len(grouped['ng']))
width = 0.35

# Plot cL (With composition term) and cL_nb (Without)
plt.bar(x - width/2, grouped['cL'], width, label='CLT + Composition Term', color='steelblue')
plt.bar(x + width/2, grouped['cL_nb'], width, label='CLT without Composition Term', color='salmon')

# Add the nominal 0.95 line
plt.axhline(y=0.95, color='red', linestyle='--', label='Nominal Coverage (0.95)')

plt.xticks(x, grouped['ng'].astype(str), rotation=45)
plt.ylabel('Coverage Rate')
plt.title('E1 Coverage: With vs Without Composition Term (m_min=1, N=20k, gamma=0.05)')
plt.legend()
plt.ylim(0.85, 1.02) 
plt.tight_layout()

# Save it
plt.savefig('e1_coverage_plot.png', dpi=300)
print("Plot saved as e1_coverage_plot.png")