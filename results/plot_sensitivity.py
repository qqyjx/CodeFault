
import matplotlib.pyplot as plt
import json
import numpy as np

# Load data
with open('sensitivity_figure_data.json', 'r') as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(8, 5))

time_windows = data['time_windows']
colors = {'CodeFault': '#2ecc71', 'DeepJIT': '#3498db', 'CC2Vec': '#e74c3c', 'LogRobust': '#9b59b6'}
markers = {'CodeFault': 'o', 'DeepJIT': 's', 'CC2Vec': '^', 'LogRobust': 'D'}

for model_name, model_data in data['models'].items():
    aucs = model_data['auc']
    stds = model_data['std']
    
    ax.errorbar(time_windows, aucs, yerr=stds, 
                label=model_name, color=colors.get(model_name, 'gray'),
                marker=markers.get(model_name, 'o'), markersize=8,
                linewidth=2, capsize=3)

ax.set_xlabel('Time Window Δt (hours)', fontsize=12)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Sensitivity Analysis: Fault Manifestation Time Window', fontsize=14)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(time_windows)
ax.set_ylim(0.7, 0.92)

plt.tight_layout()
plt.savefig('sensitivity_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
print("Figure saved!")
