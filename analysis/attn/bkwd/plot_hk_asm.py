import matplotlib.pyplot as plt
import numpy as np


# MI350x, MHA, non-causal, N=4096
data = {
    "HK 4-Warps": 668.37,
    "HK ASM 4-Warps": 814.93,
    "AITER (AMD)": 824.74
}

colors = ["#7CB9BC", "#8E69B8", "#E59952"]  # HK teal, HK ASM purple, AITER orange

# Extract data
labels = list(data.keys())
values = list(data.values())

# Create bar chart
x = np.arange(len(labels))
width = 0.5

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x, values, width, color=colors)

max_tflops = max(values)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
            f'{value:.0f}', ha='center', va='bottom', fontsize=14)

# add some padding to the top of the y-axis to prevent label overlap
ax.set_ylim(0, max_tflops * 1.15)
ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
ax.set_title('MHA Non-Causal Backward Performance Comparison MI350X (N=4096)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.tick_params(axis='y', labelsize=16)

plt.tight_layout()
plt.show()

output_file = 'mi350x_mha_non_causal_hk_asm_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_file}")

# Print summary
print(f"HK 4-Warps: {data['HK 4-Warps']:.2f} TFLOPS")
print(f"HK ASM 4-Warps: {data['HK ASM 4-Warps']:.2f} TFLOPS")
print(f"AITER (AMD): {data['AITER (AMD)']:.2f} TFLOPS")
print(f"Speedup (ASM vs HK): {data['HK ASM 4-Warps'] / data['HK 4-Warps']:.2f}x")
print(f"Gap to AITER: {(data['AITER (AMD)'] - data['HK ASM 4-Warps']) / data['HK ASM 4-Warps'] * 100:.1f}%")
