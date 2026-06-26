import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. DATA INPUT (Mean Values from Paper)
# ==========================================

difficulties = np.array([1, 2, 3, 4, 5])
labels = difficulties
x = np.arange(len(labels))  # label locations
width = 0.25  # width of the bars

# --- Position Error Means (m) ---
pos_means = {
    'Rough':   np.array([0.034, 0.034, 0.035, 0.034, 0.034]),
    'Hills':   np.array([0.029, 0.031, 0.037, 0.048, 0.073]),
    'Craters': np.array([0.035, 0.046, 0.065, 0.075, 0.135])
}

# --- Orientation Error Means (deg) ---
rot_means = {
    'Rough':   np.array([6.88, 6.89, 6.89, 6.89, 6.89]),
    'Hills':   np.array([6.76, 6.90, 7.29, 7.84, 8.96]),
    'Craters': np.array([6.86, 7.19, 7.88, 9.30, 9.12])
}

# --- Standard Deviations (PLACEHOLDERS) ---
# IMPORTANT: Replace these arrays with your actual SD data
pos_stds = {
    'Rough': np.array([0.002, 0.003, 0.002, 0.003, 0.002]),
    'Hills': np.array([0.003, 0.004, 0.006, 0.009, 0.015]),
    'Craters': np.array([0.004, 0.008, 0.012, 0.015, 0.030])
}

rot_stds = {
    'Rough':   np.array([1.17, 1.175,1.174,1.175,1.174]),
    'Hills':   np.array([1.08, 1.22, 1.54, 1.89, 2.30]),
    'Craters': np.array([1.23,1.34,1.56,1.89,2.12])
}

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def plot_line_with_shade(ax, means, stds, ylabel, title):
    # Cool-tone palette from your code
    colors = {
        'Rough':   '#17becf',  # Teal
        'Hills':   '#1f77b4',  # Steel Blue
        'Craters': '#756bb1'   # Muted Purple
    }
    
    # Iterate through terrains to plot lines and bands
    for terrain, mean_data in means.items():
        std_data = stds[terrain]
        color = colors[terrain]
        
        # 1. Plot the Mean Line
        ax.plot(difficulties, mean_data, label=terrain, color=color, 
                marker='o', markersize=5, linewidth=2)
        
        # 2. Plot the Shaded Standard Deviation Region
        ax.fill_between(difficulties, 
                        mean_data - std_data, 
                        mean_data + std_data, 
                        color=color, alpha=0.2) # Alpha controls transparency

    # Formatting
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel('Terrain Difficulty Level', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set x-ticks to be integers only (1 to 5)
    ax.set_xticks(difficulties)
    
    # Legend and Grid
    ax.legend(title='Terrain Type', loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Clean spines for journal look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_grouped_bar(ax, means, stds, ylabel, title):
# Cool-tone palette (Harmonizes with Figure 6 blue)
    colors = {
        'Rough':   '#17becf',  # Teal
        'Hills':   '#1f77b4',  # Steel Blue (Standard Matplotlib Blue)
        'Craters': '#756bb1'   # Muted Purple
    }    
    # Create bars for each terrain
    ax.bar(x - width, means['Rough'], width, label='Rough', 
           yerr=stds['Rough'], capsize=4, color=colors['Rough'], alpha=0.9, edgecolor='black')
    
    ax.bar(x, means['Hills'], width, label='Hills', 
           yerr=stds['Hills'], capsize=4, color=colors['Hills'], alpha=0.9, edgecolor='black')
    
    ax.bar(x + width, means['Craters'], width, label='Craters', 
           yerr=stds['Craters'], capsize=4, color=colors['Craters'], alpha=0.9, edgecolor='black')

    # Formatting
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_xlabel('Terrain Difficulty Level', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(title='Terrain Type', loc='upper left', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

# ==========================================
# 3. GENERATE PLOTS
# ==========================================

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# -- Grouped Bar Plots --
# Plot Position Error
plot_grouped_bar(axes[0], pos_means, pos_stds, 
                 ylabel='Mean Position Error (m)', 
                 title='Position Tracking Accuracy')

# Plot Orientation Error
plot_grouped_bar(axes[1], rot_means, rot_stds, 
                 ylabel='Mean Orientation Error (deg)', 
                 title='Orientation Tracking Accuracy')

plt.tight_layout()
plt.show()

# -- Line Plots with Shaded Std Dev --
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
# Plot Position Error
plot_line_with_shade(axes[0], pos_means, pos_stds, 
                     ylabel='Mean Position Error (m)', 
                     title='Position Tracking Accuracy')

# Plot Orientation Error
plot_line_with_shade(axes[1], rot_means, rot_stds, 
                     ylabel='Mean Orientation Error (deg)', 
                     title='Orientation Tracking Accuracy')

plt.tight_layout()
plt.show()