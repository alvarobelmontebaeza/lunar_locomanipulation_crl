import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# 1. DATA ENTRY
# ==========================================
data_rows = [
    # --- DIFFICULTY: 2 ---
    ["Rough",  "Loose",   "Diff 2", 0.0339],
    ["Rough",  "Nominal", "Diff 2", 0.0344],
    ["Rough",  "Dense",   "Diff 2", 0.0343],
    ["Hills",  "Loose",   "Diff 2", 0.0347],
    ["Hills",  "Nominal", "Diff 2", 0.0308],
    ["Hills",  "Dense",   "Diff 2", 0.0306],
    ["Crater", "Loose",   "Diff 2", 0.0443],
    ["Crater", "Nominal", "Diff 2", 0.0456],
    ["Crater", "Dense",   "Diff 2", 0.0458],

    # --- DIFFICULTY: 3 ---
    ["Rough",  "Loose",   "Diff 3", 0.0339],
    ["Rough",  "Nominal", "Diff 3", 0.0344],
    ["Rough",  "Dense",   "Diff 3", 0.0343],
    ["Hills",  "Loose",   "Diff 3", 0.0422],
    ["Hills",  "Nominal", "Diff 3", 0.0391],
    ["Hills",  "Dense",   "Diff 3", 0.0370],
    ["Crater", "Loose",   "Diff 3", 0.0628],
    ["Crater", "Nominal", "Diff 3", 0.0648],
    ["Crater", "Dense",   "Diff 3", 0.0648],

    # --- DIFFICULTY: 4 ---
    ["Rough",  "Loose",   "Diff 4", 0.0339],
    ["Rough",  "Nominal", "Diff 4", 0.0344],
    ["Rough",  "Dense",   "Diff 4", 0.0343],
    ["Hills",  "Loose",   "Diff 4", 0.0653],
    ["Hills",  "Nominal", "Diff 4", 0.0484],
    ["Hills",  "Dense",   "Diff 4", 0.0483],
    ["Crater", "Loose",   "Diff 4", 0.0744],
    ["Crater", "Nominal", "Diff 4", 0.0755],
    ["Crater", "Dense",   "Diff 4", 0.0766],
]

df = pd.DataFrame(data_rows, columns=['Terrain', 'Soil', 'Difficulty', 'Error'])

# ==========================================
# 2. PLOTTING CONFIGURATION
# ==========================================
sns.set_theme(style="whitegrid", font_scale=1.1)

g = sns.catplot(
    data=df,
    # --- SWAPPED VARIABLES ---
    x="Difficulty",      # X-Axis is now Difficulty
    col="Terrain",       # Columns are now Terrain
    # -------------------------
    hue="Soil",          # Lines remain Soil
    y="Error",
    kind="point",
    
    # Updated Ordering
    col_order=["Rough", "Hills", "Crater"],
    order=["Diff 2", "Diff 3", "Diff 4"],
    hue_order=["Loose", "Nominal", "Dense"],
    
    # Styling
    palette="deep",
    markers=["o", "s", "^"],
    linestyles=["-", "--", ":"],
    
    height=4, 
    aspect=0.9,
    capsize=0.1
)

# ==========================================
# 3. LABELS
# ==========================================
g.set_axis_labels("Difficulty Level", "Mean Tracking Accuracy (m)")
g.set_titles("Terrain: {col_name}")
g._legend.set_title("Soil Type")

plt.tight_layout()
plt.show()