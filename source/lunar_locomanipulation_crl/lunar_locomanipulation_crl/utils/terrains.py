import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

import isaaclab.sim as sim_utils


LUNAR_TERRAIN_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=5,
    num_cols=6,
    horizontal_scale=0.2,
    vertical_scale=0.01,
    slope_threshold=1.0,
    use_cache=True,
    sub_terrains={
        "lunar_craters": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.33,
            slope_range=(0.0, 0.5),
            platform_width=0.5,
            border_width=0.25,
        ),
        "lunar_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.33, noise_range=(0.01, 0.1), noise_step=0.02, border_width=0.25
        ),
        "lunar_hills": terrain_gen.HfWaveTerrainCfg(
            proportion=0.33,
            amplitude_range=(0.05, 0.5),
            num_waves=2
        ),
    },
    difficulty_range=(0.0, 1.0)
)
"""Lunar terrains configuration."""

JSC1A_MATERIAL_CFG = sim_utils.RigidBodyMaterialCfg(
    # --- Standard Traction ---
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=1.1,     # High resistance to start moving
    dynamic_friction=0.9,    # High resistance while sliding
    restitution=0.0,         # "Dead" surface (no bounce)
  
    # This mimics the foot sinking ~1-2cm into loose soil.
    # Lower stiffness = deeper sinkage. High damping = energy absorption (sand).
    compliant_contact_stiffness=50000.0,  # Adjustable: Lower values = softer soil
    compliant_contact_damping=4000.0,     # Critical: Prevents bouncing on soft soil
)    

