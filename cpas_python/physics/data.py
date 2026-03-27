"""
CPAS v2 — Physics Data Module
All engineering databases: materials, propellants, altitudes, injectors.
Translated faithfully from the JavaScript source (propulsion-synthesis-v2.jsx).
"""

# ── Materials ──────────────────────────────────────────────────────────────────
MATERIALS = {
    "copper":  {"label": "Copper (CuCrZr)", "k": 320,  "T_limit": 773,  "rho": 8900,
                "sigma_y_base": 420, "color": "#b87333"},
    "steel":   {"label": "Steel (316L)",    "k": 16,   "T_limit": 1073, "rho": 7950,
                "sigma_y_base": 480, "color": "#8899aa"},
    "inconel": {"label": "Inconel 625",     "k": 10,   "T_limit": 1423, "rho": 8440,
                "sigma_y_base": 650, "color": "#c0a060"},
}

# ── Propellants ────────────────────────────────────────────────────────────────
PROPELLANTS = {
    "kerolox": {
        "label": "Kerolox (RP-1 / LOX)",
        "T_flame": 3670, "gamma": 1.24, "Pr": 0.73, "MW": 22.6, "Isp_vac": 358,
        "T_fuel": 289, "T_ox": 90.2,
        "rho_fuel": 820,  "k_cool": 0.13,  "Pr_cool": 5.2,   "mu_cool": 2.5e-3,
        "rho_ox":  1141,  "k_ox":   0.15,  "Pr_ox":   2.1,   "mu_ox":   1.9e-4,
        "T_coolant": 289,
        "OF_nominal": 2.77, "OF_range": [1.5, 4.5],
        "T_coeff":   [-1200, 2600, -380],
        "Isp_coeff": [-80,   160,  -22],
        "Lstar_min": 1.02, "Lstar_opt": 1.27, "Lstar_max": 1.52,
        "color": "#e87040",
        "note": "Dense, high T_flame. RP-1 storable (289 K); LOX cryogenic (90 K). RP-1 excellent coolant.",
    },
    "hydrolox": {
        "label": "Hydrolox (LH₂ / LOX)",
        "T_flame": 3560, "gamma": 1.26, "Pr": 0.74, "MW": 10.0, "Isp_vac": 450,
        "T_fuel": 20.3, "T_ox": 90.2,
        "rho_fuel": 71,   "k_cool": 0.099, "Pr_cool": 0.87,  "mu_cool": 1.1e-5,
        "rho_ox":  1141,  "k_ox":   0.15,  "Pr_ox":   2.1,   "mu_ox":   1.9e-4,
        "T_coolant": 20.3,
        "OF_nominal": 5.0, "OF_range": [3.0, 8.0],
        "T_coeff":   [-600, 240,  -18],
        "Isp_coeff": [-120, 48,   -4],
        "Lstar_min": 0.56, "Lstar_opt": 0.76, "Lstar_max": 0.91,
        "color": "#40b0ff",
        "note": "Highest Isp. LH₂ (20 K) deeply cryogenic. Best regenerative coolant ever used.",
    },
    "methalox": {
        "label": "Methalox (CH₄ / LOX)",
        "T_flame": 3530, "gamma": 1.25, "Pr": 0.73, "MW": 20.1, "Isp_vac": 380,
        "T_fuel": 111.7, "T_ox": 90.2,
        "rho_fuel": 423,  "k_cool": 0.20,  "Pr_cool": 3.2,   "mu_cool": 1.2e-4,
        "rho_ox":  1141,  "k_ox":   0.15,  "Pr_ox":   2.1,   "mu_ox":   1.9e-4,
        "T_coolant": 111.7,
        "OF_nominal": 3.55, "OF_range": [2.0, 5.5],
        "T_coeff":   [-800, 450, -58],
        "Isp_coeff": [-60,  34,  -4.5],
        "Lstar_min": 0.76, "Lstar_opt": 1.02, "Lstar_max": 1.27,
        "color": "#a060ff",
        "note": "Best density-Isp balance. LCH₄ (112 K) cryogenic. Preferred for Raptor-class engines.",
    },
}

# ── Altitudes ──────────────────────────────────────────────────────────────────
ALTITUDES = {
    "sea_level": {"label": "Sea Level (0 km)",      "p_amb": 101325, "T_amb": 288.15, "icon": "🌊"},
    "km5":       {"label": "5 km altitude",          "p_amb":  54048, "T_amb": 255.65, "icon": "⛰"},
    "km10":      {"label": "10 km (airliner)",        "p_amb":  26500, "T_amb": 223.25, "icon": "✈"},
    "km20":      {"label": "20 km (stratosphere)",    "p_amb":   5529, "T_amb": 216.65, "icon": "🌤"},
    "km50":      {"label": "50 km (mesosphere)",      "p_amb":     80, "T_amb": 270.65, "icon": "🌙"},
    "km100":     {"label": "100 km (Kármán line)",    "p_amb":      3, "T_amb": 195.08, "icon": "🚀"},
    "vacuum":    {"label": "Vacuum (space)",           "p_amb":      0, "T_amb": 2.73,   "icon": "🌌"},
}

# ── Injector types ─────────────────────────────────────────────────────────────
INJECTOR_TYPES = {
    "showerhead": {
        "label": "Showerhead (Non-impinging)",
        "C_d": 0.65, "mix_eff": 0.82, "dp_opt": 0.15, "dp_min": 0.10,
        "L_d_min": 5, "theta_inj": 0, "impinges": False, "smf": 1.4,
        "stability": 0.60, "complexity": 0.25, "color": "#60c080",
        "note": "Simplest design. No impingement; relies on turbulence. Historical (V-2).",
    },
    "like_doublet": {
        "label": "Like-on-like Doublet",
        "C_d": 0.72, "mix_eff": 0.88, "dp_opt": 0.18, "dp_min": 0.15,
        "L_d_min": 5, "theta_inj": 25, "impinges": True, "smf": 0.90,
        "stability": 0.70, "complexity": 0.45, "color": "#40d0d0",
        "note": "Fuel-on-fuel + ox-on-ox pairs. Good atomisation via self-impingement.",
    },
    "unlike_doublet": {
        "label": "Unlike Doublet (Fuel–Ox)",
        "C_d": 0.70, "mix_eff": 0.93, "dp_opt": 0.20, "dp_min": 0.15,
        "L_d_min": 5, "theta_inj": 30, "impinges": True, "smf": 0.80,
        "stability": 0.55, "complexity": 0.50, "color": "#50e0a0",
        "note": "Fuel and oxidizer jets impinge directly → intimate mixing. Popular with LOX engines.",
    },
    "triplet": {
        "label": "Unlike Triplet (ox–f–ox)",
        "C_d": 0.70, "mix_eff": 0.94, "dp_opt": 0.20, "dp_min": 0.15,
        "L_d_min": 5, "theta_inj": 25, "impinges": True, "smf": 0.75,
        "stability": 0.65, "complexity": 0.65, "color": "#f0c040",
        "note": "One fuel jet between two oxidizer jets. Handles momentum mismatch well.",
    },
    "pressure_swirl": {
        "label": "Pressure-Swirl (Simplex)",
        "C_d": 0.51, "mix_eff": 0.91, "dp_opt": 0.25, "dp_min": 0.20,
        "L_d_min": 2, "theta_inj": 40, "impinges": False, "smf": 0.55,
        "stability": 0.80, "complexity": 0.70, "color": "#e06080",
        "note": "Liquid spun tangentially → exits as hollow cone. Excellent SMD (~40 µm).",
    },
    "swirl_coax": {
        "label": "Swirl Coaxial (Bi-swirl)",
        "C_d": 0.78, "mix_eff": 0.96, "dp_opt": 0.22, "dp_min": 0.15,
        "L_d_min": 3, "theta_inj": 35, "impinges": False, "smf": 0.50,
        "stability": 0.85, "complexity": 0.78, "color": "#ff8040",
        "note": "Inner and outer co-axial swirlers. Used in Raptor. Highest stability.",
    },
    "pintle": {
        "label": "Pintle",
        "C_d": 0.80, "mix_eff": 0.97, "dp_opt": 0.20, "dp_min": 0.12,
        "L_d_min": 1, "theta_inj": 90, "impinges": True, "smf": 0.70,
        "stability": 0.92, "complexity": 0.55, "color": "#c060ff",
        "note": "Single central element. Inherent stability. Deep throttle (10:1+). Merlin heritage.",
    },
}

# ── Wall layer database (Liu et al. 2023) ─────────────────────────────────────
WALL_LAYERS = {
    "copper_inner": {"label": "Cu Inner",    "k": 320, "E": 120, "alpha": 17e-6, "nu": 0.34, "color": "#b87333"},
    "carbon_phenolic": {"label": "C-Ph Insul","k": 1.2, "E": 30,  "alpha": 5e-6,  "nu": 0.30, "color": "#556677"},
    "steel_shell":  {"label": "Steel Shell", "k": 16,  "E": 193, "alpha": 16e-6, "nu": 0.30, "color": "#8899aa"},
    "ceramic_ablative": {"label": "Ceramic", "k": 1.5, "E": 40,  "alpha": 8e-6,  "nu": 0.25, "color": "#ccaa66"},
    "inconel_shell":{"label": "Inconel",     "k": 10,  "E": 205, "alpha": 13e-6, "nu": 0.31, "color": "#c0a060"},
    "graphite":     {"label": "Graphite",    "k": 25,  "E": 10,  "alpha": 4e-6,  "nu": 0.20, "color": "#446677"},
}

STRESS_CONC = {
    "throat_fillet": 1.8,
    "throat_sharp":  2.5,
}

# ── Constants ──────────────────────────────────────────────────────────────────
R_UNIV   = 8314.0   # J/(kmol·K)
G0       = 9.80665  # m/s²
N_STATIONS = 40     # axial resolution for thermal/structural profiles
