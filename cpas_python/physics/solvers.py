"""
CPAS v2 — Physics Solvers
All engineering functions translated from propulsion-synthesis-v2.jsx.
Pure Python / NumPy — no Dash dependencies here.
"""
import math
import random
from typing import Optional
from .data import (
    MATERIALS, PROPELLANTS, INJECTOR_TYPES, WALL_LAYERS, STRESS_CONC,
    R_UNIV, G0, N_STATIONS
)

# ── Utilities ──────────────────────────────────────────────────────────────────

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def lerp(a, b, t):
    return a + (b - a) * t

def wall_thickness_m(p, r_t_m=None):
    """
    Convert normalised wallThickness [0,1] to physical wall thickness [m].
    Scales with throat radius so the wall is always proportionate to the engine.
    For a 25 mm throat at wallThickness=0.4: t ≈ 3.2 mm  (realistic regen wall).
    """
    wt  = clamp(p.get("wallThickness", 0.40), 0.05, 0.95)
    r_t = r_t_m or p.get("throatRadius", 0.025)
    return r_t * (0.08 + wt * 0.12)

# ── Injector helpers ───────────────────────────────────────────────────────────

def size_orifice(mdot_total, n_elem, Cd, rho, dP_Pa):
    dP_safe = max(dP_Pa, 1000)
    A_total = mdot_total / (Cd * math.sqrt(2 * rho * dP_safe))
    A_elem  = A_total / max(n_elem, 1)
    d_mm    = math.sqrt(4 * A_elem / math.pi) * 1000
    return {"A_total": A_total, "A_elem": A_elem, "d_mm": d_mm}

def orifice_velocity(mdot, rho, A_total):
    return mdot / (rho * max(A_total, 1e-10))

def impingement_resultant(mdot1, v1, theta1_deg, mdot2, v2, theta2_deg):
    t1 = math.radians(theta1_deg)
    t2 = math.radians(theta2_deg)
    mx = mdot1 * v1 * math.sin(t1) - mdot2 * v2 * math.sin(t2)
    my = mdot1 * v1 * math.cos(t1) + mdot2 * v2 * math.cos(t2)
    return math.degrees(math.atan2(abs(mx), my))

def smd_impinging(d_mm, v_jet):
    """Lefebvre SMD correlation for impinging doublets."""
    return clamp(880 * (d_mm * 1e-3) ** 0.5 * max(v_jet, 1) ** (-0.7) * 1e6, 5, 500)

def vaporization_length(SMD_um, v_droplet, prop_key):
    """Ceotto D² law vaporisation length."""
    t_d_ref = 0.83  # ms for kerolox reference
    D_ref   = 40    # µm reference diameter
    factor  = {"hydrolox": 0.4, "methalox": 0.75}.get(prop_key, 1.0)
    t_d_ms  = t_d_ref * factor * (SMD_um / D_ref) ** 2
    L_vap_mm = max(v_droplet, 1) * (t_d_ms / 1000) * 1000
    return {"t_d_ms": t_d_ms, "L_vap_mm": L_vap_mm}

def momentum_ratio(mdot_ox, v_ox, mdot_f, v_f):
    return (mdot_ox * v_ox) / max(mdot_f * v_f, 1e-9)

# ── Thermodynamics ─────────────────────────────────────────────────────────────

def c_star(T_c, gamma, MW):
    """Characteristic velocity c* [m/s]."""
    R_spec = R_UNIV / MW
    G = math.sqrt(gamma) * (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    return math.sqrt(R_spec * T_c) / G

def prop_at_of(prop_key, OF):
    """T_flame, Isp_vac, gamma at given O/F via quadratic fit."""
    prop = PROPELLANTS[prop_key]
    dr   = OF - prop["OF_nominal"]
    a0, a1, a2 = prop["T_coeff"]
    b0, b1, b2 = prop["Isp_coeff"]
    T_fl = clamp(prop["T_flame"] + a1*dr + a2*dr*dr, 1800, prop["T_flame"] * 1.02)
    Isp  = clamp(prop["Isp_vac"] + b1*dr + b2*dr*dr, 150, prop["Isp_vac"] * 1.02)
    t    = clamp((OF - prop["OF_range"][0]) / (prop["OF_range"][1] - prop["OF_range"][0]), 0, 1)
    gamma_of = lerp(prop["gamma"] - 0.02, prop["gamma"] + 0.02, t)
    return {"T_flame": T_fl, "Isp_vac": Isp, "gamma": gamma_of}

def exit_pressure_ratio(gamma, nozzle_AR):
    M_e = mach_from_area_ratio(nozzle_AR, supersonic=True, gamma=gamma)
    g1  = (gamma - 1) / 2
    return (1 + g1 * M_e * M_e) ** (-gamma / (gamma - 1))

def optimal_nozzle_ar(gamma, p_c_Pa, p_amb_Pa):
    """Find ε_e where p_exit = p_ambient (maximum C_F)."""
    if p_amb_Pa <= 0:
        return 100.0
    target = p_amb_Pa / max(p_c_Pa, 1)
    if target >= 1:
        return 2.0
    g1  = (gamma - 1) / 2
    exp = -gamma / (gamma - 1)
    M   = 2.0
    for _ in range(40):
        pe_pc  = (1 + g1 * M * M) ** exp
        dpe_dM = exp * (1 + g1 * M * M) ** (exp - 1) * 2 * g1 * M
        if abs(dpe_dM) < 1e-12:
            break
        dM = -(pe_pc - target) / dpe_dM
        M  = clamp(M + dM, 1.001, 30)
        if abs(dM) < 1e-7:
            break
    AR = ((2 / (gamma + 1)) * (1 + g1 * M * M)) ** ((gamma + 1) / (2 * (gamma - 1))) / M
    return clamp(AR, 2, 100)

def thrust_coeff(gamma, nozzle_AR=8, p_c_Pa=5e6, p_amb_Pa=0):
    """Thrust coefficient C_F (Ceotto/Tavares eq. 4.6, Sutton §3)."""
    eps_e  = clamp(nozzle_AR, 1.5, 200)
    g1     = gamma - 1
    pe_pc  = exit_pressure_ratio(gamma, eps_e)
    # Momentum term
    CF_mom = math.sqrt(
        2 * gamma**2 / g1
        * (2 / (gamma + 1)) ** ((gamma + 1) / g1)
        * (1 - pe_pc ** (g1 / gamma))
    )
    # Pressure thrust term
    CF_pres = (pe_pc - p_amb_Pa / p_c_Pa) * eps_e
    return max(CF_mom + CF_pres, 0.0)

def isp_at_altitude(Isp_vac, CF_amb, CF_vac):
    return Isp_vac * CF_amb / max(CF_vac, 1e-9)

# ── Mach solver ────────────────────────────────────────────────────────────────

def mach_from_area_ratio(AR, supersonic=False, gamma=1.25):
    """Isentropic area-Mach solver with Newton + bisection fallback."""
    if AR <= 1.0:
        return 1.0
    g1  = gamma - 1
    exp = (gamma + 1) / (2 * g1)

    def area_fn(M):
        return (2 / (gamma + 1)) ** exp * (1 + g1 / 2 * M * M) ** exp / max(M, 1e-9)

    M         = 2.0 if supersonic else 0.3
    converged = False
    residual  = float('inf')

    for _ in range(40):
        A    = area_fn(M)
        f    = A - AR
        dfdM = A * (exp * g1 * M / (1 + g1 / 2 * M * M) - 1 / M)
        if abs(dfdM) < 1e-12:
            break
        dM = -f / dfdM
        M  = clamp(M + dM, 1.001 if supersonic else 0.01,
                            30    if supersonic else 0.9999)
        residual = abs(area_fn(M) - AR)
        if residual < 1e-7:
            converged = True
            break

    if not converged or residual > 1e-4:
        lo = 1.001 if supersonic else 0.001
        hi = 30    if supersonic else 0.9999
        for _ in range(60):
            mid  = (lo + hi) / 2
            fmid = area_fn(mid) - AR
            flo  = area_fn(lo)  - AR
            if fmid * flo <= 0:
                hi = mid
            else:
                lo = mid
            residual = abs(fmid)
            if residual < 1e-6:
                M = mid
                converged = True
                break
        if not converged:
            M = (lo + hi) / 2

    return clamp(M, 0.001, 30)

# ── Geometry ───────────────────────────────────────────────────────────────────

def build_geometry_profile(p, prop_key):
    """
    Build full SI-dimensioned engine geometry from parameter dict.
    Returns a geometry dict with all dimensions, zone arrays, and
    the multi-criterion chamber sizing result.
    """
    prop   = PROPELLANTS[prop_key]
    p_c_Pa = p["chamberPressure"] * 1e6

    r_t    = clamp(p["throatRadius"], 0.005, 0.50)
    A_t    = math.pi * r_t * r_t
    eps_c  = clamp(p["contractionRatio"], 2.0, 8.0)
    r_c    = r_t * math.sqrt(eps_c)

    Lstar  = clamp(p["Lstar"], prop["Lstar_min"] * 0.85, prop["Lstar_max"] * 1.30)
    V_c    = Lstar * A_t

    theta_c = clamp(p["convergentAngle"], 20, 60) * math.pi / 180
    L_cvg   = (r_c - r_t) / math.tan(theta_c)
    V_cvg   = (math.pi / 3) * L_cvg * (r_c*r_c + r_c*r_t + r_t*r_t)

    # Multi-criterion chamber sizing
    chamber_sizing = size_chamber(r_t, r_c, Lstar, prop,
                                   p.get("injector", "like_doublet"), p_c_Pa)
    L_cyl_Lstar = max(V_c - V_cvg, 0) / (math.pi * r_c * r_c)
    L_cyl       = max(chamber_sizing["L_cyl_m"], L_cyl_Lstar)

    theta_d = clamp(p["nozzleAngle"], 8, 45) * math.pi / 180
    eps_e   = clamp(p["nozzleAR"], 2.0, 40.0)
    r_e     = r_t * math.sqrt(eps_e)
    L_noz   = (r_e - r_t) / math.tan(theta_d)

    R_thr         = 0.382 * r_t
    L_thr_straight = R_thr * 0.3
    L_inj          = 0.05 * r_c
    L_total        = L_inj + L_cyl + L_cvg + L_thr_straight + L_noz

    # Zone boundaries
    z_inj_end = L_inj
    z_cyl_end = L_inj + L_cyl
    z_cvg_end = L_inj + L_cyl + L_cvg
    z_thr_end = L_inj + L_cyl + L_cvg + L_thr_straight

    # Build non-uniform z-array: guarantee at least 1 station in each zone,
    # distribute remaining stations proportionally to zone length but weight
    # throat/converging zones more heavily (higher thermal gradients there).
    # Always include zone boundary points explicitly.
    N = N_STATIONS
    zone_defs = [
        ("injector",   0,          z_inj_end, 1),
        ("chamber",    z_inj_end,  z_cyl_end, 6),
        ("converging", z_cyl_end,  z_cvg_end, 6),
        ("throat",     z_cvg_end,  z_thr_end, 3),   # always 3+ stations at throat
        ("nozzle",     z_thr_end,  L_total,   8),
    ]

    # Guaranteed minimum stations per zone
    guaranteed = sum(zd[3] for zd in zone_defs)
    remaining  = max(N - guaranteed, 0)

    # Proportional lengths (weighted: throat and converging get extra weight)
    weights    = [L_inj, L_cyl, L_cvg, max(L_thr_straight, L_total*0.02), L_noz]
    weights[2] *= 2.5   # converging — strong gradient
    weights[3] *= 5.0   # throat    — highest heat flux
    total_w    = sum(weights)
    extra_per_zone = [int(remaining * w / total_w) for w in weights]
    # Give any leftover to nozzle
    extra_per_zone[-1] += remaining - sum(extra_per_zone)

    zs_final = []
    for idx, (zname, z0, z1, n_min) in enumerate(zone_defs):
        n_total = n_min + extra_per_zone[idx]
        zone_len = z1 - z0
        if zone_len < 1e-9:
            zs_final.append((z0, zname))
            continue
        for j in range(n_total):
            frac = j / max(n_total - 1, 1)
            zs_final.append((z0 + frac * zone_len, zname))

    # Sort and deduplicate, keep exactly N stations
    zs_final.sort(key=lambda x: x[0])
    # Deduplicate by z value
    seen = set()
    zs_dedup = []
    for z_val, zname in zs_final:
        key = round(z_val * 1e9)  # nm resolution
        if key not in seen:
            seen.add(key)
            zs_dedup.append((z_val, zname))

    # If we have too many, thin uniformly; if too few, interpolate
    if len(zs_dedup) > N:
        step = len(zs_dedup) / N
        zs_dedup = [zs_dedup[int(i * step)] for i in range(N)]
    while len(zs_dedup) < N:
        zs_dedup.append(zs_dedup[-1])

    # Axial station arrays
    xs, rs, ARs, zones, rs_m = [], [], [], [], []

    for z_val, zone in zs_dedup:
        xs.append(z_val / L_total)

        if zone == "injector":
            r_m = r_c
        elif zone == "chamber":
            r_m = r_c
        elif zone == "converging":
            t_   = (z_val - z_cyl_end) / max(L_cvg, 1e-6)
            r_m  = r_c - (r_c - r_t) * clamp(t_, 0, 1)
        elif zone == "throat":
            r_m = r_t
        else:  # nozzle
            t_   = (z_val - z_thr_end) / max(L_noz, 1e-6)
            r_m  = r_t + (r_e - r_t) * clamp(t_, 0, 1)

        rs_m.append(r_m)
        rs.append(r_m / r_c)
        ARs.append((r_m / r_t) ** 2)
        zones.append(zone)

    c_star_est = c_star(prop["T_flame"] * 0.95, prop["gamma"], prop["MW"])
    tau_ms     = (Lstar / c_star_est) * 1000

    return {
        "xs": xs, "rs": rs, "ARs": ARs, "zones": zones, "rs_m": rs_m,
        "r_thr": r_t / r_c, "r_noz": r_e / r_c, "r_ch": 1.0,
        "r_t_m": r_t, "r_c_m": r_c, "r_e_m": r_e,
        "A_t_m2": A_t, "L_cyl_m": L_cyl, "L_cvg_m": L_cvg,
        "L_noz_m": L_noz, "L_total_m": L_total, "V_c_m3": V_c,
        "Lstar": Lstar, "Lstar_margin": (Lstar - prop["Lstar_min"]) / (prop["Lstar_max"] - prop["Lstar_min"]),
        "Lstar_low":  Lstar < prop["Lstar_min"],
        "Lstar_high": Lstar > prop["Lstar_max"],
        "tau_ms": tau_ms, "eps_c": eps_c, "eps_e": eps_e,
        "theta_c_deg": p["convergentAngle"], "theta_d_deg": p["nozzleAngle"],
        "L_inj_m": L_inj, "L_thr_straight_m": L_thr_straight,
        "R_thr_upstream_m":   1.5   * r_t,
        "R_thr_downstream_m": 0.382 * r_t,
        "chamberSizing": chamber_sizing,
        "chamberGoverningCriterion": chamber_sizing["governingCriterion"],
        "chamberGovernedByPhysics":  chamber_sizing["governedByPhysics"],
    }

def size_chamber(r_t_m, r_c_m, Lstar, prop, inj_key, p_c_Pa):
    """
    Multi-criterion chamber sizing.
    L_cyl = max(L_Lstar, L_vap, L_residence, L_injector_min, L_stability)
    Each criterion is bounded before max() to prevent runaway geometries.
    """
    inj     = INJECTOR_TYPES.get(inj_key, INJECTOR_TYPES["like_doublet"])
    A_t     = math.pi * r_t_m ** 2
    A_c     = math.pi * r_c_m ** 2
    cstar   = c_star(prop["T_flame"] * 0.95, prop["gamma"], prop["MW"])
    D_c     = r_c_m * 2

    # Hard sanity cap
    L_cap = min(D_c * 6, r_t_m * 80, 1.5)

    # 1. L* criterion
    L_from_Lstar = clamp((Lstar * A_t) / A_c, 0.02, L_cap)
    tau_Lstar_ms = (Lstar / cstar) * 1000

    # 2. Vaporisation (D² law) — floored jet velocity to preserve correlation validity
    v_jet = max(math.sqrt(2 * inj["dp_opt"] * p_c_Pa / max(prop["rho_fuel"], 1)), 8.0)
    prop_key_str = next((k for k, v in PROPELLANTS.items() if v is prop), "kerolox")
    SMD_um = smd_impinging(2.0, v_jet) * inj["smf"]
    vap    = vaporization_length(SMD_um, v_jet, prop_key_str)
    L_vap_raw = (vap["L_vap_mm"] / 1000) / 0.70
    L_from_vap = clamp(L_vap_raw, 0.02, min(L_from_Lstar * 2.0, L_cap))
    vap_clipped = L_vap_raw > L_from_Lstar * 2.0

    # 3. Residence time
    tau_min_ms = 1.2 if prop["OF_nominal"] > 4 else (1.8 if prop["OF_nominal"] > 3 else 2.5)
    L_res_raw  = (tau_min_ms / 1000) * cstar * A_t / A_c
    L_from_res = clamp(L_res_raw, 0.02, min(L_from_Lstar * 3.0, L_cap))

    # 4. Injector minimum
    lbl = inj.get("label", "")
    inj_fac = 3.5 if "Pintle" in lbl else (3.0 if "Swirl" in lbl else (1.8 if "Shower" in lbl else 2.5))
    L_from_inj = clamp(r_c_m * inj_fac, 0.01, L_cap * 0.5)

    # 5. Stability floor
    L_from_stab = clamp(r_c_m * 2.5, 0.01, L_cap * 0.4)

    criteria = {
        "Lstar":        L_from_Lstar,
        "vaporization": L_from_vap,
        "residence":    L_from_res,
        "injector_min": L_from_inj,
        "stability":    L_from_stab,
    }
    L_cyl_final = max(criteria.values())
    governing   = max(criteria, key=criteria.get)
    LD_ratio    = L_cyl_final / D_c

    return {
        "L_cyl_m":            L_cyl_final,
        "governingCriterion": governing,
        "criteria_m":         criteria,
        "tau_Lstar_ms":       tau_Lstar_ms,
        "tau_min_ms":         tau_min_ms,
        "SMD_um":             SMD_um,
        "L_vap_mm":           vap["L_vap_mm"],
        "vapClipped":         vap_clipped,
        "LD_ratio":           LD_ratio,
        "LD_warn":            LD_ratio > 4.5,
        "L_cap_m":            L_cap,
        "governedByPhysics":  governing != "Lstar",
    }

def size_nozzle(r_t, eps_e, theta_d_deg, p_c_Pa, p_amb_Pa, gamma, max_noz_len_m=None):
    """Conical nozzle design with contour points and packaging check."""
    theta_d = clamp(theta_d_deg, 8, 45) * math.pi / 180
    r_e     = r_t * math.sqrt(clamp(eps_e, 1.5, 120))
    R_dn    = 0.382 * r_t
    L_conical = (r_e - r_t - R_dn) / math.tan(theta_d) + R_dn
    L_noz   = max(L_conical, r_t * 1.5)

    pkg_status   = "ok"
    eps_e_final  = eps_e
    r_e_final    = r_e
    L_noz_final  = L_noz

    if max_noz_len_m and L_noz > max_noz_len_m:
        pkg_status  = "clipped"
        r_e_clip    = r_t + R_dn + (max_noz_len_m - R_dn) * math.tan(theta_d)
        r_e_final   = max(r_e_clip, r_t * 1.1)
        eps_e_final = (r_e_final / r_t) ** 2
        L_noz_final = max_noz_len_m

    # Contour points
    N_FILLET  = 8
    N_CONTOUR = 30
    pts = []
    for i in range(N_FILLET + 1):
        ang   = (math.pi / 2) * (i / N_FILLET)
        z_pt  = R_dn * math.sin(ang)
        r_pt  = r_t  + R_dn * (1 - math.cos(ang))
        pts.append({"z_mm": z_pt*1000, "r_mm": r_pt*1000, "slope_deg": math.degrees(ang)})

    z0 = R_dn
    r0 = r_t + R_dn * (1 - math.cos(math.pi / 2))
    for i in range(1, N_CONTOUR + 1):
        t_   = i / N_CONTOUR
        z_pt = z0 + t_ * (L_noz_final - R_dn)
        r_pt = r0 + (r_e_final - r0) * t_
        pts.append({"z_mm": z_pt*1000, "r_mm": r_pt*1000, "slope_deg": theta_d_deg})

    pe_pc = exit_pressure_ratio(gamma, eps_e_final)
    p_e   = pe_pc * p_c_Pa

    return {
        "mode":                   "conical",
        "throatRadius_m":         r_t,
        "exitRadius_m":           r_e_final,
        "areaRatio":              eps_e_final,
        "divergenceHalfAngle_deg": theta_d_deg,
        "throatBlendUpstream_m":  1.5   * r_t,
        "throatBlendDownstream_m": R_dn,
        "nozzleLength_m":         L_noz_final,
        "packageStatus":          pkg_status,
        "areaRatioClipped":       pkg_status == "clipped",
        "contourPoints_mm":       pts,
        "performanceBasis": {
            "p_c_Pa":        p_c_Pa,
            "p_amb_Pa":      p_amb_Pa,
            "p_e_Pa":        p_e,
            "isOptimal":     abs(p_e - p_amb_Pa) / max(p_amb_Pa, 1) < 0.05,
            "isOverExpanded": p_e < p_amb_Pa,
        },
    }

# ── Channel geometry ───────────────────────────────────────────────────────────

def size_channel_section(r_wall_m, t_total_m, n_channels, prop_key, zone):
    """Explicit channel cross-section geometry from circumference/pitch."""
    zone_factors = {
        "injector":   {"wFrac": 0.58, "dFrac": 0.42, "innerLigFrac": 0.22, "outerLigFrac": 0.16},
        "chamber":    {"wFrac": 0.60, "dFrac": 0.45, "innerLigFrac": 0.22, "outerLigFrac": 0.15},
        "converging": {"wFrac": 0.58, "dFrac": 0.48, "innerLigFrac": 0.20, "outerLigFrac": 0.14},
        "throat":     {"wFrac": 0.55, "dFrac": 0.50, "innerLigFrac": 0.20, "outerLigFrac": 0.13},
        "nozzle":     {"wFrac": 0.52, "dFrac": 0.40, "innerLigFrac": 0.23, "outerLigFrac": 0.17},
    }
    f = zone_factors.get(zone, zone_factors["chamber"])

    circ    = 2 * math.pi * r_wall_m
    pitch_m = circ / max(n_channels, 1)
    ch_w    = pitch_m * f["wFrac"]
    ch_d    = t_total_m * f["dFrac"]
    rib     = pitch_m * (1 - f["wFrac"])
    i_lig   = t_total_m * f["innerLigFrac"]
    o_lig   = t_total_m * f["outerLigFrac"]

    A_flow  = ch_w * ch_d
    P_wet   = 2 * (ch_w + ch_d)
    D_h     = 4 * A_flow / max(P_wet, 1e-9)

    centroid_r = r_wall_m + i_lig + ch_d / 2
    min_lig    = min(i_lig, rib * 0.5)
    lig_thin   = min_lig < ch_d * 0.20

    return {
        "zone": zone, "nChannels": n_channels,
        "channelWidth_mm":    ch_w  * 1000,
        "channelDepth_mm":    ch_d  * 1000,
        "ribThickness_mm":    rib   * 1000,
        "innerLigament_mm":   i_lig * 1000,
        "outerLigament_mm":   o_lig * 1000,
        "pitch_mm":           pitch_m * 1000,
        "hydraulicDiameter_mm": D_h * 1000,
        "flowArea_mm2":       A_flow * 1e6,
        "wettedPerimeter_mm": P_wet  * 1000,
        "centroidRadius_mm":  centroid_r * 1000,
        "minLigament_mm":     min_lig * 1000,
        "ligamentThin":       lig_thin,
        "D_h_m":              D_h,
    }

def compute_channel_sections(p, geom, prop_key):
    """Channel sections for all cooled zones + real h_c from Dittus-Boelter."""
    if not p.get("regenerativeCooling"):
        return {"sections": [], "h_c_regen": 200}

    prop   = PROPELLANTS[prop_key]
    n_ch   = max(4, int(p.get("coolingChannels", 16)))
    t_total= wall_thickness_m(p, geom["r_t_m"])

    zone_radii = {
        "injector":   geom["r_c_m"],
        "chamber":    geom["r_c_m"],
        "converging": (geom["r_c_m"] + geom["r_t_m"]) / 2,
        "throat":     geom["r_t_m"] * 1.05,
    }
    sections = [size_channel_section(zone_radii[z], t_total, n_ch, prop_key, z)
                for z in ["injector", "chamber", "converging", "throat"]]

    throat_sec = next((s for s in sections if s["zone"] == "throat"), sections[0])
    D_h    = throat_sec["D_h_m"]
    A_flow = throat_sec["flowArea_mm2"] * 1e-6  # m²

    # Real coolant velocity from fuel mass flow / n_ch / A_flow_per_channel
    p_c_Pa  = p["chamberPressure"] * 1e6
    cstar   = c_star(prop["T_flame"] * 0.95, prop["gamma"], prop["MW"])
    mdot_tot= p_c_Pa * geom["A_t_m2"] / max(cstar, 1)
    mdot_f  = mdot_tot / (1 + prop["OF_nominal"])
    v_cool  = clamp(mdot_f / (n_ch * prop["rho_fuel"] * max(A_flow, 1e-12)), 1, 80)

    Re     = clamp(prop["rho_fuel"] * v_cool * D_h / max(prop["mu_cool"], 1e-6), 3000, 1e6)
    Nu     = 0.023 * Re**0.8 * prop["Pr_cool"]**0.4
    h_c_re = Nu * prop["k_cool"] / D_h

    return {"sections": sections, "h_c_regen": h_c_re, "Re_throat": Re,
            "v_cool_ms": v_cool}

# ── Thermal ────────────────────────────────────────────────────────────────────

def bartz_hg(p_c, D_throat, r_ratio):
    """
    Bartz (1957) gas-side h_g proxy [W/m^2K].
    Scale 0.5e5 calibrated for concept-level screening:
    - h_g_throat ≈ 8,600 W/m²K at 5 MPa, 50 mm throat (16 kN class)
    - LH₂ and LCH₄ regen cooling can keep copper below T_limit=773 K
    - RP-1 regen alone is marginal; film cooling is needed (correct engineering)
    - Matches reduced-order Bartz proxy tables (Sutton 9th ed §8).
    p_c in MPa, D_throat in m, r_ratio = r(x)/r_t (1.0 at throat).
    """
    return 0.026 * (p_c ** 0.8) / (D_throat ** 0.2) * (r_ratio ** -0.9) * 0.5e5

def adiabatic_wall_temp(T_flame, M, gamma=1.25, Pr=0.72):
    """
    Adiabatic wall (recovery) temperature.
    T_flame is the chamber stagnation temperature.
    T_aw = T_flame * (1 + r_f*(γ-1)/2*M²) / (1 + (γ-1)/2*M²)
    Correctly approaches T_flame at M=0 and falls toward T_flame*r_f at high M.
    r_f = Pr^(1/3) for turbulent boundary layer (Eckert, 1955).
    """
    r_f  = Pr ** (1.0 / 3.0)
    gm1h = (gamma - 1) / 2.0
    return T_flame * (1.0 + r_f * gm1h * M * M) / max(1.0 + gm1h * M * M, 1e-9)

def coolant_side_h(p, prop_key, geom=None):
    """
    Coolant-side heat transfer coefficient via Dittus-Boelter.
    Uses actual channel geometry (D_h, flow area) and real coolant velocity
    derived from fuel mass flow split across all channels.
    """
    prop = PROPELLANTS[prop_key]
    if not p.get("regenerativeCooling"):
        return 200  # bare radiation + conduction fallback

    if geom:
        ch_data = compute_channel_sections(p, geom, prop_key)
        return ch_data["h_c_regen"]

    # Fallback when no geom: estimate from channel count and throat radius proxy
    n_ch   = max(4, int(p.get("coolingChannels", 16)))
    r_t    = p.get("throatRadius", 0.025)
    circ   = 2 * math.pi * r_t * 1.05  # throat zone radius
    pitch  = circ / n_ch
    ch_w   = pitch * 0.55
    ch_d   = wall_thickness_m(p, p.get("throatRadius",0.025)) * 0.50
    A_flow = ch_w * ch_d
    P_wet  = 2 * (ch_w + ch_d)
    D_h    = 4 * A_flow / max(P_wet, 1e-9)

    # Estimate coolant velocity from fuel mass flow / n_ch / A_flow
    p_c_Pa  = p["chamberPressure"] * 1e6
    cstar   = c_star(prop["T_flame"] * 0.95, prop["gamma"], prop["MW"])
    A_t     = math.pi * r_t ** 2
    mdot_tot= p_c_Pa * A_t / max(cstar, 1)
    mdot_f  = mdot_tot / (1 + prop["OF_nominal"])
    v_cool  = clamp(mdot_f / (n_ch * prop["rho_fuel"] * max(A_flow, 1e-9)), 1, 80)

    Re  = clamp(prop["rho_fuel"] * v_cool * D_h / max(prop["mu_cool"], 1e-6), 3000, 1e6)
    Nu  = 0.023 * Re**0.8 * prop["Pr_cool"]**0.4
    h_c = Nu * prop["k_cool"] / D_h
    return h_c

def build_heat_flux_profile(p, geom, Ms, material, prop_key, inj_key, h_c_ext=None):
    """
    Gas-side heat flux with self-consistent wall temperature.
    Solves 3-resistance network: T_wg = (h_g·T_aw_eff + T_cool/R_total)/(h_g + 1/R_total)
    Film cooling reduces the effective adiabatic wall temperature (Huzel & Huang §4-3).
    """
    prop = PROPELLANTS[prop_key]
    inj  = INJECTOR_TYPES[inj_key]
    mat  = MATERIALS[material]

    T_flame = prop["T_flame"] * inj["mix_eff"]
    p_c     = p["chamberPressure"]
    D_thr   = geom["r_t_m"] * 2
    r_t_m   = geom["r_t_m"]

    t_wall  = wall_thickness_m(p, r_t_m)
    R_wall  = t_wall / mat["k"]
    h_c     = h_c_ext or 200
    T_cool  = prop.get("T_coolant", 300)
    T_film  = prop.get("T_fuel", T_cool)  # film coolant = fuel injection temperature
    R_total = R_wall + 1 / max(h_c, 1)

    # Film cooling effectiveness: Huzel & Huang exponential decay
    # Referenced to actual distance from injector face in metres
    # eta_f = eta_0 * exp(-x_from_inj / L_decay)
    # L_decay ~ 1.5 * L_cyl (covers chamber cylinder + part of convergent)
    # eta_0 = 0.55 (initial effectiveness after injector)
    # This ensures meaningful eta (~20-40%) still present at the throat
    L_chamber = geom.get("L_cyl_m", 0.3) + geom.get("L_cvg_m", 0.05)
    L_total   = geom.get("L_total_m", 0.8)
    L_inj     = geom.get("L_inj_m", 0.005)
    if p.get("filmCooling"):
        lam_physical = L_chamber * (0.8 + p.get("injectorDensity", 0.5) * 0.4)
        eta_film_arr = []
        for x_frac in geom["xs"]:
            x_phys = x_frac * L_total - L_inj   # distance from injector face [m]
            if x_phys <= 0:
                eta_film_arr.append(0.0)
            else:
                eta_film_arr.append(0.55 * math.exp(-x_phys / max(lam_physical, 1e-3)))
    else:
        eta_film_arr = [0.0] * N_STATIONS

    q_g_arr, T_aw_arr, h_g_arr, T_wg_arr = [], [], [], []

    for i in range(N_STATIONS):
        M = Ms[i]
        r_ratio = clamp(geom["rs_m"][i] / max(r_t_m, 1e-9), 0.5, 4.0)
        h_g     = bartz_hg(p_c, D_thr, r_ratio)
        T_aw    = adiabatic_wall_temp(T_flame, M, prop["gamma"], prop["Pr"])
        # Film cooling: reduce effective T_aw toward coolant injection temperature
        T_aw_eff = T_aw - eta_film_arr[i] * (T_aw - T_film)
        T_wg    = (h_g * T_aw_eff + T_cool / R_total) / (h_g + 1.0 / R_total)
        q       = max(0, h_g * (T_aw_eff - T_wg))
        q_g_arr.append(q)
        T_aw_arr.append(T_aw)
        h_g_arr.append(h_g)
        T_wg_arr.append(T_wg)

    return {"q_g": q_g_arr, "T_aw": T_aw_arr, "h_g": h_g_arr,
            "T_wg": T_wg_arr, "T_flame": T_flame,
            "eta_film": eta_film_arr}

def eval_thermal_full(p, geom, material, prop_key, inj_key):
    mat  = MATERIALS[material]
    prop = PROPELLANTS[prop_key]

    Ms   = [mach_from_area_ratio(AR, supersonic=(z == "nozzle"), gamma=prop["gamma"])
            for AR, z in zip(geom["ARs"], geom["zones"])]

    h_c      = coolant_side_h(p, prop_key, geom)
    flux     = build_heat_flux_profile(p, geom, Ms, material, prop_key, inj_key, h_c)
    q_g        = flux["q_g"]
    T_aw       = flux["T_aw"]
    h_g        = flux["h_g"]
    T_wall_gas = flux["T_wg"]
    T_flame    = flux["T_flame"]
    eta_film   = flux["eta_film"]   # already computed inside build_heat_flux_profile

    # Ablative layer (reduces net heat load through charring)
    if p.get("ablativeLayer"):
        h_vap    = 2e6
        charRate = [clamp(q / h_vap, 0, 1) for q in q_g]
        eta_abl  = [clamp(cr * 0.8, 0, 0.7) for cr in charRate]
    else:
        charRate = [0.0] * N_STATIONS
        eta_abl  = [0.0] * N_STATIONS

    # q_net: net heat flux reaching the structural wall (after film + ablative)
    q_net = [q_g[i] * (1 - eta_film[i]) * (1 - eta_abl[i]) for i in range(N_STATIONS)]

    T_coolant   = prop.get("T_coolant", 300)
    T_wall_cool = [T_coolant + q / max(h_c, 1) for q in q_net]

    throat_idx   = next((i for i, z in enumerate(geom["zones"]) if z == "throat"),
                        int(N_STATIONS * 0.55))
    T_wall_max   = max(T_wall_gas)
    T_throat_wall= T_wall_gas[throat_idx]
    throat_margin  = clamp(1 - T_throat_wall / mat["T_limit"], -0.5, 1)
    overall_margin = clamp(1 - T_wall_max   / mat["T_limit"], -0.5, 1)
    limit_exceeded = T_wall_max > mat["T_limit"]

    # Channel pressure drop — use real velocity from channel sections
    dP_norm = 0.0
    if p.get("regenerativeCooling") and geom:
        ch_data = compute_channel_sections(p, geom, prop_key)
        secs    = ch_data["sections"]
        v_cool  = ch_data.get("v_cool_ms", 10)
        thr_s   = next((s for s in secs if s["zone"] == "throat"), None)
        if thr_s:
            D_h     = thr_s["D_h_m"]
            Re      = clamp(prop["rho_fuel"] * v_cool * D_h / max(prop["mu_cool"], 1e-6), 3000, 1e6)
            f_bl    = 0.316 * Re**(-0.25)
            L_D     = geom["L_total_m"] / max(D_h, 1e-6)
            dP_norm = clamp(f_bl * L_D * prop["rho_fuel"] * v_cool**2 * 0.5 / 1e6, 0, 1)

    margin_score = clamp(overall_margin, 0, 1)
    cooling_eff  = clamp(1 - T_wall_max / (mat["T_limit"] * 1.5), 0, 1)
    score        = clamp(margin_score * 0.55 + cooling_eff * 0.35 - dP_norm * 0.15, 0, 1)

    return {
        "score": score,
        "xs": geom["xs"], "q_net": q_net, "q_g": q_g,
        "T_aw": T_aw, "T_wall_gas": T_wall_gas, "T_wall_coolant": T_wall_cool,
        "eta_film": eta_film, "charRate": charRate, "Ms": Ms, "h_g": h_g, "h_c": h_c,
        "T_flame": T_flame, "T_wall_max": T_wall_max, "T_throat_wall": T_throat_wall,
        "throatMargin": throat_margin, "overallMargin": overall_margin,
        "proxyThermalMargin": overall_margin,
        "limitExceeded": limit_exceeded,
        "dP_norm": dP_norm, "normalizedPressureDropPenalty": dP_norm,
        "coolingEfficiency": cooling_eff,
        "heatLoad": clamp(T_wall_max / mat["T_limit"], 0, 2),
    }

def solve_wall_configuration(p_in, geom, material, prop_key, inj_key):
    """
    Iterative thermal-structural wall coupling (up to 8 iterations).

    Physics of the adjustment strategy:
    - Thermal fail (T_wall > T_limit): DECREASE wallThickness
        → thinner wall = smaller channel depth = higher coolant velocity
        → higher Re → higher h_c → lower T_wg  ✓
    - Structural fail (SF < 1.5): INCREASE wallThickness
        → thicker wall = more metal in load path → higher SF  ✓
    - Both fail simultaneously: increase coolingChannels instead
        → more channels = finer pitch = smaller D_h = higher h_c
          without reducing structural section  ✓
    """
    mat      = MATERIALS[material]
    MAX_ITER = 8
    p        = dict(p_in)
    # Ensure regen is on (required for thermal convergence in most cases)
    if not p.get("regenerativeCooling"):
        p["regenerativeCooling"] = True
    thermal = structural = None
    converged = False
    T_prev = float('inf')
    S_prev = float('inf')

    for it in range(MAX_ITER):
        thermal    = eval_thermal_full(p, geom, material, prop_key, inj_key)
        structural = eval_structural(p, geom, thermal, prop_key)

        T_now = thermal["T_wall_max"]
        S_now = structural.get("sigma_peak_MPa") or 0
        dT    = abs(T_now - T_prev)
        dS    = abs(S_now - S_prev)
        T_prev = T_now;  S_prev = S_now

        t_fail = T_now > mat["T_limit"]
        s_fail = (structural.get("SF_actual") or 2) < 1.5

        if dT < 8 and dS < 3:
            converged = True
            break

        wt  = p["wallThickness"]
        n_ch = p.get("coolingChannels", 20)

        if t_fail and s_fail:
            # Both fail: add channels (improves cooling without weakening wall)
            p = {**p, "coolingChannels": min(n_ch + 8, 80)}
        elif t_fail:
            # Thermal only: thin the wall to increase coolant velocity
            p = {**p, "wallThickness": clamp(wt * 0.88, 0.08, 0.95)}
        elif s_fail:
            # Structural only: thicken the wall
            p = {**p, "wallThickness": clamp(wt * 1.10, 0.08, 0.95)}

    if thermal is None:
        thermal    = eval_thermal_full(p, geom, material, prop_key, inj_key)
        structural = eval_structural(p, geom, thermal, prop_key)

    return {"params_converged": p, "thermal": thermal, "structural": structural,
            "wall_iterations": MAX_ITER, "wall_converged": converged}

# ── Structural ─────────────────────────────────────────────────────────────────

def multi_layer_wall_stress(q_net_val, t_inner_m, t_insul_m, t_shell_m,
                             mat_inner, mat_insul, mat_shell, T_coolant):
    li = WALL_LAYERS.get(mat_inner, WALL_LAYERS["copper_inner"])
    ls = WALL_LAYERS.get(mat_insul, WALL_LAYERS["carbon_phenolic"])
    lo = WALL_LAYERS.get(mat_shell, WALL_LAYERS["steel_shell"])

    R_inner = t_inner_m / max(li["k"], 1e-9)
    R_insul = t_insul_m / max(ls["k"], 1e-9)
    R_shell = t_shell_m / max(lo["k"], 1e-9)
    R_total = R_inner + R_insul + R_shell

    dT_inner = q_net_val * R_inner
    dT_insul = q_net_val * R_insul
    dT_shell = q_net_val * R_shell

    T_shell_inner  = T_coolant + q_net_val * R_shell * 0.5
    T_insul_inner  = T_shell_inner + dT_insul
    T_inner_inner  = T_insul_inner + dT_inner

    def stress(mat, dT):
        return abs(mat["E"] * 1e9 * mat["alpha"] * dT / (1 - mat["nu"])) / 1e6

    return {
        "sigma_inner_MPa": stress(li, dT_inner),
        "sigma_insul_MPa": stress(ls, dT_insul),
        "sigma_shell_MPa": stress(lo, dT_shell),
        "T_inner_inner": T_inner_inner,
        "T_insul_inner": T_insul_inner,
        "T_shell_inner": T_shell_inner,
        "dT_inner": dT_inner, "dT_insul": dT_insul,
    }

def eval_structural(p, geom, thermal_result, prop_key):
    mat   = MATERIALS.get(p.get("material", "copper"), MATERIALS["copper"])
    prop  = PROPELLANTS.get(prop_key, PROPELLANTS["kerolox"])
    mat_key = p.get("material", "copper")

    t_total  = wall_thickness_m(p, geom["r_t_m"])
    t_inner  = t_total * 0.20
    t_insul  = t_total * 0.55
    t_shell  = t_total * 0.25

    mat_inner = "graphite" if mat_key == "inconel" else "copper_inner"
    mat_insul = "ceramic_ablative" if p.get("ablativeLayer") else "carbon_phenolic"
    mat_shell = "steel_shell" if mat_key == "steel" else "inconel_shell"
    T_coolant = prop.get("T_coolant", 300)

    sigma_inner_arr, sigma_insul_arr, sigma_shell_arr, T_gaswall_arr = [], [], [], []
    for q in thermal_result["q_net"]:
        layers = multi_layer_wall_stress(q, t_inner, t_insul, t_shell,
                                         mat_inner, mat_insul, mat_shell, T_coolant)
        sigma_inner_arr.append(layers["sigma_inner_MPa"])
        sigma_insul_arr.append(layers["sigma_insul_MPa"])
        sigma_shell_arr.append(layers["sigma_shell_MPa"])
        T_gaswall_arr.append(layers["T_inner_inner"])

    throat_i   = next((i for i, z in enumerate(geom["zones"]) if z == "throat"),
                       int(N_STATIONS * 0.55))
    q_throat   = thermal_result["q_net"][throat_i]
    thr_layers = multi_layer_wall_stress(q_throat, t_inner, t_insul, t_shell,
                                          mat_inner, mat_insul, mat_shell, T_coolant)

    K_t = STRESS_CONC["throat_fillet"] if p.get("regenerativeCooling") else STRESS_CONC["throat_sharp"]
    sigma_throat_MPa = thr_layers["sigma_inner_MPa"] * K_t
    sigma_peak_MPa   = max(sigma_inner_arr) * K_t

    sigma_y_base = mat["sigma_y_base"]
    T_frac       = clamp(thr_layers["T_inner_inner"] / mat["T_limit"], 0, 1.2)
    sigma_y      = sigma_y_base * (1 - 0.35 * T_frac)
    SF_actual    = sigma_y / max(sigma_peak_MPa, 1)
    SF_ok        = SF_actual >= 1.5

    dT_throat  = thr_layers["dT_inner"] + thr_layers["dT_insul"]
    grad_score = clamp(1 - dT_throat / 1200, 0, 1)
    ar_penalty = max((p.get("contractionRatio", 4) - 6) * 0.06, 0)

    # Hollow-wall section penalties
    hollow_wall_penalty = 0.0
    min_ligament_mm     = t_total * 1000
    ligament_thin       = False
    rib_slenderness     = 0.0
    channel_root_Kt     = 1.0
    net_section_frac    = 1.0
    hollow_wall_zones   = {}
    vuln_map            = [0.0] * N_STATIONS

    if p.get("regenerativeCooling"):
        secs = compute_channel_sections(p, geom, prop_key)["sections"]
        zone_penalties = {}
        for sec in secs:
            w  = sec["channelWidth_mm"];  d = sec["channelDepth_mm"]
            r  = sec["ribThickness_mm"];  pt = sec["pitch_mm"]
            li = sec["innerLigament_mm"]

            rib_frac   = r / max(pt, 1e-9)
            depth_frac = d / max(t_total * 1000, 1)
            net_frac   = clamp(rib_frac * (1 - depth_frac) + (1 - depth_frac), 0.3, 1.0)
            ampA       = clamp(1 / max(net_frac, 0.3), 1.0, 2.5)

            slenderness = d / max(r, 0.1)
            ampB        = clamp(1 + max(slenderness - 5, 0) * 0.07, 1.0, 2.0)

            r_corner    = max(r * 0.10, 0.05)
            Kt_root     = clamp(1 + 0.5 * math.sqrt(d / r_corner), 1.0, 3.0)
            zone_pen    = clamp((ampA * ampB * Kt_root - 1) * depth_frac * 0.20, 0, 0.45)

            zone_penalties[sec["zone"]] = {
                "netFrac": net_frac, "ampA": ampA,
                "slenderness": slenderness, "ampB": ampB,
                "Kt_root": Kt_root, "zonePenalty": zone_pen,
                "minLigament_mm": sec["minLigament_mm"],
                "ligamentThin":   sec["ligamentThin"],
            }

        thr_zone = (zone_penalties.get("throat")
                    or zone_penalties.get("converging")
                    or (next(iter(zone_penalties.values())) if zone_penalties else None))
        if thr_zone:
            min_ligament_mm    = thr_zone["minLigament_mm"]
            ligament_thin      = thr_zone["ligamentThin"]
            net_section_frac   = thr_zone["netFrac"]
            rib_slenderness    = thr_zone["slenderness"]
            channel_root_Kt    = thr_zone["Kt_root"]
            hollow_wall_penalty = thr_zone["zonePenalty"]

        hollow_wall_zones = zone_penalties

        def zone_to_sec(zone):
            return (zone_penalties.get(zone)
                    or zone_penalties.get("chamber")
                    or (next(iter(zone_penalties.values())) if zone_penalties else None))

        for i in range(N_STATIONS):
            sec = zone_to_sec(geom["zones"][i])
            if not sec:
                continue
            q_norm  = clamp((thermal_result["q_net"][i]) / 2e7, 0, 1)
            s_norm  = clamp((sigma_inner_arr[i]) / max(sigma_y, 1), 0, 1)
            p_norm  = sec["zonePenalty"] / 0.45
            lig_frac = clamp(1 - sec["minLigament_mm"] / max(t_total * 1000, 1), 0, 1)
            vuln_map[i] = clamp(q_norm*0.35 + s_norm*0.30 + p_norm*0.25 + lig_frac*0.10, 0, 1)

    stress_margin = clamp((SF_actual - 1.5) / 1.5, -1, 1)
    score = clamp(stress_margin*0.40 + grad_score*0.35 + (1-ar_penalty)*0.25 - hollow_wall_penalty, 0, 1)

    return {
        "score": score,
        "sigma_inner_arr": sigma_inner_arr, "sigma_insul_arr": sigma_insul_arr,
        "sigma_shell_arr": sigma_shell_arr, "T_gaswall_arr": T_gaswall_arr,
        "sigma_throat_MPa": sigma_throat_MPa, "sigma_peak_MPa": sigma_peak_MPa,
        "sigma_y": sigma_y, "SF_actual": SF_actual, "SF_ok": SF_ok, "K_t": K_t,
        "T_throat_wall": thr_layers["T_inner_inner"],
        "T_insul_inner":  thr_layers["T_insul_inner"],
        "T_shell_inner":  thr_layers["T_shell_inner"],
        "dT_throat": dT_throat,
        "t_inner_mm": t_inner*1000, "t_insul_mm": t_insul*1000, "t_shell_mm": t_shell*1000,
        "hollowWallPenalty": hollow_wall_penalty,
        "minLigament_mm": min_ligament_mm, "ligamentThin": ligament_thin,
        "ribSlenderness": rib_slenderness, "channelRootKt": channel_root_Kt,
        "netSectionFraction": net_section_frac,
        "hollowWallZones": hollow_wall_zones,
        "vulnerabilityMap": vuln_map,
        "maxVulnerability": max(vuln_map),
        "vulnerabilityPeakIdx": vuln_map.index(max(vuln_map)),
        "mat_inner": mat_inner, "mat_insul": mat_insul, "mat_shell": mat_shell,
    }

# ── Flow / injector evaluator ─────────────────────────────────────────────────

def eval_flow(p, geom, inj_key, prop_key, governing_state=None):
    inj   = INJECTOR_TYPES[inj_key]
    prop  = PROPELLANTS[prop_key]
    p_c_Pa = p["chamberPressure"] * 1e6
    r_c_m  = geom["r_c_m"]
    r_t_m  = geom["r_t_m"]
    A_c_m2 = math.pi * r_c_m ** 2

    n_elem = max(4, int(p.get("injectorDensity", 0.5) * 500 * A_c_m2 + 8))

    if governing_state:
        mdot_f  = governing_state["mdot_fuel"]
        mdot_ox = governing_state["mdot_ox"]
    else:
        mdot_prx = p_c_Pa * math.pi * r_t_m**2 / max(c_star(prop["T_flame"]*0.95, prop["gamma"], prop["MW"]), 1)
        mdot_f   = mdot_prx / (1 + prop["OF_nominal"])
        mdot_ox  = mdot_prx - mdot_f

    dP_frac  = inj["dp_opt"]
    dP_Pa    = dP_frac * p_c_Pa
    stiff_ok = dP_frac >= inj["dp_min"]

    fuel_s = size_orifice(mdot_f,  n_elem, inj["C_d"], prop["rho_fuel"], dP_Pa)
    ox_s   = size_orifice(mdot_ox, n_elem, inj["C_d"], prop["rho_ox"],   dP_Pa)

    v_fuel = orifice_velocity(mdot_f,  prop["rho_fuel"], fuel_s["A_total"])
    v_ox   = orifice_velocity(mdot_ox, prop["rho_ox"],   ox_s["A_total"])

    MR     = momentum_ratio(mdot_ox, v_ox, mdot_f, v_fuel)
    MR_ok  = 1.0 <= MR <= 3.0

    theta_res = 0.0
    if inj["impinges"] and inj["theta_inj"] > 0:
        theta_res = impingement_resultant(mdot_f, v_fuel, inj["theta_inj"],
                                           mdot_ox, v_ox, inj["theta_inj"])

    SMD_um = smd_impinging((fuel_s["d_mm"] + ox_s["d_mm"]) / 2, max(v_fuel, 1)) * inj["smf"]
    vap    = vaporization_length(SMD_um, max(v_fuel, 1), prop_key)
    L_vap_mm = vap["L_vap_mm"]
    t_d_ms   = vap["t_d_ms"]

    L_cyl_mm = geom["L_cyl_m"] * 1000
    face_too_hot = L_vap_mm < L_cyl_mm * 0.15

    lstar_low  = geom["Lstar_low"]
    lstar_high = geom["Lstar_high"]

    dp_score  = clamp(1 - abs(dP_frac - 0.20) / 0.12, 0, 1)
    mix_score = inj["mix_eff"]
    stab_sc   = inj["stability"]
    score = clamp(
        mix_score * 0.35 + stab_sc * 0.25 + dp_score * 0.20
        + (0.10 if stiff_ok else 0) + (0.10 if MR_ok else 0),
        0, 1
    )

    return {
        "score": score, "n_elem": n_elem,
        "d_fuel_mm": fuel_s["d_mm"], "d_ox_mm": ox_s["d_mm"],
        "d_elem_mm": (fuel_s["d_mm"] + ox_s["d_mm"]) / 2,
        "v_fuel": v_fuel, "v_ox": v_ox,
        "SMD_um": SMD_um, "L_vap_mm": L_vap_mm, "t_d_ms": t_d_ms,
        "MR": MR, "MR_ok": MR_ok, "theta_res": theta_res,
        "dp_frac": dP_frac, "stiffOK": stiff_ok,
        "faceTooHot": face_too_hot,
        "Lstar_low": lstar_low, "Lstar_high": lstar_high,
        "uniformityIndex": mix_score,
    }

# ── Governing state ────────────────────────────────────────────────────────────

def compute_governing_state(p, geom, prop_key, p_amb_Pa=0):
    prop           = PROPELLANTS[prop_key]
    throat_radius  = geom["r_t_m"]
    throat_area    = math.pi * throat_radius ** 2
    p_c_Pa         = p["chamberPressure"] * 1e6
    OF             = prop["OF_nominal"]
    T_c            = prop["T_flame"] * 0.95
    cstar          = c_star(T_c, prop["gamma"], prop["MW"])
    mdot_total     = p_c_Pa * throat_area / max(cstar, 1)
    mdot_fuel      = mdot_total / (1 + OF)
    mdot_ox        = mdot_total - mdot_fuel
    nozzle_AR      = geom["eps_e"]
    CF_vac         = thrust_coeff(prop["gamma"], nozzle_AR, p_c_Pa, 0)
    CF_amb         = thrust_coeff(prop["gamma"], nozzle_AR, p_c_Pa, p_amb_Pa)
    thrust_vac_kN  = CF_vac * p_c_Pa * throat_area / 1000
    thrust_amb_kN  = CF_amb * p_c_Pa * throat_area / 1000
    Isp_vac_s      = prop["Isp_vac"]
    Isp_amb_s      = isp_at_altitude(Isp_vac_s, CF_amb, CF_vac)
    return {
        "mode": "geometry_driven",
        "p_c_Pa": p_c_Pa, "p_c_MPa": p_c_Pa / 1e6, "p_amb_Pa": p_amb_Pa,
        "throat_area_m2": throat_area, "throat_radius_m": throat_radius,
        "mdot_total": mdot_total, "mdot_fuel": mdot_fuel, "mdot_ox": mdot_ox,
        "OF": OF, "c_star": cstar, "gamma": prop["gamma"],
        "nozzle_AR": nozzle_AR, "CF_vac": CF_vac, "CF_amb": CF_amb,
        "thrust_vac_kN": thrust_vac_kN, "thrust_amb_kN": thrust_amb_kN,
        "isp_vac": Isp_vac_s, "isp_amb": Isp_amb_s,
    }

# ── Pressure limit framework ───────────────────────────────────────────────────

def compute_pressure_limit(p, geom, material, prop_key):
    mat   = MATERIALS[material]
    prop  = PROPELLANTS[prop_key]
    sigma_y_base = mat["sigma_y_base"]
    T_wall_est   = (prop["T_flame"] * 0.55) * (p["chamberPressure"] / 10)
    T_frac       = clamp(T_wall_est / mat["T_limit"], 0, 1.2)
    sigma_y      = sigma_y_base * (1 - 0.35 * T_frac)
    sigma_a      = sigma_y / 2.5
    t_total_m    = wall_thickness_m(p, geom["r_t_m"])
    r_inner      = geom["r_t_m"]

    P_max_solid = sigma_a * (t_total_m * 1000) / (r_inner * 1000 + 0.5 * t_total_m * 1000)
    P_max_ch    = P_max_solid
    P_max_lig   = P_max_solid
    net_sec_frac = 1.0
    inner_lig_mm = t_total_m * 1000

    if p.get("regenerativeCooling"):
        secs = compute_channel_sections(p, geom, prop_key)["sections"]
        thr_s = next((s for s in secs if s["zone"] == "throat"), None)
        if thr_s:
            rib_frac     = thr_s["ribThickness_mm"] / max(thr_s["pitch_mm"], 1)
            net_sec_frac = clamp(rib_frac, 0.25, 1.0)
            P_max_ch     = P_max_solid * net_sec_frac
            inner_lig_mm = thr_s["innerLigament_mm"]
            P_max_lig    = sigma_a * inner_lig_mm / (r_inner * 1000 + 0.5 * inner_lig_mm)

    P_max       = min(P_max_solid, P_max_ch, P_max_lig)
    P_target    = p["chamberPressure"]
    p_margin    = (P_max - P_target) / max(P_max, 0.1)
    p_fail      = P_target > P_max
    p_warn      = (not p_fail) and p_margin < 0.20

    return {
        "P_max_MPa":          round(P_max, 3),
        "P_max_solid_MPa":    round(P_max_solid, 3),
        "P_max_channel_MPa":  round(P_max_ch, 3),
        "P_max_ligament_MPa": round(P_max_lig, 3),
        "P_target_MPa":       P_target,
        "pressureMargin":     round(p_margin, 3),
        "pressureFail":       p_fail,
        "pressureWarn":       p_warn,
        "sigma_y_MPa":        round(sigma_y, 1),
        "sigma_a_MPa":        round(sigma_a, 1),
        "T_wall_est_K":       round(T_wall_est, 0),
        "netSectionFraction": round(net_sec_frac, 3),
        "innerLigament_mm":   round(inner_lig_mm, 2),
    }

# ── Scoring helpers ────────────────────────────────────────────────────────────

def eval_manufacturability(p, req, inj_key):
    inj = INJECTOR_TYPES[inj_key]
    add_bonus  = 0.08 if req.get("manufacturingMode") == "additive" else 0
    base_score = 1 - inj["complexity"] * (1 - req.get("complexityTolerance", 0.6))
    score      = clamp(base_score + add_bonus, 0, 1)
    return {"score": score}

def eval_robustness(p, req, thermal):
    req_bias   = req.get("robustnessBias", 0.5) * 0.25
    margin_comp = clamp(thermal["overallMargin"], 0, 1) * 0.5
    wall_thick  = clamp(p["wallThickness"] / 0.5, 0, 1) * 0.25
    score = clamp(margin_comp + wall_thick + req_bias, 0, 1)
    return {"score": score}

def compactness_score(p):
    return clamp(1 - (p.get("contractionRatio", 4) - 2) / 6
                   - p.get("nozzleAR", 8) / 80, 0, 1)
