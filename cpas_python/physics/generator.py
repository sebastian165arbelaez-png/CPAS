"""
CPAS v2 — Candidate Generator + Pareto Ranking
"""
import math
import random
from .data import PROPELLANTS, MATERIALS, INJECTOR_TYPES
from .solvers import (
    build_geometry_profile, compute_governing_state, solve_wall_configuration,
    eval_flow, eval_manufacturability, eval_robustness, compactness_score,
    compute_pressure_limit, size_nozzle, compute_channel_sections, thrust_coeff
)


def validate_candidate(p, geom):
    hard_fails, soft_fails, advisories = [], [], []

    if not geom or geom.get("L_total_m", 0) <= 0:
        hard_fails.append("INVALID_GEOMETRY: total length ≤ 0")
    if geom.get("r_t_m", 0) <= 0 or not math.isfinite(geom.get("r_t_m", 0)):
        hard_fails.append("INVALID_THROAT: throat radius non-positive or non-finite")
    if geom.get("r_e_m", 0) <= geom.get("r_t_m", 1):
        hard_fails.append("NON_EXPANDING_NOZZLE: exit radius ≤ throat radius")
    if geom.get("L_cyl_m", 0) < geom.get("r_c_m", 1) * 0.15:
        hard_fails.append(f"SHORT_CHAMBER: L_cyl={geom.get('L_cyl_m',0)*1000:.1f}mm < 15% of r_c")

    L_tot = geom.get("L_total_m", 1)
    D_c   = geom.get("r_c_m", 1) * 2
    aspect = L_tot / max(D_c, 1e-9)
    if aspect < 1.5:
        hard_fails.append(f"ASPECT_TOO_SHORT: L/D={aspect:.2f} < 1.5")
    if aspect > 12.0:
        hard_fails.append(f"ASPECT_TOO_LONG: L/D={aspect:.2f} > 12")

    cham_len = geom.get("L_cyl_m", 0) + geom.get("L_cvg_m", 0)
    if geom.get("L_noz_m", 0) > cham_len * 2.5:
        hard_fails.append("NOZZLE_DOMINATES: L_noz > 2.5×chamber")

    r_thr = geom.get("r_thr", 0.5)
    if not (0.10 <= r_thr <= 0.90):
        hard_fails.append(f"THROAT_FRACTION_OOB: r_thr/r_c={r_thr:.3f}")

    rs_m = geom.get("rs_m", [])
    if any(not math.isfinite(r) or r < 0 for r in rs_m):
        hard_fails.append("NAN_IN_PROFILE: non-finite or negative radius in profile")

    if geom.get("Lstar_low"):
        soft_fails.append(f"LSTAR_LOW: L*={geom.get('Lstar',0):.2f}m below propellant minimum")
    if geom.get("Lstar_high"):
        soft_fails.append(f"LSTAR_HIGH: L*={geom.get('Lstar',0):.2f}m above propellant maximum")
    if aspect > 8.0:
        soft_fails.append(f"HIGH_ASPECT: L/D={aspect:.2f}")
    if p.get("contractionRatio", 4) > 6.5:
        soft_fails.append(f"HIGH_CONTRACTION: ε_c={p.get('contractionRatio',4):.1f}")
    if p.get("nozzleAR", 8) > 14:
        soft_fails.append(f"HIGH_EXPANSION: ε_e={p.get('nozzleAR',8):.1f}")

    advisories.append("PROXY_METRICS: all scores are reduced-order screening metrics")
    if not p.get("regenerativeCooling"):
        advisories.append("NO_REGEN: ablative or film-only cooling — wall model simplified")

    return {
        "valid":     len(hard_fails) == 0,
        "hardFails": hard_fails,
        "softFails": soft_fails,
        "advisories": advisories,
        "corrected": False,
    }


def generate_candidates(req, N=300, seed=42):
    rng = random.Random(seed)
    prop_key = req.get("propellant", "kerolox")
    inj_key  = req.get("injector", "like_doublet")
    material = req.get("material", "copper")
    prop     = PROPELLANTS[prop_key]
    candidates = []
    attempt    = 0

    # Per-propellant pressure cap: RP-1 needs low pressure to cool copper;
    # LH2/CH4 can handle more. Cap reduces futile thermal-fail attempts.
    p_c_max = {"kerolox": 5.0, "hydrolox": 8.0, "methalox": 6.0}.get(prop_key, 6.0)

    while len(candidates) < N and attempt < N * 16:
        attempt += 1

        # Always use regen cooling — without it thermal fails 100%
        use_regen = True
        # Film cooling: always on for kerolox (needs it), 50% for others
        use_film  = True if prop_key == "kerolox" else rng.random() > 0.5
        use_abl   = rng.random() > 0.85

        p = {
            "throatRadius":       rng.uniform(0.010, 0.080),
            "contractionRatio":   rng.uniform(2.5,   6.5),
            "Lstar":              rng.uniform(prop["Lstar_min"] * 0.90,
                                              prop["Lstar_max"] * 1.20),
            "convergentAngle":    rng.uniform(22, 45),
            "nozzleAngle":        rng.uniform(10, 22),
            "nozzleAR":           rng.uniform(3,  20),
            "chamberPressure":    rng.uniform(0.8, p_c_max),
            "wallThickness":      rng.uniform(0.15, 0.65),   # thinner range → better cooling
            "coolingChannels":    rng.randint(20, 60),        # more channels by default
            "injectorDensity":    rng.uniform(0.35, 0.75),
            "coolantFlow":        rng.uniform(0.3, 0.7),
            "regenerativeCooling": use_regen,
            "filmCooling":         use_film,
            "ablativeLayer":       use_abl,
            "material":            material,
            "injector":            inj_key,
        }

        try:
            geom = build_geometry_profile(p, prop_key)
        except Exception:
            continue

        validation = validate_candidate(p, geom)
        if not validation["valid"]:
            continue

        try:
            wall_result = solve_wall_configuration(p, geom, material, prop_key, inj_key)
        except Exception:
            continue

        p_final    = wall_result["params_converged"]
        thermal    = wall_result["thermal"]
        structural = wall_result["structural"]

        gov_state  = compute_governing_state(p_final, geom, prop_key, 0)
        press_lim  = compute_pressure_limit(p_final, geom, material, prop_key)

        hard_fails = list(validation["hardFails"])
        soft_fails = list(validation["softFails"])
        advisories = list(validation["advisories"])

        if thermal["limitExceeded"]:
            hard_fails.append("THERMAL_LIMIT_EXCEEDED: T_wall_max > material limit")
        if (structural.get("SF_actual") or 2) < 1.5:
            hard_fails.append(f"SF_BELOW_MIN: SF={structural.get('SF_actual',0):.2f} < 1.5")
        if press_lim["pressureFail"]:
            hard_fails.append(
                f"PRESSURE_EXCEEDS_LIMIT: p_c={p_final['chamberPressure']:.2f} MPa "
                f"> P_max={press_lim['P_max_MPa']} MPa"
            )
        if not wall_result["wall_converged"]:
            soft_fails.append(f"WALL_UNCONVERGED: did not converge in {wall_result['wall_iterations']} iterations")
        if geom.get("chamberGovernedByPhysics"):
            crit = geom.get("chamberGoverningCriterion", "physics").upper()
            soft_fails.append(f"CHAMBER_GOVERNED_BY_{crit}")
        if press_lim["pressureWarn"]:
            soft_fails.append(f"PRESSURE_LOW_MARGIN: margin={press_lim['pressureMargin']*100:.0f}% < 20%")

        is_hard_failed = len(hard_fails) > 0

        try:
            flow = eval_flow(p_final, geom, inj_key, prop_key, gov_state)
            mfg  = eval_manufacturability(p_final, req, inj_key)
            rob  = eval_robustness(p_final, req, thermal)
        except Exception:
            continue

        if flow.get("faceTooHot"):
            soft_fails.append("FACE_OVERHEAT: droplets vaporising close to injector face")
        if not flow.get("stiffOK"):
            soft_fails.append("LOW_STIFFNESS: injector ΔP below stability minimum")
        if flow.get("SMD_um", 0) > 150:
            soft_fails.append("HIGH_SMD: large droplets, poor atomisation")

        wt   = 0.28 + req.get("coolingPriority", 0.7) * 0.12
        ws   = 0.18
        wf   = 0.14
        wm   = 0.14 + (0.04 if req.get("manufacturingMode") == "additive" else 0)
        wr   = 0.10 + req.get("robustnessBias", 0.5) * 0.08
        wc   = req.get("compactnessWeight", 0.5) * 0.12
        sumW = wt + ws + wf + wm + wr + wc

        c_sc  = compactness_score(p_final)
        total = max(0, min(1,
            (wt * thermal["score"] + ws * structural["score"] + wf * flow["score"]
             + wm * mfg["score"] + wr * rob["score"] + wc * c_sc) / sumW
        ))

        nozzle_design = size_nozzle(
            geom["r_t_m"], geom["eps_e"], p_final.get("nozzleAngle", 15),
            p_final["chamberPressure"] * 1e6, 0, prop["gamma"],
            (geom["L_cyl_m"] + geom["L_cvg_m"]) * 2.5
        )
        ch_secs = compute_channel_sections(p_final, geom, prop_key)["sections"]

        if any(s.get("ligamentThin") for s in ch_secs):
            soft_fails.append("THIN_LIGAMENT: inner ligament < 20% of channel depth")

        candidates.append({
            "id":        f"C-{attempt:04d}",
            "params":    p_final,
            "geom":      geom,
            "material":  material,
            "propellant": prop_key,
            "injector":  inj_key,
            "evals":     {"thermal": thermal, "structural": structural,
                          "flow": flow, "mfg": mfg, "robust": rob},
            "score":     total * 0.1 if is_hard_failed else total,
            "breakdown": {
                "thermal": thermal["score"], "structural": structural["score"],
                "flow": flow["score"], "mfg": mfg["score"],
                "robust": rob["score"], "compactness": c_sc,
            },
            "hardFails":  hard_fails,
            "softFails":  soft_fails,
            "advisories": advisories,
            "isHardFailed": is_hard_failed,
            "wall_converged":  wall_result["wall_converged"],
            "wall_iterations": wall_result["wall_iterations"],
            "governingState": gov_state,
            "pressureLimit":  press_lim,
            "nozzleDesign":   nozzle_design,
            "channelSections": ch_secs,
            "paretoRank":  0,
            "limitExceeded": thermal["limitExceeded"],
        })

    # True 6D Pareto dominance
    dims = ["thermal", "structural", "flow", "mfg", "robust", "compactness"]
    for i, ci in enumerate(candidates):
        if ci["isHardFailed"]:
            ci["paretoRank"] = 99
            continue
        dominated = False
        for j, cj in enumerate(candidates):
            if i == j or cj["isHardFailed"]:
                continue
            a = cj["breakdown"]
            b = ci["breakdown"]
            if all(a.get(d, 0) >= b.get(d, 0) for d in dims) \
               and any(a.get(d, 0) >  b.get(d, 0) for d in dims):
                dominated = True
                break
        ci["paretoRank"] = 1 if dominated else 0

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates


def run_sweep(sweep_type, base_params, prop_key, inj_key, material, N=50):
    """Trade sweep across one parameter."""
    prop = PROPELLANTS[prop_key]

    ranges = {
        "pressure": (0.5, 20,  "p_c (MPa)", "MPa"),
        "lstar":    (prop["Lstar_min"]*0.7, prop["Lstar_max"]*1.4, "L* (m)", "m"),
        "nozzleAR": (2, 40,   "Nozzle ε_e", ""),
        "channels": (4, 64,   "Channel count", ""),
    }
    x_min, x_max, x_label, x_unit = ranges.get(sweep_type, ranges["pressure"])
    points = []

    for i in range(N):
        t    = i / (N - 1)
        xval = x_min + t * (x_max - x_min)
        p = dict(base_params)
        if sweep_type == "pressure":
            p["chamberPressure"] = xval
        elif sweep_type == "lstar":
            p["Lstar"] = xval
        elif sweep_type == "nozzleAR":
            p["nozzleAR"] = xval
        elif sweep_type == "channels":
            p["coolingChannels"] = int(round(xval))

        try:
            geom  = build_geometry_profile(p, prop_key)
            gs    = compute_governing_state(p, geom, prop_key, 0)
            pl    = compute_pressure_limit(p, geom, material, prop_key)
            nd    = size_nozzle(geom["r_t_m"], geom["eps_e"], p.get("nozzleAngle", 15),
                                p["chamberPressure"]*1e6, 0, prop["gamma"],
                                (geom["L_cyl_m"]+geom["L_cvg_m"])*2.5)
            cs    = geom.get("chamberSizing", {})
            CF_vac = thrust_coeff(prop["gamma"], geom["eps_e"], p["chamberPressure"]*1e6, 0)
            points.append({
                "x":              xval,
                "valid":          not pl["pressureFail"],
                "Isp_vac":        prop["Isp_vac"] * INJECTOR_TYPES[inj_key]["mix_eff"],
                "thrust_vac_kN":  gs["thrust_vac_kN"],
                "p_c_MPa":        p["chamberPressure"],
                "L_cyl_mm":       geom["L_cyl_m"] * 1000,
                "L_total_mm":     geom["L_total_m"] * 1000,
                "L_noz_mm":       nd["nozzleLength_m"] * 1000,
                "P_max_MPa":      pl["P_max_MPa"],
                "pressureMargin": pl["pressureMargin"],
                "pressureFail":   pl["pressureFail"],
                "chamberCriterion": cs.get("governingCriterion", "Lstar"),
                "governedByPhysics": cs.get("governedByPhysics", False),
                "LD_ratio":       cs.get("LD_ratio", 0),
                "vapClipped":     cs.get("vapClipped", False),
                "nozzleAR":       geom["eps_e"],
                "nozzleClipped":  nd["areaRatioClipped"],
            })
        except Exception:
            points.append({"x": xval, "valid": False})

    return {"points": points, "xLabel": x_label, "xUnit": x_unit, "sweepType": sweep_type}
