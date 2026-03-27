# CPAS v2 — Computational Propulsion Architecture Synthesis
### Python / Dash Edition

A browser-based, concept-level engineering tool for the synthesis, thermal analysis,
structural screening, and CAD handoff of liquid-propellant rocket engines.

---

## Quick Start

```bash
# 1. Install dependencies (once)
pip install dash plotly numpy

# 2. Run
python app.py

# 3. Open browser at http://localhost:8050
```

The browser opens automatically. No Node.js, no build step, no cloud account required.

---

## Project Structure

```
cpas_python/
├── app.py                  Main Dash application (UI + callbacks)
├── requirements.txt        Python dependencies
├── assets/
│   └── style.css           Dark engineering theme
└── physics/
    ├── __init__.py         Package exports
    ├── data.py             Engineering databases (propellants, materials, injectors)
    ├── solvers.py          All physics functions (Bartz, Dittus-Boelter, Lamé, etc.)
    └── generator.py        Candidate generation, Pareto ranking, trade sweeps
```

---

## Three Workflows

### 1. Mission-Centred (◎ Mission tab)
Select propellant, injector, material, and objective weights → generate 300 candidates
→ Pareto-ranked dashboard → Inspector (thermal, structural, heat flux, 3D, CAD)
→ Trade Study (Pareto scatter, radar chart).

### 2. Parameters-Centred (⚡ Parameters tab)
Enter mass flow rates and throat area directly → real-time performance prediction,
optimal nozzle expansion ratio, injector sizing, altitude comparison, and trade sweeps
(pressure / L* / ε_e / channel count).

### 3. Verify Mode (⊕ Verify tab)
Six automated physics tests required for technical signoff (CPAS audit §8):
run them with one click — each asserts a directional physical relationship against
the actual solver functions.

---

## Supported Propellants

| Key | Propellant | Isp_vac (s) | Notes |
|-----|-----------|-------------|-------|
| kerolox | RP-1 / LOX | 358 | Storable fuel, cryogenic ox. Film cooling recommended. |
| hydrolox | LH₂ / LOX | 450 | Best Isp, deeply cryogenic. Excellent regen coolant. |
| methalox | LCH₄ / LOX | 380 | Best density-Isp balance. Good regen coolant. |

## Supported Materials

| Key | Material | T_limit (K) | k (W/mK) |
|-----|---------|-------------|---------|
| copper | CuCrZr | 773 | 320 |
| steel | 316L | 1073 | 16 |
| inconel | Inconel 625 | 1423 | 10 |

## Supported Injectors

Showerhead, Like-on-like Doublet, Unlike Doublet, Triplet, Pressure-Swirl,
Swirl Coaxial, Pintle.

---

## Physics Modules

| Module | Solver | Reference |
|--------|--------|-----------|
| Chamber sizing | 5-criterion (L*, vaporisation D² law, residence τ, injector min, stability) | Huzel & Huang Ch.4 |
| Gas-side heat transfer | Bartz (1957) proxy, scale 0.5×10⁵ | Sutton 9th §8 |
| Coolant-side h | Dittus-Boelter, real D_h from channel geometry | — |
| Wall temperature | 3-resistance network (self-consistent T_wg solve) | — |
| Film cooling | Exponential decay η_f = 0.55·exp(−x/L_decay) | Huzel & Huang §4-3 |
| Structural stress | Liu et al. 2023 multi-layer thermoelastic | J. Phys.: Conf. Ser. 2489 |
| Hollow-wall penalties | Net-section (Amp A), rib slenderness (Amp B), Inglis Kt | — |
| Pressure limits | Lamé thin-wall: solid, channel, ligament | — |
| Injector sizing | Bernoulli/Cd, Lefebvre SMD, D² vaporisation | Ceotto & Tavares |
| Nozzle | Conical, Rao blend radii, contour points | Rao (1958) |
| Performance | Isentropic CF, c*, Isp scaling | Sutton 9th §3 |
| Pareto ranking | True 6D non-dominated sort | — |

---

## Known Limitations

- All physics is **reduced-order concept-level**. Not a substitute for RPA/CEA or FEA.
- Nozzle is always **conical** (no Rao bell contour yet).
- RP-1 (kerolox) with copper + regen alone is **physically marginal** at > 2 MPa —
  film cooling is required. This is correct engineering (see Merlin engine).
- STEP/STL exports are **OPEN_SHELL concept models** — not manufacturing-grade solids.
- The JSON cadPayload is the authoritative geometric output for downstream CAD work.

---

## Sharing on a Network

To share with colleagues on your local network, change the last line in `app.py`:

```python
# Change:
app.run(debug=False, port=8050)
# To:
app.run(debug=False, port=8050, host="0.0.0.0")
```

Then access from another machine at `http://YOUR_IP:8050`.

---

## References

- Sutton & Biblarz, *Rocket Propulsion Elements*, 9th ed.
- Huzel & Huang, *Modern Engineering for Design of Liquid-Propellant Rocket Engines*
- Ceotto & Tavares, *Design of a Liquid Nitrous Oxide and Ethanol Rocket Engine Injector*
- Liu et al. (2023), *J. Phys.: Conf. Ser. 2489 012005*
- Bartz, D.R. (1957), JPL Technical Report
- Rao, G.V.R. (1958), *Jet Propulsion* 28(6)
- Lefebvre, A.H., *Atomization and Sprays*
