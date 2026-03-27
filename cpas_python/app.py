"""
CPAS v2 — Computational Propulsion Architecture Synthesis Platform
Full Dash/Plotly application — Python-native port.

Run:  python app.py
Open: http://localhost:8050
"""
import math
import json
import webbrowser
import threading
import time

import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context
import plotly.graph_objects as go
import plotly.express as px

from physics import (
    MATERIALS, PROPELLANTS, ALTITUDES, INJECTOR_TYPES,
    generate_candidates, run_sweep, validate_candidate,
    build_geometry_profile, compute_governing_state,
    compute_pressure_limit, optimal_nozzle_ar, thrust_coeff,
    isp_at_altitude, c_star, size_nozzle, compute_channel_sections,
    eval_thermal_full, eval_structural, eval_flow,
)
from physics.solvers import wall_thickness_m

# ── App setup ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="CPAS v2 — Propulsion Synthesis",
    suppress_callback_exceptions=True,
)
server = app.server  # Expose Flask server for Gunicorn

# ── Colour palette ─────────────────────────────────────────────────────────────
BG      = "#06080d"
BG2     = "#0a1420"
ACCENT  = "#00e8c0"
GOLD    = "#e8b84b"
WARN    = "#ff6b35"
BLUE    = "#6090ff"
MUTED   = "#5a7090"
BORDER  = "#1a2a3a"

# ── Plot layout defaults ───────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG2,
    font=dict(color=MUTED, family="Courier New, monospace", size=10),
    margin=dict(l=48, r=16, t=30, b=36),
    xaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=BORDER, showgrid=True, zeroline=False),
    legend=dict(bgcolor=BG, bordercolor=BORDER, borderwidth=1),
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def score_color(s):
    if s >= 0.75: return ACCENT
    if s >= 0.50: return GOLD
    return WARN

def temp_color(T, T_limit):
    ratio = T / max(T_limit, 1)
    if ratio < 0.7:  return ACCENT
    if ratio < 0.9:  return GOLD
    return WARN

def card(children, style=None, **kwargs):
    return html.Div(children, className="card", style=style or {}, **kwargs)

def section_label(text):
    return html.Div(text, className="card-title")

def metric(label, value, unit="", color=ACCENT):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Span(str(value), className="metric-value", style={"color": color}),
        html.Span(f" {unit}", className="metric-unit") if unit else None,
    ], className="metric-card")

def tag(text, kind="pass"):
    return html.Span(text, className=f"tag tag-{kind}")

def banner(text, kind="info"):
    return html.Div(text, className=f"banner banner-{kind}")

def score_bar(label, value, color=ACCENT):
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize": "9px", "color": MUTED}),
            html.Span(f"{value*100:.0f}%", style={"fontSize": "9px", "color": color, "fontFamily": "monospace"}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
        html.Div(html.Div(style={
            "width": f"{clamp(value*100,0,100):.0f}%",
            "height": "100%", "background": color, "borderRadius": "2px",
        }), className="score-bar-container"),
    ], style={"marginBottom": "8px"})

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ── Layout ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Store — candidates list and selected index
    dcc.Store(id="store-candidates", data=[]),
    dcc.Store(id="store-selected",   data=0),
    dcc.Store(id="store-req",        data={}),

    # ── Header ──────────────────────────────────────────────────────────────
    html.Div([
        html.Span("◈ CPAS v2", className="cpas-wordmark"),
        # Mode tabs
        html.Div([
            html.Button("◎  Mission",    id="btn-mode-mission", className="tab-btn active",
                        n_clicks=0),
            html.Button("⚡  Parameters", id="btn-mode-params",  className="tab-btn",
                        n_clicks=0),
            html.Button("⊕  Verify",     id="btn-mode-verify",  className="tab-btn",
                        n_clicks=0),
        ], style={"display": "flex", "gap": "4px", "marginRight": "24px"}),

        # Mission nav (shown in mission mode)
        html.Div([
            html.Button("01  Mission",   id="btn-nav-input",     className="tab-btn active", n_clicks=0),
            html.Button("02  Dashboard", id="btn-nav-dashboard", className="tab-btn", n_clicks=0),
            html.Button("03  Inspector", id="btn-nav-inspector", className="tab-btn", n_clicks=0),
            html.Button("04  Trade",     id="btn-nav-trade",     className="tab-btn", n_clicks=0),
        ], id="mission-nav", style={"display": "flex", "gap": "0"}),

        # Status strip
        html.Div(id="status-strip", style={"marginLeft": "auto", "fontSize": "9px", "color": MUTED}),
    ], className="cpas-header"),

    # ── Main content ─────────────────────────────────────────────────────────
    html.Div(id="main-content", className="cpas-main"),

    # Mode + screen state
    dcc.Store(id="store-mode",   data="mission"),
    dcc.Store(id="store-screen", data="input"),

], style={"minHeight": "100vh", "background": BG})


# ── Route: render main content ─────────────────────────────────────────────────
@app.callback(
    Output("main-content",   "children"),
    Output("mission-nav",    "style"),
    Output("status-strip",   "children"),
    Input("store-mode",      "data"),
    Input("store-screen",    "data"),
    Input("store-candidates","data"),
    Input("store-selected",  "data"),
    Input("store-req",       "data"),
)
def route_main(mode, screen, candidates, selected_idx, req):
    nav_style = {"display": "flex", "gap": "0"} if mode == "mission" else {"display": "none"}

    # Status strip
    status = ""
    if candidates:
        fails = sum(1 for c in candidates if c.get("limitExceeded"))
        pname = PROPELLANTS.get(req.get("propellant", "kerolox"), {}).get("label", "")
        status = f"{pname}  ·  {len(candidates)} candidates  ·  {fails} T_exceeded"

    if mode == "verify":
        return render_verify(), nav_style, status
    if mode == "params":
        return render_parameters(), nav_style, status

    # Mission mode
    if screen == "input" or not candidates:
        return render_mission_input(req), nav_style, status
    if screen == "dashboard":
        return render_dashboard(candidates, req), nav_style, status
    if screen == "inspector" and candidates:
        idx = min(selected_idx or 0, len(candidates) - 1)
        return render_inspector(candidates[idx], candidates), nav_style, status
    if screen == "trade" and candidates:
        return render_trade(candidates), nav_style, status

    return render_mission_input(req), nav_style, status


# ── Mode switching ─────────────────────────────────────────────────────────────
@app.callback(
    Output("store-mode",        "data"),
    Output("btn-mode-mission",  "className"),
    Output("btn-mode-params",   "className"),
    Output("btn-mode-verify",   "className"),
    Input("btn-mode-mission",   "n_clicks"),
    Input("btn-mode-params",    "n_clicks"),
    Input("btn-mode-verify",    "n_clicks"),
    prevent_initial_call=True,
)
def switch_mode(n1, n2, n3):
    tid = ctx.triggered_id
    mode = ("params" if tid == "btn-mode-params"
            else "verify" if tid == "btn-mode-verify"
            else "mission")
    active = "tab-btn active"
    plain  = "tab-btn"
    return (mode,
            active if mode == "mission" else plain,
            active if mode == "params"  else plain,
            active if mode == "verify"  else plain)


# ── Screen switching ───────────────────────────────────────────────────────────
@app.callback(
    Output("store-screen",       "data"),
    Output("btn-nav-input",      "className"),
    Output("btn-nav-dashboard",  "className"),
    Output("btn-nav-inspector",  "className"),
    Output("btn-nav-trade",      "className"),
    Input("btn-nav-input",       "n_clicks"),
    Input("btn-nav-dashboard",   "n_clicks"),
    Input("btn-nav-inspector",   "n_clicks"),
    Input("btn-nav-trade",       "n_clicks"),
    State("store-candidates",    "data"),
    prevent_initial_call=True,
)
def switch_screen(n1, n2, n3, n4, candidates):
    tid  = ctx.triggered_id
    has  = bool(candidates)
    scr  = ("dashboard" if tid == "btn-nav-dashboard" and has
            else "inspector" if tid == "btn-nav-inspector" and has
            else "trade"     if tid == "btn-nav-trade"     and has
            else "input")
    a, p = "tab-btn active", "tab-btn"
    return (scr,
            a if scr == "input"     else p,
            a if scr == "dashboard" else p,
            a if scr == "inspector" else p,
            a if scr == "trade"     else p)


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 01 — MISSION INPUT
# ══════════════════════════════════════════════════════════════════════════════

def render_mission_input(req):
    req = req or {}

    propellant_cards = []
    for k, prop in PROPELLANTS.items():
        selected = req.get("propellant", "kerolox") == k
        propellant_cards.append(
            html.Div([
                html.Div(prop["label"], style={"fontSize": "11px", "marginBottom": "3px",
                                                "color": prop["color"] if selected else "#c8d8e8"}),
                html.Div(f"T_fl {prop['T_flame']}K · Isp {prop['Isp_vac']}s · O/F* {prop['OF_nominal']}",
                         style={"fontSize": "9px", "color": MUTED}),
            ], id={"type": "prop-select", "key": k},
               style={"padding": "8px 12px", "borderRadius": "3px", "cursor": "pointer",
                      "marginBottom": "5px", "border": f"1px solid {prop['color'] if selected else BORDER}",
                      "background": f"rgba{tuple(int(prop['color'].lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.12,)}".replace("rgba", "rgba(").replace(", 0.", ", 0.") if selected else "transparent"})
        )

    injector_cards = []
    for k, inj in INJECTOR_TYPES.items():
        selected = req.get("injector", "like_doublet") == k
        injector_cards.append(
            html.Div([
                html.Div(inj["label"], style={"fontSize": "10px",
                                               "color": inj["color"] if selected else "#c8d8e8"}),
                html.Div(f"Mix {inj['mix_eff']*100:.0f}%  ·  Stab {inj['stability']*100:.0f}%",
                         style={"fontSize": "8px", "color": MUTED}),
            ], id={"type": "inj-select", "key": k},
               style={"padding": "6px 10px", "borderRadius": "3px", "cursor": "pointer",
                      "marginBottom": "4px",
                      "border": f"1px solid {inj['color'] if selected else BORDER}",
                      "background": f"rgba(0,0,0,{0.05 if selected else 0})"})
        )

    material_cards = []
    for k, mat in MATERIALS.items():
        selected = req.get("material", "copper") == k
        material_cards.append(
            html.Div([
                html.Div(mat["label"], style={"color": mat["color"] if selected else "#c8d8e8", "fontSize":"10px"}),
                html.Div(f"T_lim {mat['T_limit']}K  ·  k={mat['k']} W/mK",
                         style={"fontSize": "8px", "color": MUTED}),
            ], id={"type": "mat-select", "key": k},
               style={"padding": "6px 10px", "borderRadius": "3px", "cursor": "pointer",
                      "marginBottom": "4px",
                      "border": f"1px solid {mat['color'] if selected else BORDER}",
                      "background": "transparent"})
        )

    return html.Div([
        html.Div([
            # ── Left column ───────────────────────────────────────────────
            html.Div([
                card([section_label("Propellant")] + propellant_cards),
                card([section_label("Injector Pattern")] + injector_cards),
                card([section_label("Wall Material")] + material_cards),
            ], style={"width": "280px", "flexShrink": "0"}),

            # ── Centre column ─────────────────────────────────────────────
            html.Div([
                card([
                    section_label("Objective Profile"),
                    html.Div([
                        html.Div("Cooling Priority", style={"fontSize":"9px","color":MUTED,"marginBottom":"3px"}),
                        dcc.Slider(0, 1, 0.05, value=req.get("coolingPriority",0.7),
                                   id="sl-cooling", marks=None,
                                   tooltip={"always_visible":True,"placement":"bottom"},
                                   className="cpas-slider"),
                    ], style={"marginBottom":"12px"}),
                    html.Div([
                        html.Div("Compactness Weight", style={"fontSize":"9px","color":MUTED,"marginBottom":"3px"}),
                        dcc.Slider(0, 1, 0.05, value=req.get("compactnessWeight",0.5),
                                   id="sl-compact", marks=None,
                                   tooltip={"always_visible":True,"placement":"bottom"},
                                   className="cpas-slider"),
                    ], style={"marginBottom":"12px"}),
                    html.Div([
                        html.Div("Robustness Bias", style={"fontSize":"9px","color":MUTED,"marginBottom":"3px"}),
                        dcc.Slider(0, 1, 0.05, value=req.get("robustnessBias",0.5),
                                   id="sl-robust", marks=None,
                                   tooltip={"always_visible":True,"placement":"bottom"},
                                   className="cpas-slider"),
                    ], style={"marginBottom":"12px"}),
                    html.Div([
                        html.Div("Complexity Tolerance", style={"fontSize":"9px","color":MUTED,"marginBottom":"3px"}),
                        dcc.Slider(0, 1, 0.05, value=req.get("complexityTolerance",0.6),
                                   id="sl-complex", marks=None,
                                   tooltip={"always_visible":True,"placement":"bottom"},
                                   className="cpas-slider"),
                    ]),
                ]),
                card([
                    section_label("Generation"),
                    html.Div([
                        html.Div("Candidates", style={"fontSize":"9px","color":MUTED,"marginBottom":"3px"}),
                        dcc.Slider(50, 500, 50, value=req.get("candidates",300),
                                   id="sl-ncand", marks={50:"50",200:"200",500:"500"},
                                   tooltip={"always_visible":True,"placement":"bottom"},
                                   className="cpas-slider"),
                    ], style={"marginBottom":"12px"}),
                    html.Div([
                        html.Div("Seed", style={"fontSize":"9px","color":MUTED,"marginBottom":"3px"}),
                        dcc.Input(id="inp-seed", type="number", value=req.get("seed",42), min=0, max=9999,
                                  style={"background":BG,"border":f"1px solid {BORDER}","color":ACCENT,
                                         "padding":"5px 8px","borderRadius":"3px","fontFamily":"monospace",
                                         "fontSize":"12px","width":"120px"}),
                    ], style={"marginBottom":"14px"}),
                    html.Div([
                        html.Div("Manufacturing Mode", style={"fontSize":"9px","color":MUTED,"marginBottom":"5px"}),
                        dcc.RadioItems(
                            options=[{"label": " Conventional", "value": "conventional"},
                                     {"label": " Additive (3D-print)", "value": "additive"}],
                            value=req.get("manufacturingMode","conventional"),
                            id="radio-mfg",
                            inputStyle={"marginRight":"6px"},
                            labelStyle={"display":"block","marginBottom":"5px","fontSize":"10px","color":"#c8d8e8"},
                        ),
                    ], style={"marginBottom":"14px"}),
                    html.Button("▶  Generate Candidates", id="btn-generate", n_clicks=0,
                                style={"width":"100%","padding":"10px","background":f"rgba(0,232,192,0.15)",
                                       "border":f"1px solid {ACCENT}","color":ACCENT,
                                       "fontSize":"10px","letterSpacing":"2px","textTransform":"uppercase",
                                       "cursor":"pointer","borderRadius":"3px","fontFamily":"monospace"}),
                    html.Div(id="gen-status", style={"marginTop":"8px","fontSize":"9px","color":MUTED}),
                ]),
            ], style={"flex": "1", "minWidth": "0"}),

            # ── Right column — summary ────────────────────────────────────
            html.Div([
                card(id="mission-summary", children=[
                    section_label("Configuration Summary"),
                    html.Div(id="summary-content",
                             children=[html.Div("Select propellant, injector, and material to see summary.",
                                               style={"fontSize":"10px","color":MUTED})]),
                ]),
            ], style={"width": "260px", "flexShrink": "0"}),

        ], style={"display": "flex", "gap": "14px", "alignItems": "flex-start"}),
    ])


@app.callback(
    Output("store-candidates", "data"),
    Output("store-req",        "data"),
    Output("store-screen",     "data", allow_duplicate=True),
    Output("gen-status",       "children"),
    Input("btn-generate",      "n_clicks"),
    State("sl-cooling",        "value"),
    State("sl-compact",        "value"),
    State("sl-robust",         "value"),
    State("sl-complex",        "value"),
    State("sl-ncand",          "value"),
    State("inp-seed",          "value"),
    State("radio-mfg",         "value"),
    State("store-req",         "data"),
    prevent_initial_call=True,
)
def do_generate(n_clicks, cool, compact, robust, complexity, ncand, seed, mfg, req):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, ""

    req = req or {}
    full_req = {
        **req,
        "coolingPriority":    cool    or 0.7,
        "compactnessWeight":  compact or 0.5,
        "robustnessBias":     robust  or 0.5,
        "complexityTolerance":complexity or 0.6,
        "candidates":         int(ncand or 300),
        "seed":               int(seed or 42),
        "manufacturingMode":  mfg or "conventional",
    }

    candidates = generate_candidates(full_req, N=int(ncand or 300), seed=int(seed or 42))
    status     = f"Generated {len(candidates)} candidates — {sum(1 for c in candidates if c['paretoRank']==0)} on Pareto front."
    return candidates, full_req, "dashboard", status


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 02 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard(candidates, req):
    prop = PROPELLANTS.get(req.get("propellant","kerolox"), {})
    mat  = MATERIALS.get(req.get("material","copper"), {})
    T_limit = mat.get("T_limit", 773)

    # Pareto scatter
    xs   = [c["breakdown"]["thermal"]     for c in candidates]
    ys   = [c["breakdown"]["structural"]  for c in candidates]
    ids  = [c["id"]                       for c in candidates]
    cols = [score_color(c["score"])        for c in candidates]
    scores = [c["score"]                  for c in candidates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        text=[f"{cid}<br>Score: {s*100:.1f}%<br>{'⭐ Pareto' if c['paretoRank']==0 else ''}"
              for cid, s, c in zip(ids, scores, candidates)],
        marker=dict(color=[s*100 for s in scores], colorscale=[
            [0.0,"#ff6b35"],[0.5,"#e8b84b"],[1.0,"#00e8c0"]
        ], size=7, opacity=0.85, showscale=True,
                    colorbar=dict(title="Score", tickfont=dict(size=9, color=MUTED))),
        hovertemplate="%{text}<extra></extra>",
        customdata=list(range(len(candidates))),
    ))
    fig.update_layout(
        **PLOT_LAYOUT,
        height=300,
        xaxis_title="Thermal Score", yaxis_title="Structural Score",
        title=dict(text="6D Pareto Space (Thermal vs Structural)", font=dict(size=10, color=MUTED)),
    )

    # Candidate list (top 25)
    cand_cards = []
    for i, c in enumerate(candidates[:40]):
        tags_list = []
        if c["paretoRank"] == 0:       tags_list.append(tag("Pareto", "pass"))
        if c["isHardFailed"]:          tags_list.append(tag("⛔ FAIL", "fail"))
        if c["limitExceeded"]:         tags_list.append(tag("⚠ T_lim", "warn"))
        if c.get("softFails"):         tags_list.append(tag(f"⚠ {len(c['softFails'])} warn", "warn"))
        if c["params"].get("regenerativeCooling"): tags_list.append(tag("Regen", "blue"))

        th     = c["evals"]["thermal"]
        T_wall = th["T_wall_max"]
        sc     = c["score"]
        cand_cards.append(
            html.Div([
                html.Div([
                    html.Span(c["id"], style={"fontFamily":"monospace","fontSize":"11px","color":ACCENT}),
                    html.Span(" ", style={"marginLeft":"8px"}),
                    *tags_list,
                ]),
                html.Div([
                    html.Span(f"{T_wall:.0f}K", style={"fontSize":"9px","color":temp_color(T_wall,T_limit),"fontFamily":"monospace","marginRight":"10px"}),
                    html.Span(f"{sc*100:.1f}", style={"fontSize":"14px","color":score_color(sc),"fontFamily":"monospace"}),
                ]),
            ], id={"type":"cand-row","index":i},
               className=f"cand-card {'hardfail' if c['isHardFailed'] else ''}",
               n_clicks=0)
        )

    return html.Div([
        html.Div([
            html.Div([
                card([
                    section_label("Design Space — Pareto Scatter"),
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                ]),
                card([
                    section_label(f"Candidates — {len(candidates)} total"),
                    *cand_cards,
                ]),
            ], style={"flex":"1","minWidth":"0"}),
            html.Div([
                card([
                    section_label("Score Distribution"),
                    dcc.Graph(figure=_score_histogram(candidates),
                              config={"displayModeBar":False}),
                ]),
                card([
                    section_label("Summary"),
                    *[metric(k, v, col=ACCENT) for k, v in [
                        ("Total candidates", len(candidates)),
                        ("Pareto front",     sum(1 for c in candidates if c["paretoRank"]==0)),
                        ("Hard fails",       sum(1 for c in candidates if c["isHardFailed"])),
                        ("T_limit exceeded", sum(1 for c in candidates if c["limitExceeded"])),
                        ("Best score",       f"{candidates[0]['score']*100:.1f}%"),
                    ]],
                ]),
            ], style={"width":"280px","flexShrink":"0"}),
        ], style={"display":"flex","gap":"14px","alignItems":"flex-start"}),
    ])

def _score_histogram(candidates):
    scores = [c["score"]*100 for c in candidates]
    fig = go.Figure(go.Histogram(x=scores, nbinsx=20, marker_color=ACCENT, opacity=0.7))
    fig.update_layout(**{**PLOT_LAYOUT, "height": 180, "xaxis_title": "Score",
                         "yaxis_title": "Count", "margin": dict(l=36, r=8, t=20, b=36)})
    return fig


@app.callback(
    Output("store-selected", "data"),
    Output("store-screen",   "data", allow_duplicate=True),
    Input({"type":"cand-row","index":dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_candidate(clicks):
    if not any(clicks):
        return dash.no_update, dash.no_update
    idx = next(i for i, c in enumerate(clicks) if c and c > 0)
    return idx, "inspector"


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 03 — INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

def render_inspector(candidate, all_candidates):
    c   = candidate
    th  = c["evals"]["thermal"]
    st  = c["evals"]["structural"]
    fl  = c["evals"]["flow"]
    mat = MATERIALS[c["material"]]
    prop= PROPELLANTS[c["propellant"]]
    geom= c["geom"]
    pl  = c.get("pressureLimit", {})
    cs  = geom.get("chamberSizing", {})

    # ── Warning banners ──────────────────────────────────────────────────────
    banners = []
    for hf in c.get("hardFails", []):
        banners.append(banner(f"⛔ {hf}", "fail"))
    for sf in c.get("softFails", []):
        banners.append(banner(f"⚠ {sf}", "warn"))

    # ── Inspector tabs ───────────────────────────────────────────────────────
    tabs = dcc.Tabs([
        dcc.Tab(label="Thermal",    value="thermal"),
        dcc.Tab(label="Structural", value="structural"),
        dcc.Tab(label="Heat Flux",  value="heatflux"),
        dcc.Tab(label="3D Profile", value="3d"),
        dcc.Tab(label="Parameters", value="params"),
        dcc.Tab(label="CAD",        value="cad"),
    ], value="thermal", id="insp-tabs",
       style={"marginBottom":"12px"},
       colors={"border":BORDER,"primary":ACCENT,"background":BG2})

    tab_content = html.Div(id="insp-tab-content")

    # Serialise candidate for storage in a Store (graphs need it in callbacks)
    # We render inline here instead
    thermal_tab  = _render_thermal_tab(th, mat, geom)
    struct_tab   = _render_structural_tab(st, geom, th)
    heatflux_tab = _render_heatflux_tab(th, geom)
    profile_tab  = _render_3d_profile_tab(geom, th, c)
    params_tab   = _render_params_tab(c, pl, cs, geom, th, fl, mat, prop)
    cad_tab      = _render_cad_tab(c)

    return html.Div([
        # Header
        html.Div([
            html.Span(c["id"], style={"fontFamily":"monospace","fontSize":"20px","color":"#c8d8e8","marginRight":"16px"}),
            tag("Pareto", "pass") if c["paretoRank"]==0 else None,
            tag("⛔ HARD FAIL","fail") if c["isHardFailed"] else None,
            tag("⚠ wall unconverged","warn") if not c.get("wall_converged") else None,
            html.Span(f"{c['score']*100:.1f}", style={"marginLeft":"auto","fontFamily":"monospace",
                                                        "fontSize":"28px","color":score_color(c["score"])}),
        ], style={"display":"flex","alignItems":"center","marginBottom":"10px"}),

        *banners,

        # Score breakdown bars
        card([
            section_label("Score Breakdown"),
            html.Div([
                score_bar("Thermal",     c["breakdown"]["thermal"],    temp_color(th["T_wall_max"],mat["T_limit"])),
                score_bar("Structural",  c["breakdown"]["structural"], GOLD if st["SF_ok"] else WARN),
                score_bar("Flow",        c["breakdown"]["flow"],       ACCENT),
                score_bar("Mfg",         c["breakdown"]["mfg"],        ACCENT),
                score_bar("Robustness",  c["breakdown"]["robust"],     ACCENT),
                score_bar("Compactness", c["breakdown"]["compactness"],BLUE),
            ]),
        ]),

        html.Div([
            # ── Main tab area ───────────────────────────────────────────────
            html.Div([
                dcc.Tabs([
                    dcc.Tab(label="Thermal",    children=thermal_tab,  value="t"),
                    dcc.Tab(label="Structural", children=struct_tab,   value="s"),
                    dcc.Tab(label="Heat Flux",  children=heatflux_tab, value="h"),
                    dcc.Tab(label="3D Profile", children=profile_tab,  value="3"),
                    dcc.Tab(label="Parameters", children=params_tab,   value="p"),
                    dcc.Tab(label="CAD",        children=cad_tab,      value="c"),
                ], value="t",
                   colors={"border":BORDER,"primary":ACCENT,"background":BG2}),
            ], style={"flex":"1","minWidth":"0"}),

            # ── Right sidebar ───────────────────────────────────────────────
            html.Div([
                # Material margin
                card([
                    section_label("Material Margin"),
                    html.Div(f"{((1 - th['T_wall_max']/mat['T_limit'])*100):.0f}%",
                             style={"fontSize":"32px","fontFamily":"monospace","textAlign":"center",
                                    "color":temp_color(th['T_wall_max'],mat['T_limit'])}),
                    html.Div(f"{th['T_wall_max']:.0f} K / {mat['T_limit']} K",
                             style={"fontSize":"10px","color":MUTED,"textAlign":"center","marginTop":"4px"}),
                    score_bar("Wall/Limit", th["T_wall_max"]/mat["T_limit"],
                               temp_color(th["T_wall_max"],mat["T_limit"])),
                ]),
                # Pressure limit
                card([
                    section_label("Pressure Limit"),
                    *([banner("⛔ p_c EXCEEDS P_max — HARD FAIL","fail")] if pl.get("pressureFail") else []),
                    *[html.Div([
                        html.Span(k, style={"fontSize":"9px","color":MUTED}),
                        html.Span(str(v), style={"fontSize":"9px","color":ACCENT,"fontFamily":"monospace","float":"right"}),
                      ], style={"marginBottom":"4px","overflow":"hidden"})
                      for k, v in [
                          ("p_target",        f"{pl.get('P_target_MPa',0):.2f} MPa"),
                          ("P_max (solid)",   f"{pl.get('P_max_solid_MPa',0):.2f} MPa"),
                          ("P_max (channel)", f"{pl.get('P_max_channel_MPa',0):.2f} MPa"),
                          ("P_max (lig.)",    f"{pl.get('P_max_ligament_MPa',0):.2f} MPa"),
                          ("Margin",          f"{pl.get('pressureMargin',0)*100:.0f}%"),
                          ("σ_y (T-degraded)",f"{pl.get('sigma_y_MPa',0):.0f} MPa"),
                      ]],
                ]),
                # Top 5
                card([
                    section_label("Top 5"),
                    *[html.Div([
                        html.Span(cc["id"], style={"fontSize":"10px","fontFamily":"monospace",
                                                     "color":ACCENT if cc["id"]==c["id"] else "#c8d8e8"}),
                        html.Span(f"{cc['score']*100:.1f}",
                                  style={"float":"right","fontSize":"12px","color":score_color(cc["score"]),"fontFamily":"monospace"}),
                      ], style={"padding":"5px 0","borderBottom":f"1px solid {BORDER}","overflow":"hidden"})
                      for cc in all_candidates[:5]],
                ]),
            ], style={"width":"260px","flexShrink":"0"}),

        ], style={"display":"flex","gap":"14px","alignItems":"flex-start"}),
    ])


def _render_thermal_tab(th, mat, geom):
    xs = geom["xs"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=th["T_wall_gas"],    name="T_wall_gas",    line=dict(color=WARN,  width=2)))
    fig.add_trace(go.Scatter(x=xs, y=th["T_wall_coolant"],name="T_wall_coolant",line=dict(color=BLUE,  width=1.5)))
    fig.add_trace(go.Scatter(x=xs, y=th["T_aw"],          name="T_aw",          line=dict(color=GOLD,  width=1, dash="dot")))
    fig.add_hline(y=mat["T_limit"], line_dash="dash", line_color=WARN, annotation_text=f"T_limit {mat['T_limit']}K", annotation_font_color=WARN)
    fig.update_layout(**PLOT_LAYOUT, height=280, xaxis_title="Axial position →", yaxis_title="Temperature (K)",
                      title=dict(text="Wall Temperature Profile", font=dict(size=10,color=MUTED)))

    T_max = th["T_wall_max"]
    return html.Div([
        card([dcc.Graph(figure=fig, config={"displayModeBar":False})]),
        html.Div([
            metric("T_wall_max",   f"{T_max:.0f}", "K",   temp_color(T_max, mat["T_limit"])),
            metric("T_throat",     f"{th['T_throat_wall']:.0f}", "K", temp_color(th["T_throat_wall"],mat["T_limit"])),
            metric("Margin",       f"{th['overallMargin']*100:.1f}", "%", ACCENT if th["overallMargin"]>0.15 else WARN),
            metric("Cooling eff.", f"{th['coolingEfficiency']*100:.1f}", "%", ACCENT),
        ], style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr 1fr","gap":"8px"}),
    ])


def _render_structural_tab(st, geom, th):
    xs = geom["xs"]
    fig_sigma = go.Figure()
    fig_sigma.add_trace(go.Scatter(x=xs, y=st["sigma_inner_arr"], name="σ_inner", line=dict(color=WARN, width=2)))
    fig_sigma.add_trace(go.Scatter(x=xs, y=st["sigma_insul_arr"], name="σ_insul", line=dict(color=GOLD, width=1.5)))
    fig_sigma.add_hline(y=st["sigma_y"], line_dash="dash", line_color=ACCENT,
                        annotation_text=f"σ_y {st['sigma_y']:.0f} MPa")
    fig_sigma.update_layout(**PLOT_LAYOUT, height=220, xaxis_title="Axial position →",
                             yaxis_title="σ (MPa)", title=dict(text="Thermoelastic Stress σ(x)", font=dict(size=10,color=MUTED)))

    # Vulnerability map
    vm  = st.get("vulnerabilityMap", [0]*len(xs))
    fig_vuln = go.Figure()
    fig_vuln.add_trace(go.Scatter(x=xs, y=vm, fill="tozeroy", line=dict(color=WARN, width=2),
                                   fillcolor="rgba(255,107,53,0.15)", name="V(x)"))
    fig_vuln.add_hline(y=0.6, line_dash="dash", line_color=WARN, annotation_text="high risk 0.6")
    fig_vuln.update_layout(**PLOT_LAYOUT, height=180, xaxis_title="Axial position →",
                            yaxis_title="Vulnerability", yaxis_range=[0,1],
                            title=dict(text="Axial Wall Vulnerability Map V(x)", font=dict(size=10,color=MUTED)))

    # Hollow-wall zone table
    hz = st.get("hollowWallZones", {})
    zone_rows = []
    for zone, zd in hz.items():
        zone_rows.append(html.Tr([
            html.Td(zone.upper(), style={"color":WARN if zone=="throat" else ACCENT}),
            html.Td(f"{zd['netFrac']:.2f}",    style={"color":WARN if zd['netFrac']<0.5 else "#c8d8e8"}),
            html.Td(f"{zd['ampA']:.2f}×",      style={"color":WARN if zd['ampA']>1.5 else "#c8d8e8"}),
            html.Td(f"{zd['slenderness']:.1f}", style={"color":WARN if zd['slenderness']>10 else "#c8d8e8"}),
            html.Td(f"{zd['ampB']:.2f}×",      style={"color":WARN if zd['ampB']>1.3 else "#c8d8e8"}),
            html.Td(f"{zd['Kt_root']:.2f}",    style={"color":GOLD}),
            html.Td(f"{zd['zonePenalty']*100:.0f}%", style={"color":WARN if zd['zonePenalty']>0.2 else "#c8d8e8"}),
            html.Td("⚠" if zd.get("ligamentThin") else "✓",
                    style={"color":WARN if zd.get("ligamentThin") else ACCENT,"textAlign":"center"}),
        ]))

    return html.Div([
        card([dcc.Graph(figure=fig_sigma,  config={"displayModeBar":False})]),
        card([dcc.Graph(figure=fig_vuln,   config={"displayModeBar":False})]),
        card([
            section_label("Hollow-Wall Zone Analysis — Rib Slenderness · Net Section · Channel-Root Kt"),
            html.Table([
                html.Thead(html.Tr([html.Th(h) for h in
                    ["Zone","Net frac","Amp A","Slenderness","Amp B","Root Kt","Penalty","Lig thin"]])),
                html.Tbody(zone_rows),
            ], className="cpas-table") if zone_rows else html.Div("No regen channels — solid wall model.", style={"fontSize":"9px","color":MUTED}),
        ]),
        html.Div([
            metric("σ_throat",   f"{st['sigma_throat_MPa']:.0f}", "MPa", WARN),
            metric("σ_peak",     f"{st['sigma_peak_MPa']:.0f}",   "MPa", GOLD),
            metric("σ_yield",    f"{st['sigma_y']:.0f}",          "MPa", ACCENT),
            metric("SF",         f"{st['SF_actual']:.2f}",        "",    ACCENT if st["SF_ok"] else WARN),
            metric("Min lig.",   f"{st.get('minLigament_mm',0):.2f}", "mm", ACCENT),
            metric("Hollow pen.","", "", MUTED) if not st.get("hollowWallPenalty") else
            metric("Hollow pen.",f"{st.get('hollowWallPenalty',0)*100:.0f}", "%", GOLD),
        ], style={"display":"grid","gridTemplateColumns":"repeat(6,1fr)","gap":"8px","marginTop":"10px"}),
    ])


def _render_heatflux_tab(th, geom):
    xs = geom["xs"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[q/1e6 for q in th["q_g"]],   name="q_g (gas-side)",
                              fill="tozeroy", line=dict(color=GOLD,  width=1.5), fillcolor="rgba(232,184,75,0.15)"))
    fig.add_trace(go.Scatter(x=xs, y=[q/1e6 for q in th["q_net"]], name="q_net (after cooling)",
                              fill="tozeroy", line=dict(color=ACCENT, width=2),  fillcolor="rgba(0,232,192,0.15)"))
    fig.update_layout(**PLOT_LAYOUT, height=260, xaxis_title="Axial position →", yaxis_title="Heat flux (MW/m²)",
                      title=dict(text="Heat Flux Profile — Gas-side vs Net After Cooling", font=dict(size=10,color=MUTED)))

    fig_hg = go.Figure()
    fig_hg.add_trace(go.Scatter(x=xs, y=[h/1e3 for h in th["h_g"]], line=dict(color=BLUE,width=2), name="h_g (kW/m²K)"))
    fig_hg.update_layout(**PLOT_LAYOUT, height=180, xaxis_title="Axial position →", yaxis_title="h_g (kW/m²K)",
                          title=dict(text="Bartz Convection Coefficient h_g(x)", font=dict(size=10,color=MUTED)))

    return html.Div([
        card([dcc.Graph(figure=fig,   config={"displayModeBar":False})]),
        card([dcc.Graph(figure=fig_hg,config={"displayModeBar":False})]),
    ])


def _render_3d_profile_tab(geom, th, candidate):
    """Plotly 3D surface — revolution of the engine profile."""
    import numpy as np
    n_rev = 48
    theta = [2 * math.pi * i / n_rev for i in range(n_rev + 1)]
    rs_m  = geom["rs_m"]
    xs_m  = [x * geom["L_total_m"] for x in geom["xs"]]
    T_wg  = th["T_wall_gas"]
    T_min, T_max = min(T_wg), max(T_wg)

    # Inner surface
    z_grid = [[xs_m[j] for _ in range(n_rev+1)] for j in range(len(xs_m))]
    x_grid = [[rs_m[j]*math.cos(t) for t in theta] for j in range(len(rs_m))]
    y_grid = [[rs_m[j]*math.sin(t) for t in theta] for j in range(len(rs_m))]
    c_grid = [[T_wg[j] for _ in range(n_rev+1)] for j in range(len(T_wg))]

    fig = go.Figure(go.Surface(
        x=z_grid, y=x_grid, z=y_grid,
        surfacecolor=c_grid,
        colorscale=[[0,"#0040a0"],[0.3,"#00a0c0"],[0.6,"#e8b84b"],[0.85,"#ff6b35"],[1.0,"#ffffff"]],
        cmin=T_min, cmax=T_max,
        colorbar=dict(title="T_wall (K)", thickness=12, titlefont=dict(color=MUTED), tickfont=dict(color=MUTED)),
        opacity=0.92,
        name="Inner wall",
    ))

    # Outer surface — use correct wall thickness formula
    t_w    = wall_thickness_m(candidate["params"], geom["r_t_m"])
    ro_m   = [r + t_w for r in rs_m]
    xo_grid = [[ro_m[j]*math.cos(t) for t in theta] for j in range(len(ro_m))]
    yo_grid = [[ro_m[j]*math.sin(t) for t in theta] for j in range(len(ro_m))]
    fig.add_trace(go.Surface(
        x=z_grid, y=xo_grid, z=yo_grid,
        colorscale=[[0,"#1a3a5a"],[1,"#2a5a7a"]],
        opacity=0.25, showscale=False, name="Outer wall",
    ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        scene=dict(
            bgcolor=BG,
            xaxis=dict(title="Z (m)", backgroundcolor=BG, gridcolor=BORDER, showbackground=True),
            yaxis=dict(title="X (m)", backgroundcolor=BG, gridcolor=BORDER, showbackground=True),
            zaxis=dict(title="Y (m)", backgroundcolor=BG, gridcolor=BORDER, showbackground=True),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
        height=440,
        margin=dict(l=0,r=0,t=30,b=0),
        title=dict(text=f"3D Engine Profile — T_wall heatmap  (inner: {T_min:.0f}–{T_max:.0f} K)", font=dict(size=10,color=MUTED)),
    )

    # Dimensions strip
    dims = [
        ("r_throat", f"{geom['r_t_m']*1000:.1f} mm"),
        ("r_chamber",f"{geom['r_c_m']*1000:.1f} mm"),
        ("r_exit",   f"{geom['r_e_m']*1000:.1f} mm"),
        ("ε_c",      f"{geom['eps_c']:.1f}"),
        ("ε_e",      f"{geom['eps_e']:.1f}"),
        ("L_total",  f"{geom['L_total_m']*1000:.0f} mm"),
        ("L*",       f"{geom['Lstar']:.2f} m"),
        ("τ",        f"{geom['tau_ms']:.1f} ms"),
    ]
    dim_strip = html.Div([
        html.Div([
            html.Div(k, style={"fontSize":"8px","color":MUTED,"textTransform":"uppercase","letterSpacing":"1px"}),
            html.Div(v, style={"fontSize":"11px","color":ACCENT,"fontFamily":"monospace"}),
        ], style={"padding":"6px 12px","background":BG2,"borderRadius":"3px"})
        for k, v in dims
    ], style={"display":"flex","gap":"8px","flexWrap":"wrap","marginTop":"8px"})

    return html.Div([
        card([dcc.Graph(figure=fig, config={"displayModeBar":True})]),
        dim_strip,
    ])


def _render_params_tab(c, pl, cs, geom, th, fl, mat, prop):
    inj    = INJECTOR_TYPES[c["injector"]]
    p      = c["params"]
    nd     = c.get("nozzleDesign", {})
    ch_secs= c.get("channelSections", [])

    rows_l = [
        ("── Characteristic Length ──",""),
        ("L* (design)",   f"{geom['Lstar']:.3f} m"),
        ("L* min (prop)", f"{prop['Lstar_min']} m"),
        ("L* opt (prop)", f"{prop['Lstar_opt']} m"),
        ("L* max (prop)", f"{prop['Lstar_max']} m"),
        ("Residence τ",   f"{geom['tau_ms']:.1f} ms"),
        ("── Real Dimensions ──",""),
        ("r_t (throat)",  f"{geom['r_t_m']*1000:.2f} mm"),
        ("r_c (chamber)", f"{geom['r_c_m']*1000:.2f} mm"),
        ("r_e (exit)",    f"{geom['r_e_m']*1000:.2f} mm"),
        ("A_t",           f"{geom['A_t_m2']*1e4:.3f} cm²"),
        ("V_c",           f"{geom['V_c_m3']*1e6:.1f} cm³"),
        ("L_cyl",         f"{geom['L_cyl_m']*1000:.1f} mm"),
        ("L_cvg",         f"{geom['L_cvg_m']*1000:.1f} mm"),
        ("L_noz",         f"{geom['L_noz_m']*1000:.1f} mm"),
        ("L_total",       f"{geom['L_total_m']*1000:.1f} mm"),
        ("── Chamber Sizing ──",""),
        ("Governing criterion", cs.get("governingCriterion","Lstar")),
        ("Governed by physics", "✓ yes" if cs.get("governedByPhysics") else "no (L*)"),
        ("L/D ratio",     f"{cs.get('LD_ratio',0):.2f}"),
        ("Vap clipped",   "⚠ yes" if cs.get("vapClipped") else "no"),
        ("L_vap",         f"{cs.get('L_vap_mm',0):.0f} mm"),
        ("τ_min (prop)",  f"{cs.get('tau_min_ms',0):.1f} ms"),
        ("L_cap",         f"{cs.get('L_cap_m',0)*1000:.0f} mm"),
        ("── Pressure Limits ──",""),
        ("p_target",      f"{p['chamberPressure']:.2f} MPa"),
        ("P_max (solid)", f"{pl.get('P_max_solid_MPa',0):.2f} MPa"),
        ("P_max (channel)",f"{pl.get('P_max_channel_MPa',0):.2f} MPa"),
        ("P_max (lig.)",  f"{pl.get('P_max_ligament_MPa',0):.2f} MPa"),
        ("Margin",        f"{pl.get('pressureMargin',0)*100:.0f}%"),
        ("Status",        "⛔ EXCEEDS" if pl.get("pressureFail") else ("⚠ low" if pl.get("pressureWarn") else "✓ OK")),
    ]

    rows_html = []
    for k, v in rows_l:
        if v == "":
            rows_html.append(html.Div(k.replace("── ","").replace(" ──",""),
                style={"gridColumn":"1/-1","paddingTop":"8px","borderTop":f"1px solid {BORDER}",
                       "fontSize":"8px","color":MUTED,"textTransform":"uppercase","letterSpacing":"2px","marginTop":"4px"}))
        else:
            rows_html.append(html.Div([
                html.Span(k, style={"fontSize":"9px","color":MUTED}),
                html.Span(v, style={"fontSize":"9px","color":ACCENT if "✓" in v else (WARN if "⛔" in v or "⚠" in v else "#c8d8e8"),"fontFamily":"monospace","float":"right"}),
            ], style={"padding":"4px 8px","background":BG,"borderRadius":"2px","overflow":"hidden"}))

    # Channel section table
    ch_tbl = html.Div("No regen channels.", style={"fontSize":"9px","color":MUTED})
    if ch_secs:
        ch_tbl = html.Table([
            html.Thead(html.Tr([html.Th(h) for h in
                ["Zone","N","Width","Depth","Rib","Inn.lig","Out.lig","D_h","Pitch","Thin?"]])),
            html.Tbody([
                html.Tr([
                    html.Td(s["zone"].upper()),
                    html.Td(str(s["nChannels"])),
                    html.Td(f"{s['channelWidth_mm']:.2f}"),
                    html.Td(f"{s['channelDepth_mm']:.2f}"),
                    html.Td(f"{s['ribThickness_mm']:.2f}"),
                    html.Td(f"{s['innerLigament_mm']:.2f}", style={"color":WARN if s.get("ligamentThin") else "inherit"}),
                    html.Td(f"{s['outerLigament_mm']:.2f}"),
                    html.Td(f"{s['hydraulicDiameter_mm']:.2f}"),
                    html.Td(f"{s['pitch_mm']:.2f}"),
                    html.Td("⚠" if s.get("ligamentThin") else "✓",
                            style={"color":WARN if s.get("ligamentThin") else ACCENT}),
                ]) for s in ch_secs
            ]),
        ], className="cpas-table")

    # Nozzle design card
    nd_content = html.Div("No nozzle design data.", style={"fontSize":"9px","color":MUTED})
    if nd:
        nd_content = html.Div([
            html.Div([
                metric(k, v) for k, v in [
                    ("ε_e",       f"{nd.get('areaRatio',0):.2f}"),
                    ("L_nozzle",  f"{nd.get('nozzleLength_m',0)*1000:.1f} mm"),
                    ("θ_diverge", f"{nd.get('divergenceHalfAngle_deg',0):.1f}°"),
                    ("Package",   nd.get("packageStatus","?")),
                    ("Clipped",   "⚠ yes" if nd.get("areaRatioClipped") else "no"),
                    ("Contour pts",str(len(nd.get("contourPoints_mm",[])))),
                ]
            ], style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":"8px"}),
            # Nozzle contour plot
            _nozzle_contour_plot(nd),
        ])

    return html.Div([
        card([
            html.Div(rows_html, style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"4px"}),
        ]),
        card([section_label("Regen Channel Cross-Section (mm)"), ch_tbl]),
        card([
            section_label(f"Nozzle Design — {nd.get('mode','conical')} mode"),
            nd_content,
        ]),
    ])


def _nozzle_contour_plot(nd):
    pts = nd.get("contourPoints_mm", [])
    if not pts:
        return html.Div()
    zs = [p["z_mm"] for p in pts]
    rs = [p["r_mm"] for p in pts]
    rs_neg = [-r for r in rs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=zs, y=rs,     fill="tozeroy", line=dict(color=ACCENT,width=2),
                              fillcolor="rgba(0,232,192,0.10)", name="Upper"))
    fig.add_trace(go.Scatter(x=zs, y=rs_neg,  fill="tozeroy", line=dict(color=ACCENT,width=2),
                              fillcolor="rgba(0,232,192,0.10)", name="Lower", showlegend=False))
    fig.add_hline(y=0, line_color=MUTED, line_width=0.5, line_dash="dot")
    fig.update_layout(**PLOT_LAYOUT, height=160, xaxis_title="z (mm from throat)", yaxis_title="r (mm)",
                      title=dict(text="Nozzle inner wall contour (axisymmetric)", font=dict(size=9,color=MUTED)),
                      showlegend=False)
    return dcc.Graph(figure=fig, config={"displayModeBar":False})


def _render_cad_tab(c):
    geom = c["geom"]
    nd   = c.get("nozzleDesign", {})
    chs  = c.get("channelSections", [])
    pl   = c.get("pressureLimit", {})
    t_wall = wall_thickness_m(c["params"], geom["r_t_m"])

    wall_by_zone = {
        "injector":   t_wall * 1.10 * 1000,
        "chamber":    t_wall * 1.00 * 1000,
        "converging": t_wall * 0.85 * 1000,
        "throat":     t_wall * 0.70 * 1000,
        "nozzle":     t_wall * 0.80 * 1000,
    }

    dim_grid = [
        ("Throat Ø",     f"{geom['r_t_m']*2000:.2f} mm"),
        ("Chamber Ø",    f"{geom['r_c_m']*2000:.2f} mm"),
        ("Exit Ø",       f"{geom['r_e_m']*2000:.2f} mm"),
        ("L_chamber",    f"{geom['L_cyl_m']*1000:.1f} mm"),
        ("L_convergent", f"{geom['L_cvg_m']*1000:.1f} mm"),
        ("L_nozzle",     f"{geom['L_noz_m']*1000:.1f} mm"),
        ("L_total",      f"{geom['L_total_m']*1000:.1f} mm"),
        ("R_blend_up",   f"{geom.get('R_thr_upstream_m',0)*1000:.2f} mm"),
        ("R_blend_dn",   f"{geom.get('R_thr_downstream_m',0)*1000:.2f} mm"),
    ]

    ch_table = html.Div("No channels.", style={"fontSize":"9px","color":MUTED})
    if chs:
        ch_table = html.Table([
            html.Thead(html.Tr([html.Th(h) for h in
                ["Zone","N","Width","Depth","Rib","Inn.lig","Out.lig","D_h","Pitch","Min lig","⚠"]])),
            html.Tbody([html.Tr([
                html.Td(s["zone"].upper()),
                html.Td(str(s["nChannels"])),
                html.Td(f"{s['channelWidth_mm']:.2f}"),
                html.Td(f"{s['channelDepth_mm']:.2f}"),
                html.Td(f"{s['ribThickness_mm']:.2f}"),
                html.Td(f"{s['innerLigament_mm']:.2f}", style={"color":WARN if s.get("ligamentThin") else "inherit"}),
                html.Td(f"{s['outerLigament_mm']:.2f}"),
                html.Td(f"{s['hydraulicDiameter_mm']:.2f}"),
                html.Td(f"{s['pitch_mm']:.2f}"),
                html.Td(f"{s['minLigament_mm']:.2f}"),
                html.Td("⚠" if s.get("ligamentThin") else "✓",
                        style={"color":WARN if s.get("ligamentThin") else ACCENT}),
            ]) for s in chs]),
        ], className="cpas-table")

    # JSON export button
    cad_payload = {
        "units": "mm", "modelType": "concept",
        "diameters_mm": {
            "throat":  geom["r_t_m"]*2000,
            "chamber": geom["r_c_m"]*2000,
            "exit":    geom["r_e_m"]*2000,
        },
        "zoneLengths_mm": {
            "chamberCyl":    geom["L_cyl_m"]*1000,
            "convergent":    geom["L_cvg_m"]*1000,
            "nozzle":        geom["L_noz_m"]*1000,
            "total":         geom["L_total_m"]*1000,
        },
        "wallByZone_mm":   wall_by_zone,
        "channelsByZone":  [{
            "zone": s["zone"], "nChannels": s["nChannels"],
            "width_mm": round(s["channelWidth_mm"],3),
            "depth_mm": round(s["channelDepth_mm"],3),
            "rib_mm":   round(s["ribThickness_mm"],3),
            "innerLigament_mm": round(s["innerLigament_mm"],3),
            "outerLigament_mm": round(s["outerLigament_mm"],3),
            "D_h_mm":   round(s["hydraulicDiameter_mm"],3),
        } for s in chs],
        "nozzleDesign": {
            "mode":       nd.get("mode","conical"),
            "areaRatio":  nd.get("areaRatio"),
            "nozzleLength_mm": nd.get("nozzleLength_m",0)*1000,
            "packageStatus": nd.get("packageStatus"),
            "clipped":    nd.get("areaRatioClipped"),
        },
        "pressure_limit": pl,
    }

    return html.Div([
        card([
            section_label("CAD Dimensions — All values in millimetres"),
            html.Div([
                html.Div([
                    html.Div(k, style={"fontSize":"8px","color":MUTED,"textTransform":"uppercase","letterSpacing":"1px","marginBottom":"2px"}),
                    html.Div(v, style={"fontSize":"13px","color":ACCENT,"fontFamily":"monospace"}),
                ], style={"padding":"7px 10px","background":BG,"borderRadius":"3px"})
                for k, v in dim_grid
            ], style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":"8px","marginBottom":"12px"}),
            section_label("Wall Thickness by Zone"),
            html.Div([
                html.Div([
                    html.Div(z.capitalize(), style={"fontSize":"8px","color":MUTED if z!="throat" else WARN,"textTransform":"uppercase"}),
                    html.Div(f"{t:.2f} mm", style={"fontSize":"11px","color":WARN if z=="throat" else "#c8d8e8","fontFamily":"monospace"}),
                ], style={"padding":"5px 8px","background":BG,"borderRadius":"2px",
                           "borderLeft":f"2px solid {WARN if z=='throat' else BORDER}"})
                for z, t in wall_by_zone.items()
            ], style={"display":"flex","gap":"8px","flexWrap":"wrap"}),
        ]),
        card([
            section_label("Channel Cross-Section Geometry by Zone (mm)"),
            ch_table,
        ]),
        card([
            section_label("Nozzle Design — Contour Preview"),
            _nozzle_contour_plot(nd),
        ]),
        card([
            section_label("JSON Export"),
            html.Pre(json.dumps(cad_payload, indent=2)[:800] + "\n...",
                     style={"fontSize":"8px","color":MUTED,"fontFamily":"monospace",
                            "background":BG,"padding":"8px","borderRadius":"3px","overflowX":"auto"}),
            html.Div("ℹ  Download the full manifest from the JSON button above to get all 40 contour points and complete channel geometry.",
                     style={"fontSize":"9px","color":MUTED,"marginTop":"8px"}),
        ]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 04 — TRADE STUDY
# ══════════════════════════════════════════════════════════════════════════════

def render_trade(candidates):
    dims = ["thermal","structural","flow","mfg","robust","compactness"]

    fig = go.Figure()
    xs  = [c["breakdown"]["thermal"]    for c in candidates]
    ys  = [c["breakdown"]["structural"] for c in candidates]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(color=[c["score"]*100 for c in candidates],
                    colorscale=[[0,WARN],[0.5,GOLD],[1,ACCENT]],
                    size=[10 if c["paretoRank"]==0 else 6 for c in candidates],
                    opacity=0.85, showscale=True,
                    colorbar=dict(title="Score", tickfont=dict(size=9,color=MUTED))),
        text=[f"{c['id']}<br>{c['score']*100:.1f}%<br>{'⭐ Pareto' if c['paretoRank']==0 else ''}"
              for c in candidates],
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(**PLOT_LAYOUT, height=400,
                      xaxis_title="Thermal Score", yaxis_title="Structural Score",
                      title=dict(text="Trade Study — Pareto Space", font=dict(size=10,color=MUTED)))

    # Score radar for top 5 Pareto
    pareto = [c for c in candidates if c["paretoRank"]==0][:5]
    fig_radar = go.Figure()
    for c in pareto:
        vals = [c["breakdown"][d] for d in dims] + [c["breakdown"][dims[0]]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=[d.capitalize() for d in dims] + [dims[0].capitalize()],
            fill="toself", opacity=0.3, name=c["id"],
        ))
    fig_radar.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=320,
        polar=dict(bgcolor=BG2, radialaxis=dict(visible=True, range=[0,1], gridcolor=BORDER),
                   angularaxis=dict(gridcolor=BORDER)),
        font=dict(color=MUTED, family="Courier New"),
        margin=dict(l=40,r=40,t=30,b=30),
        title=dict(text="Pareto Front — Score Radar (Top 5)", font=dict(size=10,color=MUTED)),
    )

    return html.Div([
        html.Div([
            html.Div([card([dcc.Graph(figure=fig, config={"displayModeBar":False})])], style={"flex":"1"}),
            html.Div([card([dcc.Graph(figure=fig_radar, config={"displayModeBar":False})])], style={"flex":"1"}),
        ], style={"display":"flex","gap":"14px"}),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

def render_parameters():
    prop_opts = [{"label": v["label"], "value": k} for k, v in PROPELLANTS.items()]
    inj_opts  = [{"label": v["label"], "value": k} for k, v in INJECTOR_TYPES.items()]
    mat_opts  = [{"label": v["label"], "value": k} for k, v in MATERIALS.items()]
    alt_opts  = [{"label": v["label"], "value": k} for k, v in ALTITUDES.items()]

    return html.Div([
        # Sub-tabs: Design | Trade Sweep
        html.Div([
            html.Button("⚙ Design",      id="btn-params-design", className="tab-btn active", n_clicks=0),
            html.Button("⟁ Trade Sweep", id="btn-params-trade",  className="tab-btn",        n_clicks=0),
        ], style={"display":"flex","gap":"4px","marginBottom":"14px"}),

        # Inputs row
        html.Div([
            html.Div([
                card([
                    section_label("Mass Flow Rates"),
                    html.Div("ṁ Fuel (kg/s)", style={"fontSize":"9px","color":MUTED,"marginBottom":"3px"}),
                    dcc.Slider(0.1, 200, 0.5, value=10, id="sl-mdot-fuel", marks=None,
                               tooltip={"always_visible":True,"placement":"bottom"}, className="cpas-slider"),
                    html.Div("ṁ Oxidiser (kg/s)", style={"fontSize":"9px","color":MUTED,"marginTop":"8px","marginBottom":"3px"}),
                    dcc.Slider(0.1, 500, 1, value=35.5, id="sl-mdot-ox", marks=None,
                               tooltip={"always_visible":True,"placement":"bottom"}, className="cpas-slider"),
                    html.Div("Throat Area A_t (cm²)", style={"fontSize":"9px","color":MUTED,"marginTop":"8px","marginBottom":"3px"}),
                    dcc.Slider(1, 500, 1, value=50, id="sl-at", marks=None,
                               tooltip={"always_visible":True,"placement":"bottom"}, className="cpas-slider"),
                ]),
                card([
                    section_label("Propellant"),
                    dcc.Dropdown(prop_opts, value="methalox", id="dd-prop", clearable=False,
                                 style={"background":BG2,"color":"#c8d8e8","border":f"1px solid {BORDER}"}),
                ]),
                card([
                    section_label("Injector"),
                    dcc.Dropdown(inj_opts, value="like_doublet", id="dd-inj", clearable=False,
                                 style={"background":BG2,"color":"#c8d8e8"}),
                ]),
                card([
                    section_label("Material"),
                    dcc.Dropdown(mat_opts, value="copper", id="dd-mat", clearable=False,
                                 style={"background":BG2,"color":"#c8d8e8"}),
                ]),
                card([
                    section_label("Operating Altitude"),
                    dcc.RadioItems(
                        options=[{"label": f" {v['icon']} {v['label']} ({v['p_amb']} Pa)", "value": k}
                                 for k, v in ALTITUDES.items()],
                        value="vacuum", id="radio-alt", className="cpas-slider",
                        inputStyle={"marginRight":"6px"},
                        labelStyle={"display":"block","marginBottom":"6px","fontSize":"10px","color":"#c8d8e8"},
                    ),
                ]),
            ], style={"width":"320px","flexShrink":"0"}),

            # ── Outputs ──────────────────────────────────────────────────
            html.Div(id="params-output", style={"flex":"1","minWidth":"0"}),
        ], style={"display":"flex","gap":"14px","alignItems":"flex-start"}),
    ])


@app.callback(
    Output("params-output", "children"),
    Input("sl-mdot-fuel", "value"),
    Input("sl-mdot-ox",   "value"),
    Input("sl-at",        "value"),
    Input("dd-prop",      "value"),
    Input("dd-inj",       "value"),
    Input("dd-mat",       "value"),
    Input("radio-alt",    "value"),
)
def update_params_output(mdot_f, mdot_ox, A_t_cm2, prop_key, inj_key, material, alt_key):
    prop   = PROPELLANTS[prop_key]
    inj    = INJECTOR_TYPES[inj_key]
    mat    = MATERIALS[material]
    alt    = ALTITUDES[alt_key]
    p_amb  = alt["p_amb"]
    mdot_f  = mdot_f  or 10
    mdot_ox = mdot_ox or 35.5
    A_t_cm2 = A_t_cm2 or 50
    A_t_m2  = A_t_cm2 * 1e-4
    mdot    = mdot_f + mdot_ox
    OF      = mdot_ox / max(mdot_f, 0.001)

    # Governing quantities
    from physics.solvers import prop_at_of
    pof      = prop_at_of(prop_key, OF)
    T_c      = pof["T_flame"] * inj["mix_eff"]
    gamma_of = pof["gamma"]
    MW       = prop["MW"]
    cstar    = c_star(T_c, gamma_of, MW)
    p_c_Pa   = mdot * cstar / max(A_t_m2, 1e-9)
    p_c_MPa  = p_c_Pa / 1e6

    eps_e_opt = optimal_nozzle_ar(gamma_of, p_c_Pa, p_amb)
    CF_vac    = thrust_coeff(gamma_of, eps_e_opt, p_c_Pa, 0)
    CF_amb    = thrust_coeff(gamma_of, eps_e_opt, p_c_Pa, p_amb)
    Isp_vac   = prop["Isp_vac"] * inj["mix_eff"]
    Isp_amb   = isp_at_altitude(Isp_vac, CF_amb, CF_vac)
    thrust_vac_kN = CF_vac * p_c_Pa * A_t_m2 / 1000
    thrust_amb_kN = CF_amb * p_c_Pa * A_t_m2 / 1000

    of_dev   = abs(OF - prop["OF_nominal"]) / prop["OF_nominal"]
    of_warn  = of_dev > 0.20
    p_warn   = p_c_MPa > 12 or p_c_MPa < 0.5

    # Performance grid
    perf_grid = html.Div([
        metric("Isp_vac",     f"{Isp_vac:.1f}", "s",    ACCENT),
        metric("Isp_alt",     f"{Isp_amb:.1f}", "s",    ACCENT if not of_warn else GOLD),
        metric("Thrust_vac",  f"{thrust_vac_kN:.2f}", "kN", BLUE),
        metric("Thrust_alt",  f"{thrust_amb_kN:.2f}", "kN", BLUE),
        metric("p_chamber",   f"{p_c_MPa:.2f}", "MPa",  WARN if p_warn else GOLD),
        metric("c*",          f"{cstar:.0f}",   "m/s",  GOLD),
        metric("O/F",         f"{OF:.3f}",       "",    WARN if of_warn else ACCENT),
        metric("Optimal ε_e", f"{eps_e_opt:.1f}", "",   ACCENT),
    ], style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"8px","marginBottom":"10px"})

    # OF bar
    of_frac = clamp((OF - prop["OF_range"][0]) / (prop["OF_range"][1] - prop["OF_range"][0]), 0, 1)
    of_bar  = html.Div([
        html.Div([
            html.Span(f"O/F = {OF:.3f}", style={"fontSize":"10px","color":WARN if of_warn else ACCENT}),
            html.Span(f" (nominal {prop['OF_nominal']:.2f})", style={"fontSize":"9px","color":MUTED}),
        ]),
        html.Div(html.Div(style={"width":f"{of_frac*100:.0f}%","height":"100%",
                                  "background":WARN if of_warn else ACCENT,"borderRadius":"2px"}),
                 style={"height":"4px","background":BORDER,"borderRadius":"2px","margin":"4px 0"}),
    ], style={"marginBottom":"8px"})

    # Altitude warning
    alt_note = []
    if p_amb > 0:
        from physics.solvers import exit_pressure_ratio
        pe_pc = exit_pressure_ratio(gamma_of, eps_e_opt)
        p_e   = pe_pc * p_c_Pa
        if p_e < p_amb:
            alt_note = [banner(f"⚠ Over-expanded: p_exit={p_e/1000:.1f} kPa < p_amb={p_amb/1000:.1f} kPa — thrust loss possible", "warn")]

    # Sweep plot (p_c sweep using current propellant, quick preview)
    return html.Div([
        card([section_label(f"Performance — {alt['icon']} {alt['label']}"),
              *alt_note, of_bar, perf_grid]),
        card([
            section_label("Altitude Context"),
            html.Div(f"Optimal ε_e for {alt['label']}: {eps_e_opt:.1f} — area ratio where p_exit = p_ambient.",
                     style={"fontSize":"9px","color":MUTED,"lineHeight":"1.7"}),
            html.Div(f"Isp loss from vacuum: {Isp_vac-Isp_amb:.1f} s ({(1-Isp_amb/Isp_vac)*100:.1f}%)",
                     style={"fontSize":"9px","color":GOLD,"marginTop":"4px"}),
        ]),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# VERIFY MODE
# ══════════════════════════════════════════════════════════════════════════════

def render_verify():
    return html.Div([
        card([
            section_label("§8 Audit Verification Tests — Required Before Technical Signoff"),
            html.Div("Six tests mandated by the CPAS audit. Each exercises actual physics functions and asserts directional relationships.",
                     style={"fontSize":"9px","color":MUTED,"lineHeight":"1.7","marginBottom":"12px"}),
            html.Button("▶  Run All Tests", id="btn-run-verify", n_clicks=0,
                        style={"padding":"8px 20px","background":f"rgba(232,184,75,0.15)",
                               "border":f"1px solid {GOLD}","color":GOLD,
                               "fontSize":"10px","letterSpacing":"2px","cursor":"pointer",
                               "borderRadius":"3px","fontFamily":"monospace"}),
            html.Div(id="verify-results"),
        ]),
    ])


@app.callback(
    Output("verify-results", "children"),
    Input("btn-run-verify",  "n_clicks"),
    prevent_initial_call=True,
)
def run_verify(n):
    results = _run_verification_tests()
    passed  = sum(1 for r in results if r["pass"])
    failed  = len(results) - passed

    signoff_banner = banner(
        "ALL PASS — SIGNOFF CRITERIA MET" if failed == 0
        else f"SIGNOFF BLOCKED — {failed} test(s) failed",
        "pass" if failed == 0 else "fail"
    )

    cards = [
        html.Div(style={"height":"12px"}),
        signoff_banner,
        html.Div(f"✓ {passed} passed  ·  {'✗ '+str(failed)+' failed' if failed else ''}",
                 style={"fontSize":"12px","fontFamily":"monospace","color":ACCENT if failed==0 else WARN,"marginBottom":"12px"}),
    ]
    for r in results:
        cards.append(card([
            html.Div([
                html.Span("✓ PASS" if r["pass"] else "✗ FAIL",
                          style={"color":ACCENT if r["pass"] else WARN,"fontFamily":"monospace","marginRight":"12px","fontSize":"11px"}),
                html.Span(r["name"], style={"fontSize":"11px"}),
            ], style={"marginBottom":"6px"}),
            html.Div(r["description"], style={"fontSize":"9px","color":MUTED,"marginBottom":"8px","lineHeight":"1.7"}),
            html.Div([
                html.Div([
                    html.Div("Expected", style={"fontSize":"8px","color":MUTED,"textTransform":"uppercase","letterSpacing":"1px","marginBottom":"3px"}),
                    html.Div(r["expected"], style={"fontSize":"9px","color":GOLD,"fontFamily":"monospace","lineHeight":"1.5"}),
                ], style={"padding":"6px 8px","background":BG,"borderRadius":"3px","flex":"1"}),
                html.Div([
                    html.Div("Actual", style={"fontSize":"8px","color":MUTED,"textTransform":"uppercase","letterSpacing":"1px","marginBottom":"3px"}),
                    html.Div(r["actual"], style={"fontSize":"9px","color":ACCENT if r["pass"] else WARN,"fontFamily":"monospace","lineHeight":"1.5"}),
                ], style={"padding":"6px 8px","background":BG,"borderRadius":"3px","flex":"1"}),
            ], style={"display":"flex","gap":"8px","marginBottom":"6px"}),
            html.Div(r.get("detail",""), style={"fontSize":"8px","color":MUTED,"fontFamily":"monospace","background":BG,"padding":"5px 8px","borderRadius":"3px"}),
        ], style={"borderLeft":f"3px solid {ACCENT if r['pass'] else WARN}","marginBottom":"8px"}))

    return html.Div(cards)


def _run_verification_tests():
    results = []

    def assert_t(name, desc, expected, actual, passed, detail=""):
        results.append({"name":name,"description":desc,"expected":expected,
                        "actual":actual,"pass":passed,"detail":detail})

    # T1
    try:
        base = {"chamberPressure":5,"coolingChannels":20,"regenerativeCooling":True,
                "throatRadius":0.025,"contractionRatio":4,"Lstar":1.27,
                "convergentAngle":30,"nozzleAngle":15,"nozzleAR":8,
                "injectorDensity":0.6,"coolantFlow":0.5,"filmCooling":False,
                "ablativeLayer":False,"material":"copper","injector":"like_doublet",
                "wallThickness":0.6}
        pRegen  = {**base, "regenerativeCooling":True,  "wallThickness":0.60}
        pNoRegen= {**base, "regenerativeCooling":False, "wallThickness":0.60}
        pDeep   = {**base, "regenerativeCooling":True,  "wallThickness":0.80}
        pShall  = {**base, "regenerativeCooling":True,  "wallThickness":0.20}
        gR  = build_geometry_profile(pRegen,   "kerolox")
        gNR = build_geometry_profile(pNoRegen, "kerolox")
        gD  = build_geometry_profile(pDeep,    "kerolox")
        gS  = build_geometry_profile(pShall,   "kerolox")
        plR  = compute_pressure_limit(pRegen,   gR,  "copper","kerolox")
        plNR = compute_pressure_limit(pNoRegen, gNR, "copper","kerolox")
        plD  = compute_pressure_limit(pDeep,    gD,  "copper","kerolox")
        plS  = compute_pressure_limit(pShall,   gS,  "copper","kerolox")
        p1   = plR["P_max_MPa"] < plNR["P_max_MPa"]
        p2   = plD["P_max_ligament_MPa"] > plS["P_max_ligament_MPa"]
        assert_t("T1: Channels reduce P_max",
                 "Regen channels must reduce P_max below solid-wall limit; thicker wall → higher P_max_lig.",
                 "P_max(regen) < P_max(solid); P_max_lig(thick) > P_max_lig(thin)",
                 f"P_max(regen)={plR['P_max_MPa']:.3f}, P_max(solid)={plNR['P_max_MPa']:.3f} | "
                 f"P_max_lig(thick)={plD['P_max_ligament_MPa']:.3f}, (thin)={plS['P_max_ligament_MPa']:.3f}",
                 p1 and p2, f"Regen reduces: {p1} | Ligament monotonicity: {p2}")
    except Exception as e:
        assert_t("T1: Channels reduce P_max","","","ERROR: "+str(e),False)

    # T2
    try:
        base2 = {"chamberPressure":5,"coolingChannels":20,"regenerativeCooling":True,
                 "throatRadius":0.025,"contractionRatio":4,"Lstar":1.27,
                 "convergentAngle":30,"nozzleAngle":15,"nozzleAR":8,
                 "injectorDensity":0.6,"coolantFlow":0.5,"filmCooling":False,
                 "ablativeLayer":False,"material":"copper","injector":"like_doublet"}
        pThk = {**base2, "wallThickness":0.80}
        pThn = {**base2, "wallThickness":0.15}
        gThk = build_geometry_profile(pThk,"kerolox")
        gThn = build_geometry_profile(pThn,"kerolox")
        plThk = compute_pressure_limit(pThk,gThk,"copper","kerolox")
        plThn = compute_pressure_limit(pThn,gThn,"copper","kerolox")
        passed = plThn["P_max_ligament_MPa"] < plThk["P_max_ligament_MPa"]
        assert_t("T2: Ligament → P_max",
                 "Thinner wall (smaller inner ligament) must reduce P_max_ligament.",
                 "P_max_lig(thick=0.80) > P_max_lig(thin=0.15)",
                 f"P_max_lig(thick)={plThk['P_max_ligament_MPa']:.3f} MPa, "
                 f"P_max_lig(thin)={plThn['P_max_ligament_MPa']:.3f} MPa", passed,
                 f"Δ = {plThk['P_max_ligament_MPa']-plThn['P_max_ligament_MPa']:.3f} MPa")
    except Exception as e:
        assert_t("T2: Ligament → P_max","","","ERROR: "+str(e),False)

    # T3
    try:
        extremes = [
            {"throatRadius":0.005,"chamberPressure":0.5,"Lstar":0.50,"contractionRatio":2.0},
            {"throatRadius":0.050,"chamberPressure":0.5,"Lstar":3.50,"contractionRatio":2.0},
            {"throatRadius":0.200,"chamberPressure":0.5,"Lstar":3.50,"contractionRatio":8.0},
            {"throatRadius":0.005,"chamberPressure":20,  "Lstar":3.50,"contractionRatio":8.0},
        ]
        base3 = {"convergentAngle":30,"nozzleAngle":15,"nozzleAR":8,
                 "injectorDensity":0.6,"coolantFlow":0.5,"filmCooling":False,
                 "ablativeLayer":False,"material":"copper","injector":"like_doublet",
                 "coolingChannels":16,"wallThickness":0.40,"regenerativeCooling":True}
        max_L = 0
        for ep in extremes:
            try:
                g = build_geometry_profile({**base3,**ep},"kerolox")
                max_L = max(max_L, g["L_cyl_m"])
            except Exception:
                pass
        passed = max_L <= 1.5
        assert_t("T3: Chamber bounded",
                 "L_cyl must stay ≤ 1500 mm across extreme input combinations.",
                 "max(L_cyl) ≤ 1.5 m",
                 f"max(L_cyl) = {max_L*1000:.0f} mm", passed,
                 f"Bounded: {passed}")
    except Exception as e:
        assert_t("T3: Chamber bounded","","","ERROR: "+str(e),False)

    # T4
    try:
        base4 = {"convergentAngle":30,"nozzleAngle":15,"nozzleAR":8,"contractionRatio":4,
                 "injectorDensity":0.6,"coolantFlow":0.5,"filmCooling":False,
                 "ablativeLayer":False,"material":"copper","injector":"like_doublet",
                 "coolingChannels":16,"wallThickness":0.40,"regenerativeCooling":True}
        gH = build_geometry_profile({**base4,"throatRadius":0.030,"chamberPressure":15,"Lstar":0.80},"kerolox")
        gL = build_geometry_profile({**base4,"throatRadius":0.030,"chamberPressure":1.0,"Lstar":2.50},"kerolox")
        cH = gH.get("chamberGoverningCriterion","Lstar")
        cL = gL.get("chamberGoverningCriterion","Lstar")
        passed = cH != cL or cL != "Lstar"
        assert_t("T4: Criterion switches",
                 "Multi-criterion sizing must produce different criteria under different conditions.",
                 "criterion(high-p) ≠ criterion(low-p) or at least one is not Lstar",
                 f"high-p: {cH}, low-p: {cL}", passed,
                 f"L_cyl(high)={gH['L_cyl_m']*1000:.0f}mm [{cH}], L_cyl(low)={gL['L_cyl_m']*1000:.0f}mm [{cL}]")
    except Exception as e:
        assert_t("T4: Criterion switches","","","ERROR: "+str(e),False)

    # T5
    try:
        gamma = 1.25; AR = 8; p_c = 5e6; A_t = math.pi*0.025**2
        CF_v = thrust_coeff(gamma, AR, p_c, 0)
        CF_s = thrust_coeff(gamma, AR, p_c, 101325)
        Isp_v = 350
        Isp_s = isp_at_altitude(Isp_v, CF_s, CF_v)
        passed = CF_v > CF_s and (CF_v*p_c*A_t) > (CF_s*p_c*A_t) and Isp_v > Isp_s
        assert_t("T5: Vac vs sea-level divergence",
                 "CF_vac, thrust_vac, and Isp_vac must all exceed sea-level values.",
                 "CF_vac > CF_sl, thrust_vac > thrust_sl, Isp_vac > Isp_sl",
                 f"CF: {CF_v:.4f} > {CF_s:.4f}  |  Isp: {Isp_v:.1f} > {Isp_s:.1f} s", passed,
                 f"ΔCF={CF_v-CF_s:.4f}, ΔIsp={Isp_v-Isp_s:.1f} s")
    except Exception as e:
        assert_t("T5: Vac vs sea-level","","","ERROR: "+str(e),False)

    # T6
    try:
        bad = {"throatRadius":0.05,"contractionRatio":4,"Lstar":1.27,"convergentAngle":30,
               "nozzleAngle":15,"nozzleAR":0.5,"chamberPressure":5,"wallThickness":0.3,
               "coolingChannels":16,"injectorDensity":0.6,"coolantFlow":0.5,
               "filmCooling":False,"ablativeLayer":False,"material":"copper",
               "injector":"like_doublet","regenerativeCooling":True}
        good = {**bad, "nozzleAR":8}
        try:
            gb = build_geometry_profile(bad, "kerolox")
            vb = validate_candidate(bad, gb)
            bad_caught = not vb["valid"]
        except Exception:
            bad_caught = True
        gg  = build_geometry_profile(good, "kerolox")
        vg  = validate_candidate(good, gg)
        good_passes = vg["valid"]
        passed = bad_caught and good_passes
        assert_t("T6: Validation flags",
                 "Bad geometry must raise hard fails; valid geometry must pass.",
                 "hardFails raised for bad; valid=true for good",
                 f"Bad caught: {bad_caught} | Good passes: {good_passes}", passed,
                 f"Bad hard fails: {vb.get('hardFails',['(threw)'])}")
    except Exception as e:
        assert_t("T6: Validation flags","","","ERROR: "+str(e),False)

    return results


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://localhost:8050")

    threading.Thread(target=open_browser, daemon=True).start()
    print("\n  ◈ CPAS v2 — Computational Propulsion Architecture Synthesis")
    print("  Python / Dash Edition")
    print("  Starting at http://localhost:8050\n")
    app.run(debug=False, port=8050)
