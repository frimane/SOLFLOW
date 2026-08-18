#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Aug 17 21:16:30 2026
# @author: Âzeddine Frimane

# Streamlit front-end for forecast_variants.py

import os
import sys
import warnings
import tempfile
import gc
import contextlib
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st

import core

_user_csv_import_error = None
try:
    import user_csv_windows as ucw
except Exception as exc:
    _user_csv_import_error = exc
    ucw = None


# DEPLOYMENT CONFIG 
# 
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("FORECAST_MODELS_DIR", os.path.join(_APP_DIR, "models"))
DEFAULT_WINDOWS_PATH = os.environ.get(
    "FORECAST_WINDOWS_PATH", os.path.join(_APP_DIR, "data", "fold_test.npz")
)
EXTRA_IMPORT_PATH = os.environ.get("FORECAST_IMPORT_PATH", _APP_DIR)
DEVICE = os.environ.get("FORECAST_DEVICE", "auto")
# Every window served by this app is generated at 10-minute resolution 
# fixed by the underlying model, not a user choice.
RESOLUTION_MIN = 10.0
APP_VERSION = "v1.0.0"


# Safety limits prevent accidental multi-gigabyte allocations from a browser
# They can be overridden only by environment variables, not by an
# unchecked user input.
MAX_ENSEMBLE = int(os.environ.get("FORECAST_MAX_ENSEMBLE", "150"))
MAX_SELECTED_ROWS = int(os.environ.get("FORECAST_MAX_ROWS", "8"))
MAX_UPLOAD_MB = int(os.environ.get("FORECAST_MAX_UPLOAD_MB", "50"))
MAX_WINDOW_ROWS = int(os.environ.get("FORECAST_MAX_WINDOW_ROWS", "100000"))
MAX_ARRAY_MB = int(os.environ.get("FORECAST_MAX_ARRAY_MB", "150"))
# ---------------------------------------------------------------------------

if EXTRA_IMPORT_PATH:
    sys.path.insert(0, EXTRA_IMPORT_PATH)

st.set_page_config(page_title="SOLFLOW", layout="wide")

try:
    import forecast_variants as fv
except Exception as e:
    st.error(
        "Could not import forecast_variants.py. Make sure both files sit next to app.py, or set "
        f"EXTRA_IMPORT_PATH at the top of app.py.\n\nImport error: {e}"
    )
    st.stop()

# STYLE -- one dark, uniform look shared by the page chrome AND the charts
# (charts are rendered with a transparent background by forecast_variants.py,
# so the card behind them, styled here, shows straight through).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne:wght@400;500;600;700;800&display=swap');

:root {
    --bg0:   #07111e;
    --bg1:   #0c1929;
    --bg2:   #111f33;
    --brd:   #1a2e48;
    --brd2:  #243d5c;
    --amber: #e8b84b;
    --amber2:#c9963a;
    --blue:  #5b9bd5;
    --tx:    #f0f4f8;
    --mu:    #9db4c8;
    --dim:   #5a7a96;
    --mono:  'IBM Plex Mono', 'Courier New', Courier, monospace;
    --syne:  'Syne', system-ui, sans-serif;
    --r:     8px;
    color-scheme: dark;
}

:root {
    --amber-btn: #c9a04a;      /* muted, less saturated than --amber */
    --amber-btn-hover: #ad8a3f;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--bg0) !important;
    color: var(--tx) !important;
    font-family: var(--mono) !important;
}
[data-testid="stHeader"], header,
[data-testid="stToolbar"] {
    background: var(--bg0) !important;
    color: var(--tx) !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 100% !important;
}
p, div, span, li, td, th,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] div,
[data-testid="stMarkdownContainer"] span { color: var(--tx) !important; }

[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--brd) !important;
}
[data-testid="stSidebar"] [data-testid="stMainBlockContainer"],
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.25rem !important;
}

label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--mu) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

.page-title {
    font-family: var(--syne);
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--tx);
    letter-spacing: -0.01em;
    padding-bottom: 0.15rem;
}
.page-title span { color: var(--amber); }
.page-subtitle {
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--mu);
    padding-bottom: 1.3rem;
}
.sec-rule {
    font-family: var(--syne); font-size: 0.95rem;
    font-weight: 600; color: var(--tx);
    letter-spacing: 0.01em;
    padding-bottom: 0.5rem; margin-top: 0.4rem;
    border-bottom: 1px solid var(--brd);
    margin-bottom: 1rem;
}
.side-rule {
    font-family: var(--mono); font-size: 0.68rem;
    font-weight: 600; color: var(--dim);
    text-transform: uppercase; letter-spacing: 0.11em;
    padding-bottom: 0.35rem; border-bottom: 1px solid var(--brd);
    margin: 1.1rem 0 0.7rem 0;
}
.side-rule:first-child { margin-top: 0; }

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stDateInput"] input {
    background: var(--bg0) !important;
    color: var(--tx) !important;
    border: 1px solid var(--brd) !important;
    border-radius: var(--r) !important;
    font-family: var(--mono) !important;
}
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(232,184,75,0.12) !important;
    outline: none !important;
}

div[role="radiogroup"] label, [data-baseweb="select"] {
    font-family: var(--mono) !important;
}
[data-baseweb="select"] > div {
    background: var(--bg0) !important;
    border-color: var(--brd) !important;
    border-radius: var(--r) !important;
}
[data-testid="stCheckbox"] label p {
    text-transform: none !important;
    font-size: 0.82rem !important;
    color: var(--tx) !important;
    letter-spacing: normal !important;
}

.stButton > button {
    background: var(--amber-btn) !important;
    color: #0b1420 !important;
    border: none !important;
    font-family: var(--mono) !important;
    font-weight: 600 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.4rem !important;
    border-radius: var(--r) !important;
    transition: background 0.15s !important;
    width: 100%;
}
.stButton > button:hover { background: var(--amber-btn-hover) !important; }

.stDownloadButton > button {
    background: transparent !important;
    color: var(--amber) !important;
    border: 1px solid var(--amber) !important;
    font-family: var(--mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 0.45rem 1.1rem !important;
    border-radius: var(--r) !important;
    width: 100%;
}
.stDownloadButton > button:hover { background: #2a2413 !important; }

[data-testid="stExpander"] {
    border: 1px solid var(--brd) !important;
    border-radius: var(--r) !important;
    background: var(--bg1) !important;
    margin-bottom: 0.6rem !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    color: var(--mu) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    padding: 0.7rem 1rem !important;
}

/* card that wraps every plain-language callout / note */
.note-box {
    background: #10263a; border: 1px solid #31516c;
    border-left: 3px solid var(--blue); border-radius: 0 var(--r) var(--r) 0;
    padding: 0.85rem 1.1rem; font-family: var(--mono); font-size: 0.8rem;
    color: var(--tx); line-height: 1.85; margin: 0.6rem 0 1.1rem 0;
}
.warn-box {
    background: #2a171b; border: 1px solid #6b3036;
    border-left: 3px solid #dc3c3c; border-radius: 0 var(--r) var(--r) 0;
    padding: 0.65rem 1.1rem; font-family: var(--mono); font-size: 0.76rem;
    color: #f08080; margin: 0.4rem 0 0.7rem 0;
}
.skip-box {
    background: #2f2815; border: 1px solid #705b2b;
    border-left: 3px solid var(--amber); border-radius: 0 var(--r) var(--r) 0;
    padding: 0.6rem 1.1rem; font-family: var(--mono); font-size: 0.76rem;
    color: var(--mu); margin: 0.5rem 0 0.8rem 0; line-height: 1.7;
}

/* section card -- groups a model picker + its rows together, visually
   consistent everywhere it appears */
.card {
    background: var(--bg1); border: 1px solid var(--brd);
    border-radius: var(--r); padding: 1.1rem 1.3rem 1.3rem;
    margin-bottom: 1.1rem;
}
.card-title {
    font-family: var(--syne); font-size: 0.92rem; font-weight: 600;
    color: var(--tx); margin-bottom: 0.7rem;
}
.result-title {
    font-family: var(--syne); font-size: 1.0rem; font-weight: 600;
    color: var(--amber); margin: 1.2rem 0 0.5rem 0;
}

[data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p {
    font-family: var(--syne) !important;
    font-size: 0.92rem !important;
    font-weight: 650 !important;
    text-transform: none !important;
    letter-spacing: 0.01em !important;
    white-space: nowrap !important;
    color: var(--mu) !important;
}
[data-testid="stTabs"] {
    position: sticky; top: 0; z-index: 20;
    background: var(--bg0) !important;
    border-bottom: 1px solid var(--brd) !important;
    padding: .6rem 0 .35rem;
}
[data-testid="stTabs"] [role="tablist"] {
    gap: .25rem !important;
    align-items: stretch !important;
}
[data-testid="stTabs"] button {
    min-height: 3rem !important;
    padding: .72rem 1.05rem !important;
    border-radius: 7px 7px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: #112437 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] [data-testid="stMarkdownContainer"] p {
    color: var(--amber) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--amber) !important; height: 2px !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--brd) !important; }

/* charts sit in a card that matches the app background exactly, so the
   transparent PNG blends seamlessly into the page */
[data-testid="stImage"] {
    background: var(--bg1);
    border: 1px solid var(--brd);
    border-radius: var(--r);
    padding: 0.9rem;
}
[data-testid="stImage"] img { border-radius: 4px; }

[data-testid="stDataFrame"] { border: 1px solid var(--brd) !important; border-radius: var(--r) !important; }

hr { border-color: var(--brd) !important; }

.footer {
    font-family: var(--mono); font-size: 0.68rem; color: var(--dim);
    text-align: center; padding: 1.4rem 0 0.8rem; letter-spacing: 0.04em;
    border-top: 1px solid var(--brd); margin-top: 2.8rem;
}
</style>
""", unsafe_allow_html=True)

# Premium presentation layer: restrained navy/amber palette, stronger hierarchy,
# modern cards, and explicit status styles without changing the forecasting logic.
st.markdown("""
<style>
:root { --bg0:#071018; --bg1:#0d1b2a; --bg2:#13263b; --brd:#203a55; --amber:#f1bd56; --blue:#68b6e8; --green:#6fd0a5; --tx:#f4f7fb; --mu:#a9bfd2; }
[data-testid="stAppViewContainer"] { background: var(--bg0) !important; }
[data-testid="stSidebar"] { background: var(--bg1) !important; }
.block-container { max-width:1500px !important; }
.hero { padding:1.45rem 1.65rem 1.35rem; border:1px solid var(--brd2); border-radius:var(--r); background:var(--bg1); margin-bottom:1.15rem; }
.hero-kicker { color:var(--amber); font-family:var(--mono); text-transform:uppercase; letter-spacing:.14em; font-size:.68rem; font-weight:700; }
.hero h1 { margin:.25rem 0 .35rem; color:var(--tx); font-family:var(--syne); font-size:2.35rem; line-height:1.08; letter-spacing:-.035em; }
.hero p { margin:0; color:var(--mu); font-family:var(--mono); font-size:.86rem; line-height:1.7; max-width:960px; }
.identity-card { padding:1rem 1.15rem; border:1px solid var(--brd2); border-left:4px solid var(--amber); border-radius:var(--r); background:var(--bg2); margin:.65rem 0 1rem; }
.identity-card .model-id { color:var(--amber); font-family:var(--syne); font-size:1.05rem; font-weight:700; }
.identity-card .model-detail { color:var(--mu); font-family:var(--mono); font-size:.77rem; line-height:1.65; margin-top:.35rem; }
.status-pill { display:inline-block; padding:.22rem .55rem; border-radius:999px; font-family:var(--mono); font-size:.64rem; letter-spacing:.06em; text-transform:uppercase; margin:.2rem .25rem .2rem 0; border:1px solid #31516c; background:#10263a; color:var(--blue); }
.status-pill.ok { border-color:#39785f; background:#123125; color:var(--green); }
.status-pill.warn { border-color:#8d6c32; background:#332910; color:var(--amber); }
.small-muted { color:var(--mu); font-size:.72rem; line-height:1.65; }
.about-hero { padding:2.8rem 3rem 2.4rem; border:1px solid var(--brd2); border-radius:var(--r); background:#0d1b2a; margin:.4rem 0 1.2rem; }
.about-kicker, .about-eyebrow { color:var(--amber); font-family:var(--mono); text-transform:uppercase; letter-spacing:.14em; font-size:.66rem; font-weight:700; }
.about-hero h1 { margin:.8rem 0 .7rem; color:var(--tx); font-family:var(--syne); font-size:clamp(2rem,4vw,3.6rem); line-height:1.05; letter-spacing:-.045em; }
.about-hero h1 {
    color: var(--tx) !important;
}

.about-hero h1 .solflow-sol {
    color: var(--amber) !important;
}
.about-lede { max-width:720px; margin:0; color:var(--mu); font-family:var(--mono); font-size:.95rem; line-height:1.8; }
.about-status { display:inline-flex; align-items:center; gap:.55rem; margin-top:1.5rem; padding:.45rem .7rem; border:1px solid var(--brd); border-radius:999px; color:var(--mu); font-family:var(--mono); font-size:.68rem; }
.about-status-dot { width:.5rem; height:.5rem; border-radius:50%; background:var(--green); display:inline-block; }
.about-grid { display:grid; grid-template-columns:1.35fr 1fr; gap:1rem; margin:1rem 0; }
.about-panel { padding:1.35rem 1.45rem; border:1px solid var(--brd); border-radius:var(--r); background:#112437; }
.about-panel-wide { min-height:220px; }
.about-panel-full {
    grid-column: 1 / -1;
}
.about-panel-warning { border-left:3px solid var(--amber); }
.about-panel h2 { margin:.55rem 0 .65rem; color:var(--tx); font-family:var(--syne); font-size:1.18rem; letter-spacing:-.02em; }
.about-panel p { color:var(--mu); font-family:var(--mono); font-size:.78rem; line-height:1.8; margin:.4rem 0; }
.csv-table-wrap { width:100%; overflow-x:auto; margin:.85rem 0; border:1px solid var(--brd); border-radius:10px; }
.csv-table { width:100%; min-width:640px; border-collapse:collapse; color:var(--mu); font-family:var(--mono); font-size:.7rem; }
.csv-table th { color:var(--tx); background:#162d43; text-align:left; font-family:var(--syne); font-weight:600; padding:.55rem .65rem; border-bottom:1px solid var(--brd); }
.csv-table td { padding:.5rem .65rem; border-bottom:1px solid rgba(136,165,190,.16); vertical-align:top; }
.csv-table tr:last-child td { border-bottom:0; }
.csv-table code { color:var(--amber); font-size:.68rem; }
.about-small { font-size:.68rem !important; line-height:1.65 !important; }
.about-section-head { display:flex; justify-content:space-between; align-items:baseline; gap:1rem; margin:1.8rem 0 .8rem; padding-bottom:.55rem; border-bottom:1px solid var(--brd); color:var(--tx); font-family:var(--syne); }
.about-section-head span { color:var(--amber); font-family:var(--mono); font-size:.66rem; letter-spacing:.14em; text-transform:uppercase; }
.about-steps { display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; }
.about-step { min-height:155px; padding:1rem; border:1px solid var(--brd); border-radius:var(--r); background:#0d1b2a; }
.about-step span { color:var(--amber); font-family:var(--mono); font-size:.7rem; font-weight:700; }
.about-step h3 { margin:.65rem 0 .4rem; color:var(--tx); font-family:var(--syne); font-size:.95rem; }
.about-step p { margin:0; color:var(--mu); font-family:var(--mono); font-size:.72rem; line-height:1.7; }
.about-panel p,
.about-hero p,
.about-lede,
.about-small {
    text-align: justify;
    text-justify: inter-word;
}

/***********************************/
.sidebar-footer {
    margin-top: 1.5rem;
    padding-top: 0.85rem;
    border-top: 1px solid var(--brd);
    color: #5a7a96;
    font-family: var(--mono);
    font-size: 0.62rem;
    line-height: 1.7;
    text-align: center;
}

.sidebar-footer-muted {
    color: #5a7a96 !important;
}

.sidebar-footer-separator {
    color: #5a7a96 !important;
    padding: 0 0.15rem;
}

.sidebar-footer-solflow {
    color: #e8b84b !important;
    font-weight: 600;
    padding-right: 0.25rem;
}

.sidebar-footer-links {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 0.45rem;
    margin-top: 0.4rem;
}

.sidebar-footer-links a {
    text-decoration: none !important;
}

.sidebar-footer-links img {
    height: 18px;
    vertical-align: middle;
}

.sidebar-footer-email {
    color: #5b9bd5 !important;
    font-size: 0.58rem;
}

[data-testid="stSidebarContent"] {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

[data-testid="stSidebarUserContent"] {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

.sidebar-logo {
    font-family: var(--syne);
    font-size: 0.70rem;
    font-weight: 550;
    letter-spacing: 0.003em;
    color: var(--tx) !important;
    margin: -5rem -1rem 0.9rem -1rem;
    padding: 0.55rem 0 0.55rem 1rem;
    border-bottom: 1px solid var(--brd);
}
.sidebar-logo span {
    color: var(--amber) !important;
}

[data-testid="stSidebarUserContent"] > div:last-child {
    margin-top: auto;
}
/***********************************/

@media (max-width: 900px) { .about-grid, .about-steps { grid-template-columns:1fr 1fr; } .about-hero { padding:2rem 1.4rem; } }
@media (max-width: 600px) { .about-grid, .about-steps { grid-template-columns:1fr; } .about-section-head { display:block; } }
[data-testid="stMetric"] { background:#112437; border:1px solid var(--brd); padding:.7rem .85rem; border-radius:11px; }
[data-testid="stMetricLabel"] { color:var(--mu) !important; }
    [data-testid="stMetricValue"] { color:var(--tx) !important; font-family:var(--syne) !important; }

    /* Mobile presentation layer: keep the research layout intact while making
       the Streamlit shell usable at phone widths. */
    @media (max-width: 700px) {
      html, body {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: hidden !important;
      }
      [data-testid="stAppViewContainer"],
      [data-testid="stMain"],
      [data-testid="stMainBlockContainer"],
      .block-container {
        width: 100% !important;
        max-width: 100vw !important;
        min-width: 0 !important;
        box-sizing: border-box !important;
      }
      [data-testid="stMainBlockContainer"], .block-container {
        padding-top: .75rem !important;
        padding-left: .7rem !important;
        padding-right: .7rem !important;
        padding-bottom: 1rem !important;
      }
      [data-testid="stTabs"] {
        width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        padding: .3rem .25rem .2rem !important;
        overflow: hidden !important;
      }
      [data-testid="stTabs"] [role="tablist"] {
        display: flex !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        flex-wrap: wrap !important;
        gap: .18rem !important;
        justify-content: stretch !important;
      }
      [data-testid="stTabs"] button {
        flex: 1 1 30% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        min-height: 2.45rem !important;
        padding: .42rem .28rem !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
      }
      [data-testid="stTabs"] [role="tabpanel"] {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;
        overflow-x: hidden !important;
      }
      [data-testid="stTabs"] button [data-testid="stMarkdownContainer"] p {
        font-size: .68rem !important;
        line-height: 1.15 !important;
        letter-spacing: 0 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        text-align: center !important;
      }
      .hero, .about-hero {
        padding: 1.25rem 1rem !important;
        margin-bottom: .8rem !important;
      }
      .hero h1 { font-size: clamp(1.65rem, 8vw, 2.15rem) !important; }
      .hero p, .about-lede { font-size: .78rem !important; line-height: 1.65 !important; }
      .identity-card, .card, .about-panel {
        padding: .9rem .85rem !important;
        margin-bottom: .8rem !important;
      }
      .about-panel h2 { font-size: 1.02rem !important; }
      .about-panel p { font-size: .72rem !important; line-height: 1.65 !important; }
      .csv-table-wrap { margin-top: .65rem !important; }
      .csv-table { min-width: 560px !important; font-size: .64rem !important; }
      .csv-table th, .csv-table td { padding: .42rem .48rem !important; }
      .csv-table code { font-size: .61rem !important; }
      .about-grid, .about-steps { grid-template-columns: 1fr !important; }
      .about-section-head { display: block !important; line-height: 1.4 !important; }
      .about-section-head span { display: block; margin-top: .25rem; }
      .status-pill { font-size: .58rem !important; padding: .22rem .42rem !important; }
      .note-box, .warn-box, .skip-box {
        padding: .7rem .8rem !important;
        font-size: .7rem !important;
        line-height: 1.65 !important;
      }
      [data-testid="stImage"] { padding: .35rem !important; }
      [data-testid="stImage"] img { width: 100% !important; height: auto !important; }
      [data-testid="stDataFrame"] { max-width: 100% !important; overflow-x: auto !important; }
      [data-testid="stDownloadButton"] button, .stButton > button {
        min-height: 2.75rem !important;
        padding: .65rem .75rem !important;
      }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { max-width: 100% !important; }
      [data-testid="stSidebar"] label { font-size: .66rem !important; }
    }
</style>
""", unsafe_allow_html=True)



# SCIENTIFIC LABELS -- exact model identities are primary; plain-language text is secondary.

PRIOR_LABELS = {
    "clearsky": "starts from a clear-sky assumption",
    "climatology": "starts from typical seasonal patterns",
    "persistence": "starts by assuming conditions stay the same",
    "blend": "starts from a blend of several assumptions",
    "white": "makes no starting assumption",
    "nwp": "starts from weather-forecast data",
}


def checkpoint_display_name(c):
    # Return a traceable scientific checkpoint label for selectors and tables
    if c["kind"] == "flow":
        identity = f"FlowMatcher | rep={c['rep']} | prior={c['prior']}"
    elif c["kind"] == "deep_quantile":
        identity = "DeepQuantile | multi-quantile pinball baseline"
    else:
        identity = "Unrecognized checkpoint"
    run = c.get("run_tag") or "un tagged run"
    stage = "final" if c.get("fold") is None else f"fold {c['fold']}"
    return f"{identity} | {run} | {stage}"


def model_identity(fm, path=None):
    # Read the scientific identity from the loaded checkpoint, not from UI guesses
    kind = "DeepQuantile" if fm.__class__.__name__ == "DeepQuantile" else "FlowMatcher"
    rep = getattr(getattr(fm, "rep", None), "kind", "raw")
    prior = getattr(getattr(fm, "prior", None), "kind", None)
    cfg = getattr(fm, "cfg", {}) or {}
    data = cfg.get("data", {})
    task = cfg.get("task", {})
    model = cfg.get("model", {})
    nwp = bool(getattr(fm, "nwp_spec", []))
    site_conditioned = getattr(fm, "site_vec", None) is not None
    return {
        "kind": kind, "representation": rep, "prior": prior,
        "nwp": nwp, "site_conditioned": site_conditioned,
        "resolution_min": data.get("resolution_min", RESOLUTION_MIN),
        "history_days": task.get("history_days", "?"),
        "forecast_days": task.get("forecast_days", "?"),
        "condition_clearsky": bool(model.get("condition_clearsky_ghi", False)),
        "condition_nwp": bool(model.get("condition_nwp", False)) and nwp,
        "condition_site": bool(model.get("condition_site", False)) and site_conditioned,
        "path": path or "",
    }


def render_model_identity(fm, path=None, title="Loaded checkpoint"):
    info = model_identity(fm, path)
    model_id = info["kind"]
    if model_id == "FlowMatcher":
        model_id += f" · representation={info['representation']} · prior={info['prior']}"
    details = (
        f"{info['history_days']}-day history → {info['forecast_days']}-day horizon · "
        f"{info['resolution_min']}-minute grid · target: clear-sky index (CSI)"
    )
    pills = ["trained checkpoint", "mask-aware inference"]
    pills.append("NWP conditioning" if info["condition_nwp"] else "no NWP conditioning")
    pills.append("site-conditioned" if info["condition_site"] else "single-site / no site vector")
    pill_html = "".join(
        f"<span class='status-pill {'ok' if ('trained' in p or 'mask' in p) else ''}'>{p}</span>"
        for p in pills
    )
    st.markdown(
        f"<div class='identity-card'><div class='model-id'>{title}: {model_id}</div>"
        f"<div class='model-detail'>{details}<br>"
        f"<b>Inputs used by this checkpoint:</b> history CSI + solar geometry"
        f"{' + future clear-sky GHI' if info['condition_clearsky'] else ''}"
        f"{' + future NWP channels' if info['condition_nwp'] else ''}"
        f"{' + normalized site coordinates' if info['condition_site'] else ''}.<br>"
        f"{pill_html}</div></div>", unsafe_allow_html=True
    )


def capability_table(cap):
    if not cap:
        return pd.DataFrame([{"Capability": "Checkpoint metadata", "Status": "Unavailable"}])
    nwp_arch = "accepted by zero-fill path" if cap["has_nwp"] else "not used"
    nwp_emp = "empirical robustness not established by checkpoint metadata"
    return pd.DataFrame([
        {"Capability": "Standard forecast", "Status": "Supported"},
        {"Capability": "History dropout trained", "Status": "Yes" if cap["history_dropout_trained"] else "No"},
        {"Capability": "NWP conditioning", "Status": "Present" if cap["has_nwp"] else "Not present"},
        {"Capability": "Absent-NWP input path", "Status": nwp_arch},
        {"Capability": "Absent-NWP robustness", "Status": nwp_emp},
        {"Capability": "Site conditioning", "Status": "Present" if cap["site_conditioned"] else "Not present"},
        {"Capability": "Leave-one-site-out evidence", "Status": "Not encoded in checkpoint metadata"},
    ])


def _dedupe_labels(entries):
    # entries: list of (label, path, ...). Appends a short suffix to any
    # label that collides with another, so every option in a picker stays
    # unique without exposing the underlying filename
    from collections import Counter
    counts = Counter(e[0] for e in entries)
    seen = Counter()
    out = []
    for e in entries:
        label = e[0]
        if counts[label] > 1:
            seen[label] += 1
            label = f"{label} (#{seen[label]})"
        out.append((label,) + e[1:])
    return out


# ==============================================================================
# CAPABILITY HELPERS -- plain-language translation of capability_report()
# ==============================================================================
def _nwp_ok(cap):
    return fv._nwp_absence_ok(cap)


def _coldstart_ready(cap):
    if not cap:
        return False
    return cap["history_dropout_trained"] and (not cap["has_nwp"] or _nwp_ok(cap))


def capability_lines(cap):
    lines = ["Standard forecast: supported"]
    lines.append(
        "History-free input was seen during training: " +
        ("yes" if cap["history_dropout_trained"] else "no"))
    if not cap["has_nwp"]:
        lines.append("NWP conditioning: not used by this checkpoint")
    else:
        lines.append("Absent-NWP input: accepted by the architecture through zero-fill")
        lines.append("Absent-NWP robustness: requires empirical ablation; not proven by metadata")
    lines.append(
        "True unseen-site generalization: "
        + ("requires held-out-site evaluation" if cap.get("multi_site") else "not supported by a single-site checkpoint"))
    return lines


def render_capability(cap):
    st.markdown(
        "<div class='note-box'><b>What the checkpoint metadata establish</b><br>" +
        "<br>".join(capability_lines(cap)) + "</div>",
        unsafe_allow_html=True)
    st.dataframe(capability_table(cap), width="stretch", hide_index=True)


# ==============================================================================
# SHARED PLUMBING
# ==============================================================================
def _release_runtime_memory():
    # Best-effort cleanup after a model/data switch or failed inference
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        # Cleanup must never become a new source of failure.
        pass


def _clear_model_cache():
    # Drop cached checkpoint objects and release their CPU/GPU allocations
    try:
        _load_checkpoint.clear()
    except Exception:
        pass
    _release_runtime_memory()


@st.cache_resource(show_spinner=False, max_entries=1)
def _out_dir():
    return tempfile.mkdtemp(prefix="forecast_app_")


OUT_DIR = _out_dir()


@st.cache_data(show_spinner=False)
def _discover(models_dir):
    return fv.discover_checkpoints(models_dir)


@st.cache_resource(show_spinner="Loading model...", max_entries=2)
def _load_checkpoint(path, device):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = fv.load_any(path, device=fv.core.get_device(device))
    return model


@st.cache_data(show_spinner="Loading data file...", max_entries=1)
def _load_windows_from_path(path):
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Window file not found: {path}")
    return _validate_windows(fv.load_windows(path), source=path)


@st.cache_data(show_spinner="Loading uploaded window file...", max_entries=1)
def _load_windows_from_bytes(file_bytes):
    import io
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Uploaded file exceeds the {MAX_UPLOAD_MB} MB safety limit.")
    with np.load(io.BytesIO(file_bytes), allow_pickle=False) as data:
        W = {k: data[k] for k in data.files}
    return _validate_windows(W, source="uploaded .npz")


@st.cache_data(show_spinner="Preparing uploaded solar/NWP CSV...", max_entries=2)
def _load_user_csv_from_bytes(file_bytes, checkpoint_path=None):
    if ucw is None:
        raise RuntimeError("User CSV support is unavailable: " + str(_user_csv_import_error))
    if len(file_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(f"Uploaded file exceeds the {MAX_UPLOAD_MB} MB safety limit.")
    model_or_cfg = fv.core.DEFAULT_CONFIG
    if checkpoint_path:
        model_or_cfg = _load_checkpoint(checkpoint_path, DEVICE)
    W, meta = ucw.build_user_windows(
        file_bytes, model_or_cfg,
        resolution_min=int(RESOLUTION_MIN), minimum_history_days=4)
    return _validate_windows(W, source="uploaded user CSV"), meta


def _load_selected_checkpoint(path):
    # Load one selected checkpoint and release the previous model first
    previous = st.session_state.get("_active_checkpoint")
    if previous != path:
        _clear_model_cache()
        st.session_state["_active_checkpoint"] = path
    try:
        return _load_checkpoint(path, DEVICE)
    except Exception as exc:
        _release_runtime_memory()
        st.error(f"Could not load this checkpoint safely: {exc}")
        st.stop()


def _validate_windows(W, source="window file"):
    # Validate window arrays before they reach a checkpoint or plot
    required = {"hist_csi", "fut_csi", "hist_zen", "fut_zen",
                "hist_mask", "fut_mask"}
    missing = sorted(required - set(W))
    if missing:
        raise ValueError(f"{source} is missing required arrays: {', '.join(missing)}")
    n = int(np.asarray(W["fut_csi"]).shape[0])
    if n < 1 or n > MAX_WINDOW_ROWS:
        raise ValueError(f"{source} contains {n:,} rows; allowed range is 1–{MAX_WINDOW_ROWS:,}.")
    shapes = {k: np.asarray(W[k]).shape for k in required}
    if any(len(s) != 2 for s in shapes.values()):
        raise ValueError(f"{source} arrays must all be two-dimensional; got {shapes}")
    if any(np.asarray(W[k]).shape[0] != n for k in required):
        raise ValueError(f"{source} arrays have inconsistent row counts: {shapes}")
    if W["hist_csi"].shape[1] != W["hist_mask"].shape[1] or W["fut_csi"].shape[1] != W["fut_mask"].shape[1]:
        raise ValueError(f"{source} CSI and mask widths do not match: {shapes}")
    total_bytes = sum(np.asarray(v).nbytes for v in W.values() if isinstance(v, np.ndarray))
    if total_bytes > MAX_ARRAY_MB * 1024 * 1024:
        raise ValueError(f"{source} expands to {total_bytes / 2**20:.0f} MB; limit is {MAX_ARRAY_MB} MB.")
    if "K" in W and int(np.asarray(W["K"]).item()) <= 0:
        raise ValueError(f"{source} contains an invalid K value.")
    return W


def _validate_checkpoint_user_contract(fm, W):
    # Validate normalized user windows against the frozen checkpoint contract
    if not hasattr(fm, "_resolve_nwp_spec"):
        return None
    supplied = fv.slice_nwp_from_windows(W, np.array([0])) if W["fut_csi"].shape[0] else None
    actual = fm._resolve_nwp_spec(supplied or {})
    if actual != getattr(fm, "nwp_spec", []):
        return ("The uploaded CSV was normalized successfully, but its NWP "
                "channel contract does not match this checkpoint. "
                f"Checkpoint={getattr(fm, 'nwp_spec', [])}; supplied={actual}. "
                "Choose a compatible checkpoint or retrain with the same NWP "
                "channel configuration.")
    if getattr(fm, "site_vec", None) is not None:
        if "site_coords" not in W:
            return ("This checkpoint uses site conditioning, but the uploaded CSV "
                    "has no site coordinates. Add latitude, longitude, and "
                    "elevation columns to the CSV.")
        if tuple(np.asarray(W["site_coords"]).shape) != (W["fut_csi"].shape[0], 3):
            return ("The uploaded site coordinates must have one normalized "
                    "[latitude, longitude, elevation] vector per forecast window.")
    return None


def _safe_int(value, low, high, label):
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be an integer.")
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low} and {high}.")
    return value


def _check_selected_rows(rows, n_rows, label="selected rows"):
    rows = [int(r) for r in rows]
    if len(rows) > MAX_SELECTED_ROWS:
        raise ValueError(f"Select at most {MAX_SELECTED_ROWS} {label} to protect memory.")
    if any(r < 0 or r >= n_rows for r in rows):
        raise ValueError(f"{label} contains an index outside 0–{max(n_rows - 1, 0)}.")
    return rows


def _is_memory_error(exc):
    text = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in text or "cuda error" in text


def run_capturing_warnings(fn, *args, **kwargs):
    # Run inference defensively and always release temporary allocations
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            _release_runtime_memory()
            if _is_memory_error(exc):
                raise RuntimeError(
                    "Inference exceeded available memory. Reduce the ensemble "
                    "size, process fewer rows, or use CPU inference.") from exc
            raise
        finally:
            _release_runtime_memory()
    return result, [str(w.message) for w in wlist]


def show_warnings(msgs):
    for m in msgs:
        st.markdown(f"<div class='warn-box'>{m}</div>", unsafe_allow_html=True)


def _checkpoint_default_priority(c):
    # Order the scientifically recommended default first without hiding
    # any checkpoint. The merged/final gauss+nwp FlowMatcher is the preferred
    # application default; fold checkpoints remain available for inspection
    if c.get("kind") != "flow":
        return (3, 9, 9, c.get("path", ""))
    is_final = c.get("fold") is None
    rep = c.get("rep")
    prior = c.get("prior")
    if is_final and rep == "gauss" and prior == "nwp":
        return (0, 0, 0, c.get("path", ""))
    if rep == "gauss" and prior == "nwp":
        fold_rank = c.get("fold") if c.get("fold") is not None else 99
        return (1, 0 if is_final else 1, fold_rank, c.get("path", ""))
    if is_final and rep == "gauss" and prior == "climatology":
        return (2, 0, 0, c.get("path", ""))
    fold_rank = c.get("fold") if c.get("fold") is not None else 99
    return (3, 0 if is_final else 1, fold_rank, c.get("path", ""))


def checkpoint_picker(kind_filter, label, key, allow_none=False,
                       require_coldstart=False):
    # Lists checkpoints of `kind_filter` from MODELS_DIR under readable
    # names. When require_coldstart=True, models that structurally cannot
    # start from scratch are left off the list entirely
    ckpts = _discover(MODELS_DIR)
    if kind_filter is not None:
        ckpts = [c for c in ckpts if c["kind"] == kind_filter]
    ckpts = sorted(ckpts, key=_checkpoint_default_priority)
    if not ckpts:
        st.info(f"No models of this type were found in the models folder.")
        return None, None

    entries = []  # (display_label, path, cap)
    # Never load checkpoints while constructing a selector. This function runs
    # on every Streamlit rerun. Capability validation is repeated after the
    # user presses Run, where an unsupported cold-start request is handled
    # safely and explained.
    for c in ckpts:
        entries.append((checkpoint_display_name(c), c["path"], None))
    if require_coldstart:
        st.caption("Cold-start compatibility is checked after Run forecast.")

    if not entries:
        st.warning(
            "None of the available models support a new site with no data. "
            "A model needs to be built to cope with missing recent data, "
            "and either not use weather-forecast data or not depend on it "
            "as its starting point. Try the Forecast or Data Availability "
            "tab with an existing site's data instead.")
        return None, None

    entries = _dedupe_labels(entries)
    options = {disp: (path, cap) for disp, path, cap in entries}
    labels = (["None"] if allow_none else []) + list(options.keys())
    choice = st.selectbox(label, labels, key=key)
    if choice == "None":
        return None, None
    return options[choice]


def build_csv_bytes(series, primary, quantiles, mask, truth, mode, ghi_cs):
    def convert(v):
        v = np.asarray(v, float)
        if mode == "ghi" and ghi_cs is not None:
            return v * np.asarray(ghi_cs, float)
        return v

    prim = np.asarray(series[primary], float)
    H = prim.shape[-1]
    df = pd.DataFrame({"step": np.arange(H)})
    if mask is not None:
        df["valid"] = np.asarray(mask, bool)
    for q in sorted(quantiles):
        df[f"{primary}_q{q}"] = convert(np.quantile(prim, q, axis=0))
    med = np.quantile(prim, 0.5, axis=0) if prim.shape[0] > 1 else prim[0]
    df[f"{primary}_median"] = convert(med)
    for name, v in series.items():
        if name == primary or v is None:
            continue
        v = np.asarray(v, float)
        med_v = np.median(v, axis=0) if v.shape[0] > 1 else v[0]
        df[f"{name}_median"] = convert(med_v)
    if truth is not None:
        df["truth"] = convert(np.asarray(truth, float))
    return df.to_csv(index=False).encode()


def png_download_button(path, key):
    with open(path, "rb") as f:
        st.download_button("Download chart (PNG)", data=f.read(),
                            file_name=os.path.basename(path),
                            mime="image/png", key=key)


def csv_download_button(csv_bytes, filename, key):
    st.download_button("Download data (CSV)", data=csv_bytes,
                        file_name=filename, mime="text/csv", key=key)


def quantiles_of(fm):
    return fm.cfg["eval"].get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])


def plot_kwargs():
    # Every chart-drawing call reads its display choices from the one,
    # global Settings panel in the sidebar -- nothing is repeated per tab
    kw = {"style": st.session_state.get("chart_style", "bands")}
    if kw["style"] == "scenarios":
        kw["spaghetti_alpha"] = st.session_state.get("chart_alpha", 0.05)
        kw["spaghetti_max"] = st.session_state.get("chart_max_lines", 60)
    return kw


def current_mode():
    return "ghi" if st.session_state.get("units", "power") == "power" else "csi"


# ==============================================================================
# SIDEBAR -- every shared setting lives here, once, and applies everywhere.
# ==============================================================================
st.sidebar.markdown(
    f"<div class='sidebar-logo'><span>SOL</span>FLOW {APP_VERSION}</div>",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div class='side-rule'>Data source</div>",
                     unsafe_allow_html=True)
windows_choice = st.sidebar.radio(
    "Site data", ["Use default data file", "Upload your own"], index=0,
    label_visibility="collapsed",
    help="The dataset of recent readings and forecast periods the app "
         "works from.")
if windows_choice == "Use default data file":
    uploaded_windows = None
    user_csv_meta = None
else:
    uploaded_windows = st.sidebar.file_uploader(
        "Upload processed solar + NWP CSV", type=["csv"],  max_upload_size=MAX_UPLOAD_MB,
        help="It must already contain regular 10-minute timestamps, at least four "
             "complete days, and the following physical quantities: GHI, clear-sky "
             "GHI, solar zenith, NWP shortwave irradiance, and NWP cloud cover. "
             "Latitude, longitude, and elevation are mandatory.")


def _sync_data_resource_context():
    """Invalidate data caches when the selected source changes."""
    if windows_choice == "Use default data file":
        try:
            token = ("default", DEFAULT_WINDOWS_PATH, os.path.getmtime(DEFAULT_WINDOWS_PATH))
        except OSError:
            token = ("default", DEFAULT_WINDOWS_PATH, None)
    else:
        size = getattr(uploaded_windows, "size", 0) if uploaded_windows is not None else 0
        name = getattr(uploaded_windows, "name", "") if uploaded_windows is not None else ""
        token = ("upload_csv", name, size)
    if st.session_state.get("_data_context") != token:
        for fn_name in ("_load_windows_from_path", "_load_windows_from_bytes", "_load_user_csv_from_bytes"):
            try:
                globals()[fn_name].clear()
            except Exception:
                pass
        st.session_state["_data_context"] = token
        _release_runtime_memory()


def get_windows(checkpoint_path=None):
    _sync_data_resource_context()
    try:
        if windows_choice == "Use default data file":
            if not os.path.exists(DEFAULT_WINDOWS_PATH):
                return None, f"Data file not found: {DEFAULT_WINDOWS_PATH}"
            return _load_windows_from_path(DEFAULT_WINDOWS_PATH), None
        if uploaded_windows is None:
            return None, "Upload a processed solar + NWP CSV above to get started."
        if getattr(uploaded_windows, "size", 0) > MAX_UPLOAD_MB * 1024 * 1024:
            return None, f"Uploaded file exceeds the {MAX_UPLOAD_MB} MB safety limit."
        loaded, meta = _load_user_csv_from_bytes(uploaded_windows.getvalue(), checkpoint_path)
        st.session_state["_user_csv_meta"] = meta
        return loaded, None
    except Exception as exc:
        _release_runtime_memory()
        return None, f"Data file was rejected safely: {exc}"


st.sidebar.markdown("<div class='side-rule'>Display</div>",
                     unsafe_allow_html=True)
st.sidebar.radio(
    "Units",
    ["GHI (W/m²)", "CSI"],
    index=0,
    key="units_display",
    help="How every chart and downloaded file shows the forecast."
)

st.session_state["units"] = (
    "power"
    if st.session_state["units_display"].startswith("GHI")
    else "clarity"
)


st.sidebar.radio(
    "Chart style",
    ["Predictive intervals", "Spaghetti plot"],
    index=0, key="chart_style_display",
    help="A central predictive interval summarizes the trained ensemble. "
         "Coverage is empirical and should be interpreted with the truth "
         "series when available. Spaghetti plot shows individual "
         "simulated outcomes.")
st.session_state["chart_style"] = (
    "bands" if st.session_state["chart_style_display"].startswith("Predictive")
    else "scenarios")

if st.session_state["chart_style"] == "scenarios":
    st.sidebar.slider("Line visibility", min_value=0.01, max_value=0.30,
                       value=0.2, step=0.01, key="chart_alpha",
                       help="Lower is fainter, useful when many lines "
                            "overlap.")
    st.sidebar.slider("Lines shown", min_value=10, max_value=300, value=60,
                       step=10, key="chart_max_lines")

st.sidebar.markdown("<div class='side-rule'>Simulation</div>",
                     unsafe_allow_html=True)
n_ensemble = st.sidebar.number_input(
    "Ensemble size", min_value=1, max_value=MAX_ENSEMBLE, value=min(25, MAX_ENSEMBLE),
    help=f"Bounded to {MAX_ENSEMBLE} outcomes to protect CPU/GPU memory.")
seed = st.sidebar.number_input(
    "Random seed", value=0, step=1,
    help="Same seed and same inputs always reproduce the same result.")

_YEAR = 2026
st.sidebar.markdown(
    f"""
    <div class="sidebar-footer">
        <!-- <div>
            <span class="sidebar-footer-muted">&copy; {_YEAR}</span>
            <span class="sidebar-footer-separator">&middot;</span>
            <span class="sidebar-footer-solflow">SOLFLOW</span>
            <span class="sidebar-footer-muted">{APP_VERSION}</span>
        </div> -->
        <div class="sidebar-footer-links">
            <a href="https://github.com/frimane/SOLFLOW" target="_blank">
                <img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github">
            </a>
            <a href="https://www.gnu.org/licenses/gpl-3.0.html" target="_blank">
                <img src="https://img.shields.io/badge/License-GPLv3-blue.svg">
            </a>
            <a href="mailto:Azeddine.frimane@yahoo.com">
                <img
                    src="https://img.shields.io/badge/Email-Contact-e8b84b?logo=minutemailer&logoColor=white"
                    alt="Email"
                >
            </a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
 )

# Global navigation is intentionally the first main-pane element. The
# introduction and usage guide live only inside About so operational tabs remain
# focused on their task.
tab_about, tab_list, tab_forecast, tab_avail, tab_cmp, tab_install = st.tabs(
    [
        "About",
        "Models",
        "Forecast",
        "Data Availability",
        "Compare Models",
        "Installation",
    ]
)


#------------------------------------------------------
# helper for the comparator tab
def render_comparison_metrics(series, truth, mask, fm, ghi_cs=None):
    truth = np.asarray(truth, dtype=np.float64).reshape(1, -1)
    mask = np.asarray(mask, dtype=bool).reshape(1, -1)

    # Never score missing observations.
    score_mask = mask & np.isfinite(truth)

    ghi = None
    if ghi_cs is not None:
        ghi = np.asarray(ghi_cs, dtype=np.float64).reshape(1, -1)
        if ghi.shape != truth.shape:
            st.error(
                f"GHI metric error: truth has shape {truth.shape}, "
                f"but fut_ghi_cs has shape {ghi.shape}."
            )
            ghi = None

    valid_csi = int(np.count_nonzero(score_mask))
    valid_ghi = (
        int(np.count_nonzero(score_mask & np.isfinite(ghi)))
        if ghi is not None else 0
    )

    st.caption(
        f"Metric inputs — truth: {truth.shape}, mask: {mask.shape}, "
        f"clear-sky GHI: {None if ghi is None else ghi.shape}, "
        f"valid CSI points: {valid_csi}, valid GHI points: {valid_ghi}"
    )

    if valid_csi == 0:
        st.error("No valid future observations are available for this window.")
        return

    rows = []
    returned_keys = {}

    for model_name, ensemble in series.items():
        pred = np.asarray(ensemble, dtype=np.float64)

        # Forecast output from predict_ensemble is [members, horizon].
        # evaluate() requires [windows, members, horizon].
        if pred.ndim == 2:
            pred = pred[None, :, :]

        if pred.ndim != 3:
            st.warning(
                f"Skipping {model_name}: expected [1, members, horizon], "
                f"received {pred.shape}."
            )
            continue

        if pred.shape[0] != 1 or pred.shape[2] != truth.shape[1]:
            st.warning(
                f"Skipping {model_name}: forecast shape {pred.shape} is "
                f"incompatible with truth shape {truth.shape}."
            )
            continue

        # Invalid model values must not silently become a good score.
        if not np.isfinite(pred[:, :, score_mask[0]]).all():
            st.warning(
                f"Skipping {model_name}: forecast contains non-finite values "
                "on valid evaluation points."
            )
            continue

        try:
            scores = core.evaluate(
                pred_ens=pred,
                truth=truth,
                mask=score_mask,
                cfg=fm.cfg,
                K=int(fm.K),
                n_days=int(fm.n_days),
                ghi_cs=ghi if valid_ghi > 0 else None,
            )
        except Exception as exc:
            st.warning(f"Metric calculation failed for {model_name}: {exc}")
            continue

        returned_keys[model_name] = sorted(scores.keys())

        rows.append(
            {
                "Model": model_name,
                "GHI CRPS (W/m²)": scores.get("crps_ghi", np.nan),
                "GHI nCRPS": scores.get("ncrps_ghi", np.nan),
                "GHI RMSE (W/m²)": scores.get("rmse_ghi", np.nan),
                "GHI nRMSE": scores.get("nrmse_ghi", np.nan),
                "GHI MAE (W/m²)": scores.get("mae_ghi", np.nan),
                "GHI 80% coverage": scores.get("coverage_80_ghi", np.nan),
                "GHI calibration error": scores.get("calibration_err_ghi", np.nan),
                "GHI energy score": scores.get("energy_score_ghi", np.nan),
                "GHI variogram score": scores.get("variogram_score_ghi", np.nan),
                "GHI ramp CRPS": scores.get("ramp_crps_ghi", np.nan),
                "GHI n ramp CRPS": scores.get("nramp_crps_ghi", np.nan),
                "GHI day-total CRPS": scores.get("daytotal_ghi_crps", np.nan),
                "GHI n day-total CRPS": scores.get("ndaytotal_ghi_crps", np.nan),
                "CSI CRPS": scores.get("crps", np.nan),
                "CSI nCRPS": scores.get("ncrps_csi", np.nan),
                "CSI RMSE": scores.get("rmse", np.nan),
                "CSI nRMSE": scores.get("nrmse_csi", np.nan),
                "CSI MAE": scores.get("mae", np.nan),
                "CSI coverage": scores.get("coverage_80", np.nan),
                "CSI energy score": scores.get("energy_score", np.nan),
                "CSI variogram score": scores.get("variogram_score", np.nan),
                "CSI ramp CRPS": scores.get("ramp_crps", np.nan),
                "CSI day-total CRPS": scores.get("daytotal_crps", np.nan),
            }
        )

    if not rows:
        st.error("No valid model forecasts were available for metric evaluation.")
        return

    table = pd.DataFrame(rows).set_index("Model")

    ghi_columns = [
        "GHI CRPS (W/m²)",
        "GHI nCRPS",
        "GHI RMSE (W/m²)",
        "GHI nRMSE",
        "GHI MAE (W/m²)",
        "GHI 80% coverage",
        "GHI calibration error",
        "GHI energy score",
        "GHI variogram score",
        "GHI ramp CRPS",
        "GHI n ramp CRPS",
        "GHI day-total CRPS",
        "GHI n day-total CRPS",
    ]

    st.markdown(
        "<div class='sec-rule'>Primary metrics in GHI space</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "GHI is obtained member by member as CSI × future clear-sky GHI. "
        "Lower is better for error scores; 80% coverage should be close to 0.80."
    )
    st.dataframe(
        table[ghi_columns].style.format("{:.4f}", na_rep="—"),
        width="stretch",
    )

    with st.expander("Secondary CSI diagnostics"):
        csi_columns = [
            "CSI CRPS",
            "CSI nCRPS",
            "CSI RMSE",
            "CSI nRMSE",
            "CSI MAE",
            "CSI coverage",
            "CSI energy score",
            "CSI variogram score",
            "CSI ramp CRPS",
            "CSI day-total CRPS",
        ]
        st.dataframe(
            table[csi_columns].style.format("{:.4f}", na_rep="—"),
            width="stretch",
        )

    with st.expander("Metric diagnostics"):
        st.json(
            {
                "valid_csi_points": valid_csi,
                "valid_ghi_points": valid_ghi,
                "returned_keys": returned_keys,
            }
        )


# ==============================================================================
# TAB: About
# ==============================================================================
with tab_about:
    st.markdown("""
    <section class='about-hero'>
      <h1><span class='solflow-sol'>SOL</span>FLOW</h1>
      <h2>Conditional Flow Matching for Solar Irradiance Forecasting</span></h2>
      <p class='about-lede' style="width: 100%; max-width: none; font-size: 0.82rem">Trained on observations from 7 SURFRAD stations at 10-minute temporal resolution. 
      Generalization assessed using 4 cross-validation folds, designed to evaluate robustness across 
      both temporal variation and spatially distinct measurement sites</p>
    </section>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-grid'>
      <div class='about-panel about-panel-wide'>
        <div class='about-eyebrow'>Scientific objective</div>
        <h2>Forecast the trajectory, not isolated time points</h2>
        <p>Solar irradiance is represented through the <b>clear-sky index</b>, <b>CSI = GHI / GHI<sub>clear</sub></b>, which separates deterministic solar geometry from cloud-driven variability. The objective is to model the conditional distribution of the complete future CSI trajectory given recent observations, solar geometry, future NWP information, and site coordinates.</p>
        <p>The central hypothesis is that a joint trajectory model can preserve temporal dependence, coherent cloud transitions, and irradiance ramps more faithfully than independent pointwise predictions.</p>
      </div>
      <div class='about-panel'>
        <div class='about-eyebrow'>Primary model</div>
        <h2>FlowMatcher</h2>
        <p><b>FlowMatcher</b> learns a time-dependent velocity field that transports samples from a structured prior distribution to the conditional distribution of future CSI paths. Sampling the learned flow produces an ensemble of physically interpretable trajectories rather than a single estimate.</p>
        <p>Conditioning includes historical CSI and solar geometry, future solar geometry and masks, checkpoint-compatible NWP channels, and normalized site coordinates for pooled models.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-section-head'><span>Model variants</span></div>
    <div class='about-grid'>
      <div class='about-panel'>
        <div class='about-eyebrow'>Representation</div>
        <h2>Transforming the CSI state space</h2>
        <p>The representation defines the coordinate system in which the conditional flow is learned. The available choices are <b>raw</b>, which retains CSI directly; <b>log</b>, which models a positive-valued logarithmic coordinate; <b>logit</b>, which maps bounded CSI to an unconstrained logistic coordinate; and <b>gauss</b>, which applies an empirical probability transform followed by a Gaussian normal-score mapping. Each transform is fitted using training data, and generated trajectories are decoded back to physical CSI before evaluation.</p>
      </div>
      <div class='about-panel'>
        <div class='about-eyebrow'>Prior</div>
        <h2><code>nwp</code>, <code>clearsky</code>, and alternatives</h2>
        <p>The prior defines the initial trajectory family. <b>nwp</b> uses the NWP-derived CSI anchor; <b>clearsky</b> follows the clear-sky reference; <b>climatology</b> uses typical patterns; <b>persistence</b> uses recent conditions; <b>white</b> is weakly structured; and <b>blend</b> combines starting assumptions.</p>
      </div>
      <div class='about-panel about-panel-wide about-panel-full'>
    <div class='about-eyebrow'>Comparator</div>
        <h2>DeepQuantile and classical baselines</h2>
        <p><b>DeepQuantile</b> estimates conditional quantiles directly using multi-quantile pinball loss. Persistence, climatology, analog-day, and direct-NWP forecasts provide reference baselines. This comparison separates pointwise accuracy from probabilistic trajectory structure.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # st.markdown("""
    # <div class='about-section-head'><span>04 · Results</span><b>Merged evaluation across four folds</b></div>
    # <div class='about-grid'>
    #   <div class='about-panel about-panel-wide'>
    #     <p>The recommended application checkpoint is <b>FlowMatcher · gauss · nwp</b>. Across the merged four-fold evaluation, it achieved <b>CSI CRPS 0.1161</b>, <b>GHI-weighted CRPS 0.1087</b>, <b>GHI RMSE 126.9 W/m²</b>, <b>80% coverage 0.7716</b>, <b>calibration error 0.0156</b>, and <b>skill 0.3710 versus direct NWP</b>.</p>
    #     <p>FlowMatcher is not superior on every scalar metric: merged <b>DeepQuantile</b> achieved CRPS 0.1138. The principal advantage of FlowMatcher is trajectory structure: merged <b>ramp CRPS was 0.1099</b> versus 0.3009 for DeepQuantile, and the variogram score was 0.0425 versus 0.0643. These results support a focused claim: FlowMatcher better represents temporal dependence and ramp behavior, while DeepQuantile remains a strong pointwise comparator.</p>
    #   </div>
    #   <div class='about-panel'>
    #     <div class='about-eyebrow'>Fold interpretation</div>
    #     <h2>What to inspect</h2>
    #     <p><b>Fold 2</b> is the strongest overall illustrative fold. <b>Fold 3</b> is the ramp-focused reference. <b>Fold 0</b> is a useful difficult-case check. The merged result and all folds should remain available for transparent reporting.</p>
    #   </div>
    # </div>
    # """, unsafe_allow_html=True)

    st.markdown("""
    <div class='about-section-head'><span>Use Your Own data</span></div>
    <div class='about-grid'>
      <div class='about-panel about-panel-wide about-panel-full'>
        <p>To evaluate the model on a new site, provide at least <b>four complete days</b> of regular, gap-free data at exactly <b>10-minute resolution</b>: three history days for conditioning and one future day containing both NWP inputs and measured truth. Additional days create additional forecast windows.</p>
        <div class='csv-table-wrap'>
          <table class='csv-table'>
            <thead><tr><th>CSV column</th><th>Definition</th><th>Unit / requirement</th></tr></thead>
            <tbody>
              <tr><td><code>timestamp</code></td><td>UTC observation/forecast valid time</td><td>Regular 10-minute cadence</td></tr>
              <tr><td><code>ghi</code></td><td>Measured global horizontal irradiance</td><td>W/m²</td></tr>
              <tr><td><code>clear_sky_ghi</code></td><td>Clear-sky GHI used for CSI</td><td>W/m²; same convention as training</td></tr>
              <tr><td><code>solar_zenith</code></td><td>Solar zenith angle</td><td>Degrees</td></tr>
              <tr><td><code>nwp_shortwave</code></td><td>Provider forecast shortwave irradiance</td><td>W/m²; future horizon required</td></tr>
              <tr><td><code>nwp_cloud_cover</code></td><td>Provider cloud-cover forecast</td><td>Percent; future horizon required</td></tr>
              <tr><td><code>latitude</code></td><td>Site latitude</td><td>Degrees; mandatory for pooled checkpoints</td></tr>
              <tr><td><code>longitude</code></td><td>Site longitude</td><td>Degrees; mandatory for pooled checkpoints</td></tr>
              <tr><td><code>elevation</code></td><td>Site elevation</td><td>Metres; mandatory for pooled checkpoints</td></tr>
            </tbody>
          </table>
        </div>
        <p class='about-small'>This model is trained using HRRR NWP data as the weather-forecast input. 
        You may use NWP data from another provider, provided that it follows the CSV-formatting instructions specified above.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
# ==============================================================================
# TAB: Models
# ==============================================================================
with tab_list:
    st.markdown("<div class='sec-rule'>Available models</div>",
                unsafe_allow_html=True)
    st.caption(
    "This list contains all trained variants of the Conditional Flow Matching model. "
    "Each entry identifies its representation and prior. The recommended default is "
    "the final FlowMatcher checkpoint with the gauss representation and nwp prior."
)

    if st.button("Refresh list", key="refresh_list"):
        _discover.clear()
        _clear_model_cache()
    ckpts = _discover(MODELS_DIR)
    if not ckpts:
        st.info("No models were found in the models folder.")
    else:
        entries = _dedupe_labels(
            [(checkpoint_display_name(c), c) for c in ckpts])
        rows = []
        # Do not load .pt files here. This tab is an inventory view and must
        # remain instant when any other widget changes. Exact checkpoint
        # metadata is shown after a model is explicitly used for inference.
        for name, c in entries:
            row = {"Model": name}
            row["Representation"] = c.get("rep") or ("DeepQuantile" if c["kind"] == "deep_quantile" else "unknown")
            row["Prior"] = c.get("prior") or "—"
            # row["NWP conditioning"] = "Encoded in checkpoint" if c["kind"] == "flow" else "—"
            # row["History dropout trained"] = "Inspect on run" if c["kind"] == "flow" else "—"
            # row["New-site evidence"] = "Inspect on run" if c["kind"] == "flow" else "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)


# ==============================================================================
# TAB: Forecast
# ==============================================================================
with tab_forecast:
    st.markdown("<div class='sec-rule'>Run forecast</div>",
                unsafe_allow_html=True)
    st.caption("Real time forecast for a site, using its real recent data.")
    mode = current_mode()

    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    ckpt_path, _ = checkpoint_picker("flow", "Model", key="fc_ckpt")
    W, w_err = get_windows(ckpt_path)

    if W is not None:
        n_rows = W["fut_csi"].shape[0]
        rows_sel = st.multiselect(
            f"Days to forecast (maximum {MAX_SELECTED_ROWS})", list(range(n_rows)),
            default=[0], key="fc_rows",
            help="Each number is one saved site-and-period combination in the data file.")
        have_gcs = "fut_ghi_cs" in W
        if mode == "ghi" and not have_gcs:
            st.error("Power-output units need clear-sky data that isn't in this data file. Switch to Sky clarity units in the sidebar.")
        run = st.button("Run forecast", key="fc_run")
    else:
        rows_sel = []
        run = False
        if w_err:
            st.info(w_err)
    st.markdown("</div>", unsafe_allow_html=True)

    if ckpt_path and W is not None and run and rows_sel:
        if len(rows_sel) > MAX_SELECTED_ROWS:
            st.error(f"Select at most {MAX_SELECTED_ROWS} rows to protect memory.")
            st.stop()
        fm = _load_selected_checkpoint(ckpt_path)
        render_model_identity(fm, ckpt_path, title="Selected checkpoint")
        if W["fut_mask"].shape[1] != fm.H_out or W["hist_mask"].shape[1] != fm.H_in:
            st.error("This data file does not match this checkpoint's expected history/forecast shape.")
            st.stop()
        # Uploaded Forecast always uses the complete user-provided history and
        # NWP. Reduced-information experiments belong exclusively to the
        # Data Availability workspace, where they are explicitly labeled.
        no_history = False
        if windows_choice == "Upload my own":
            contract_error = _validate_checkpoint_user_contract(fm, W)
            if contract_error:
                st.error(contract_error)
                st.stop()
        W_infer = W
        qs = quantiles_of(fm)
        have_gcs = "fut_ghi_cs" in W
        for r in rows_sel:
            gcs_row = W["fut_ghi_cs"][r] if have_gcs else None
            fut_nwp = fv.slice_nwp_from_windows(W_infer, np.array([r]))
            with st.spinner(f"Generating forecast for day {r}..."):
                (pred, warns) = run_capturing_warnings(
                    fv.forecast_from_arrays,
                    fm,
                    W_infer["hist_csi"][[r]],
                    W_infer["hist_zen"][[r]],
                    W_infer["fut_zen"][[r]],
                    W_infer["hist_mask"][[r]],
                    W_infer["fut_mask"][[r]],
                    n_ensemble=int(n_ensemble),
                    seed=int(seed),
                    fut_ghi_cs=(gcs_row[None] if have_gcs else None),
                    fut_nwp=fut_nwp,
                    site_coords=(
                        W_infer["site_coords"][[r]]
                        if "site_coords" in W_infer else None
                    ),
                )
            show_warnings(warns)
            truth = W["fut_csi"][r] if "fut_csi" in W else None
            out_path = os.path.join(OUT_DIR, f"forecast_row{r}_{mode}.png")
            result_label = "Forecast"
            fv.plot_forecast(pred[0], truth, W["fut_mask"][r], fm.K,
                              fm.n_days, title=f"{result_label} — day {r}",
                              out_path=out_path, quantiles=qs, mode=mode,
                              ghi_cs_row=gcs_row,
                              zenith_row=W["fut_zen"][r],
                              zenith_cutoff=fv.zenith_cutoff_from_config(fm.cfg),
                              **plot_kwargs())
            st.markdown(f"<div class='result-title'>{result_label} · Day {r}</div>",
                        unsafe_allow_html=True)
            st.image(out_path, width='stretch')
            c1, c2 = st.columns(2)
            with c1:
                png_download_button(out_path, key=f"fc_png_{r}")
            with c2:
                csv_bytes = build_csv_bytes(
                    {"forecast": pred[0]}, "forecast", qs,
                    W["fut_mask"][r], truth, mode, gcs_row)
                csv_download_button(
                    csv_bytes, f"forecast_day{r}_{mode}.csv",
                    key=f"fc_csv_{r}")


# ==============================================================================
# TAB: Data Availability
# ==============================================================================
with tab_avail:
    st.markdown(
    "<div class='sec-rule'>Forecasts With and Without Historical Data</div>",
    unsafe_allow_html=True,
)
    st.caption(
    "Here we evaluate forecasts for the same site with and without (training with drop-out) historical CSI data. "
    "It isolates the contribution of recent-history conditioning and assesses the temporal dependence "
    "between consecutive days. Both settings retain the model's learned representation and prior, "
    "as well as the available future solar geometry and NWP information."
)

    mode = current_mode()

    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    ckpt_path, _ = checkpoint_picker("flow", "Model", key="av_ckpt")
    W, w_err = get_windows(ckpt_path)
    # st.info(
    #     "This workspace intentionally exposes only supported conditions: "
    #     "the full-input forecast and, when the checkpoint was trained with "
    #     "history dropout, a history-free diagnostic. NWP-absence and "
    #     "combined no-history/no-NWP variants are hidden because the current "
    #     "checkpoints were not trained with an explicit NWP-dropout objective."
    # )

    if W is not None:
        n_rows = W["fut_csi"].shape[0]
        rows_sel = st.multiselect(f"Days to forecast (maximum {MAX_SELECTED_ROWS})", list(range(n_rows)),
                                   default=[0], key="av_rows")
        have_gcs = "fut_ghi_cs" in W
        if mode == "ghi" and not have_gcs:
            st.error("Power-output units need clear-sky data that isn't in this data file. Switch to Sky clarity units in the sidebar.")
        # st.caption("Checkpoint capabilities and metadata are loaded only when you run this diagnostic.")
        run = st.button("Run comparison", key="av_run")
    else:
        rows_sel = []
        run = False
        if w_err:
            st.info(w_err)
    st.markdown("</div>", unsafe_allow_html=True)

    if ckpt_path and W is not None and run and rows_sel:
        if len(rows_sel) > MAX_SELECTED_ROWS:
            st.error(f"Select at most {MAX_SELECTED_ROWS} rows to protect memory.")
            st.stop()
        fm = _load_selected_checkpoint(ckpt_path)
        render_model_identity(fm, ckpt_path, title="Selected checkpoint")
        if windows_choice == "Upload my own":
            contract_error = _validate_checkpoint_user_contract(fm, W)
            if contract_error:
                st.error(contract_error)
                st.stop()
        qs = quantiles_of(fm)
        have_gcs = "fut_ghi_cs" in W
        for r in rows_sel:
            with st.spinner(
                f"Evaluating forecast variants for day {r}..."
            ):
                (variants, skipped), warns = run_capturing_warnings(
                    fv.availability_variants,
                    fm,
                    W,
                    np.array([r]),
                    n_ensemble=int(n_ensemble),
                    seed=int(seed),
                )
            # The app exposes only conditions supported by the current
            # checkpoint-training contract. Keep the lower-level diagnostic
            # variants available to the CLI for research, but never present
            # no_nwp or no_history_no_nwp as app forecasts.
            hidden = {"no_nwp", "no_history_no_nwp", "no_nwp_neutral"}
            for hidden_name in hidden:
                if hidden_name in variants:
                    skipped[hidden_name] = (
                        "hidden in the app: this checkpoint was not explicitly "
                        "trained for NWP absence")
                    variants.pop(hidden_name, None)
            show_warnings(warns)
            if skipped:
                pass
                # st.markdown(
                #     "<div class='skip-box'>Not shown (this model wasn't "
                #     "built to handle these conditions): " +
                #     ", ".join(skipped.keys()) +
                #     ". These modes are intentionally unavailable in the app "
                #     "until a checkpoint is retrained with the required "
                #     "missing-input objective.</div>", unsafe_allow_html=True)
            series = {name: ens[0] for name, ens in variants.items()}
            truth = W["fut_csi"][r] if "fut_csi" in W else None
            gcs_row = W["fut_ghi_cs"][r] if have_gcs else None
            out_path = os.path.join(OUT_DIR, f"availability_row{r}_{mode}.png")
            fv.plot_comparison(series, truth, W["fut_mask"][r], fm.K,
                                fm.n_days,
                                title=f"Forecast under reduced data — day {r}",
                                out_path=out_path, quantiles=qs, mode=mode,
                                ghi_cs_row=gcs_row,
                                zenith_row=W["fut_zen"][r],
                                zenith_cutoff=fv.zenith_cutoff_from_config(fm.cfg),
                                primary="full", **plot_kwargs())
            st.markdown(f"<div class='result-title'>Day {r}</div>",
                        unsafe_allow_html=True)
            st.image(out_path, width='stretch')
            c1, c2 = st.columns(2)
            with c1:
                png_download_button(out_path, key=f"av_png_{r}")
            with c2:
                csv_bytes = build_csv_bytes(series, "full", qs,
                                             W["fut_mask"][r], truth,
                                             mode, gcs_row)
                csv_download_button(csv_bytes,
                                     f"availability_day{r}_{mode}.csv",
                                     key=f"av_csv_{r}")


# ==============================================================================
# TAB: Compare Models
# ==============================================================================
with tab_cmp:
    st.markdown("<div class='sec-rule'>Compare against reference methods</div>",
                unsafe_allow_html=True)
    st.caption("Compare the selected FlowMatcher checkpoint with the "
               "DeepQuantile baseline and classical reference methods under "
               "the same saved window.")
    mode = current_mode()

    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    ckpt_path, _ = checkpoint_picker("flow", "Main model", key="cmp_ckpt")
    dq_path, _ = checkpoint_picker("deep_quantile",
                                   "DeepQuantile baseline (optional)",
                                   key="cmp_dq_ckpt", allow_none=True)
    W, w_err = get_windows(ckpt_path)
    baseline_names = list(fv.BASELINE_CLASSES.keys())
    baseline_labels = {
        "day_persistence": "DayPersistence · yesterday repeated",
        "peen": "PeEn · recent-days persistence ensemble",
        "ch_peen": "CHPeEn · complete-history empirical baseline",
        "analog_day": "AnalogDay · most similar past day",
        "nwp_direct": "NWPDirect · direct NWP-CSI reference",
    }
    baselines_display = st.multiselect(
        "Reference methods", list(baseline_labels.values()),
        default=list(baseline_labels.values()), key="cmp_baselines")
    reverse_labels = {v: k for k, v in baseline_labels.items()}
    baselines_sel = [reverse_labels[b] for b in baselines_display]

    if W is not None:
        n_rows = W["fut_csi"].shape[0]
        rows_sel = st.multiselect(f"Days to forecast (maximum {MAX_SELECTED_ROWS})", list(range(n_rows)),
                                   default=[0], key="cmp_rows")
        st.caption("Selected checkpoints are loaded only when you press Run comparison.")
        run = st.button("Run comparison", key="cmp_run")
    else:
        rows_sel = []
        run = False
        if w_err:
            st.info(w_err)
    st.markdown("</div>", unsafe_allow_html=True)

    if ckpt_path and W is not None and run and rows_sel:
        if len(rows_sel) > MAX_SELECTED_ROWS:
            st.error(f"Select at most {MAX_SELECTED_ROWS} rows to protect memory.")
            st.stop()
        fm = _load_selected_checkpoint(ckpt_path)
        render_model_identity(fm, ckpt_path, title="Selected checkpoint")
        if windows_choice == "Upload my own":
            contract_error = _validate_checkpoint_user_contract(fm, W)
            if contract_error:
                st.error(contract_error)
                st.stop()
        qs = quantiles_of(fm)
        have_gcs = "fut_ghi_cs" in W
        dq = None
        if dq_path:
            dq = _load_selected_checkpoint(dq_path)
            if dq.H_in != fm.H_in or dq.H_out != fm.H_out or dq.K != fm.K:
                st.error("The DeepQuantile checkpoint does not match the "
                          "selected FlowMatcher horizon shape, so the two "
                          "cannot be shown together. Choose a compatible "
                          "DeepQuantile checkpoint.")
                dq = None
                dq_path = None

        baselines, bwarns = run_capturing_warnings(
            fv.fit_baselines, baselines_sel, W, fm.cfg, fm.K, fm.n_days)
        show_warnings(bwarns)

        for r in rows_sel:
            fut_nwp = fv.slice_nwp_from_windows(W, np.array([r]))
            gcs_row = W["fut_ghi_cs"][r:r + 1] if have_gcs else None
            rng = np.random.default_rng(int(seed))
            flow_label = f"FlowMatcher · rep={getattr(fm.rep, 'kind', 'raw')} · prior={getattr(getattr(fm, 'prior', None), 'kind', 'n/a')}"
            with st.spinner(f"Generating model comparison for day {r}..."):
                series = {
                    flow_label: fm.predict_ensemble(
                        W["hist_csi"][r:r + 1],
                        W["hist_zen"][r:r + 1],
                        W["fut_zen"][r:r + 1],
                        W["hist_mask"][r:r + 1],
                        W["fut_mask"][r:r + 1],
                        fut_ghi_cs=gcs_row,
                        fut_nwp=fut_nwp,
                        n_ensemble=int(n_ensemble),
                        rng=rng,
                    )[0]
                }
            if dq is not None:
                rng = np.random.default_rng(int(seed))
                series["DeepQuantile"] = dq.predict_ensemble(
                    W["hist_csi"][r:r + 1], W["hist_zen"][r:r + 1],
                    W["fut_zen"][r:r + 1], W["hist_mask"][r:r + 1],
                    W["fut_mask"][r:r + 1], fut_ghi_cs=gcs_row,
                    fut_nwp=fut_nwp, n_ensemble=int(n_ensemble), rng=rng)[0]
            for name, b in baselines.items():
                rng = np.random.default_rng(int(seed))
                try:
                    series[baseline_labels.get(name, name)] = b.predict_ensemble(
                        W["hist_csi"][r:r + 1], W["hist_zen"][r:r + 1],
                        W["fut_zen"][r:r + 1], W["hist_mask"][r:r + 1],
                        W["fut_mask"][r:r + 1], fut_ghi_cs=gcs_row,
                        fut_nwp=fut_nwp, n_ensemble=int(n_ensemble),
                        rng=rng)[0]
                except Exception as e:
                    st.warning(f"\"{baseline_labels.get(name, name)}\" "
                                "couldn't be computed for this day.")

            truth = W["fut_csi"][r] if "fut_csi" in W else None
            gcs_full = W["fut_ghi_cs"][r] if have_gcs else None
            out_path = os.path.join(OUT_DIR, f"compare_row{r}_{mode}.png")
            fv.plot_comparison(series, truth, W["fut_mask"][r], fm.K,
                                fm.n_days,
                                title=f"Model comparison — day {r}",
                                out_path=out_path, quantiles=qs, mode=mode,
                                ghi_cs_row=gcs_full,
                                zenith_row=W["fut_zen"][r],
                                zenith_cutoff=fv.zenith_cutoff_from_config(fm.cfg),
                                primary=flow_label, **plot_kwargs())
            st.markdown(
                f"<div class='result-title'>Day {r}</div>",
                unsafe_allow_html=True,
            )
            st.image(out_path, width="stretch")
            
            render_comparison_metrics(
                series=series,
                truth=W["fut_csi"][r],
                mask=W["fut_mask"][r],
                fm=fm,
                ghi_cs=(W["fut_ghi_cs"][r] if "fut_ghi_cs" in W else None),
            )

            
            c1, c2 = st.columns(2)
            
            with c1:
                png_download_button(out_path, key=f"cmp_png_{r}")
            with c2:
                csv_bytes = build_csv_bytes(series, flow_label, qs,
                                             W["fut_mask"][r], truth,
                                             mode, gcs_full)
                csv_download_button(csv_bytes, f"compare_day{r}_{mode}.csv",
                                     key=f"cmp_csv_{r}")

# ==============================================================================
# TAB: Installation
# ==============================================================================
with tab_install:
    st.markdown(
        """
        <section class='about-hero'>
          <h1>Install and run locally</h1>
          <p class='about-lede' style="width: 100%; max-width: none; font-size: 0.82rem">
            SOLFLOW can run locally. Clone the repository,
            create a virtual environment, and install the requirements.
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='about-grid'>
          <div class='about-panel'>
            <div class='about-eyebrow'>01 · Get the code</div>
            <h2>Clone the repository</h2>
            <p>Download SolFlow from GitHub.</p>
          </div>
          <div class='about-panel'>
            <div class='about-eyebrow'>02 · Environment (optional)</div>
            <h2>Create a virtual environment</h2>
            <p>Python 3.10 or newer is recommended.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        """#01
git clone https://github.com/frimane/SOLFLOW.git
cd solflow

#02
python3 -m venv .venv
source .venv/bin/activate""",
        language="bash",
    )

    st.markdown(
        """
        <div class='about-grid'>
        <div class='about-panel'>
          <div class='about-eyebrow'>03 · Install requirements</div>
          <h2>Install dependencies</h2>
          <p>Install everything listed in <code>requirements.txt</code>.</p>
        </div>
        <div class='about-panel'>
          <div class='about-eyebrow'>04 · Run the app</div>
          <h2>Start Streamlit</h2>
          <p>Run this from the <code>solflow</code> project directory.</p>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.code(
        """#03
python -m pip install --upgrade pip
python -m pip install -r requirements.txt 

#04
streamlit run app.py""",
        language="bash",
    )

