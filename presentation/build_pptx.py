"""
build_pptx.py  —  CRIS-HAZARD Flood Detection Presentation
White background, minimal color, every slide centers on a flowchart/table/figure.

Run:  python3 presentation/build_pptx.py
Requires: pip install python-pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_BASE = os.path.join(ROOT, "results", "figures")
OUT_PPTX = os.path.join(ROOT, "presentation", "flood_detection.pptx")

FIG = {
    "eff_bce_prroc":  os.path.join(FIG_BASE, "baselines", "efficientnet_bce", "pr_roc_curves.png"),
    "res_bce_prroc":  os.path.join(FIG_BASE, "baselines", "resnet50_bce",     "pr_roc_curves.png"),
    "eff_bce_cm":     os.path.join(FIG_BASE, "baselines", "efficientnet_bce", "confusion_matrix.png"),
    "res_bce_cm":     os.path.join(FIG_BASE, "baselines", "resnet50_bce",     "confusion_matrix.png"),
    "fp_comparison":  os.path.join(FIG_BASE, "confounder_fp_comparison.png"),
    "fp_heatmap":     os.path.join(FIG_BASE, "confounder_fp_heatmap.png"),
}

# ── Color palette ─────────────────────────────────────────────────────────────
OCEAN     = RGBColor(0x1A, 0x3A, 0x5C)   # dark navy   — titles, borders
STEEL     = RGBColor(0x4A, 0x60, 0x70)   # gray-blue   — body text
AMBER     = RGBColor(0xD4, 0x7E, 0x00)   # muted amber — accent only
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
RULE      = RGBColor(0xD0, 0xD8, 0xE0)   # thin separator lines
BOX_BG    = RGBColor(0xF2, 0xF6, 0xF9)   # flowchart box fill — very light
ROW_ALT   = RGBColor(0xF7, 0xF9, 0xFB)   # alternating table row

# ── Dimensions ────────────────────────────────────────────────────────────────
W = Inches(13.33)
H = Inches(7.5)


# ── Core helpers ──────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank(prs):
    sl = prs.slides.add_slide(prs.slide_layouts[6])
    sl.background.fill.solid()
    sl.background.fill.fore_color.rgb = WHITE
    return sl


def box(slide, l, t, w, h, fill=BOX_BG, border=RULE, border_pt=0.75):
    """Rectangle with optional border."""
    shp = slide.shapes.add_shape(1, l, t, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if border:
        shp.line.color.rgb = border
        shp.line.width = Pt(border_pt)
    else:
        shp.line.fill.background()
    return shp


def nobox(slide, l, t, w, h, fill=WHITE):
    """Rectangle with no border."""
    shp = slide.shapes.add_shape(1, l, t, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    return shp


def rule(slide, l, t, w, color=RULE):
    """Thin horizontal rule."""
    shp = slide.shapes.add_shape(1, l, t, w, Pt(1))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    return shp


def txt(slide, text, l, t, w, h,
        font="Calibri", size=13, bold=False,
        color=STEEL, align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return tb


def title_line(slide, headline, subtitle=None):
    """Slide title + optional subtitle + thin rule. Returns bottom y."""
    txt(slide, headline,
        Inches(0.55), Inches(0.25), Inches(12.2), Inches(0.65),
        font="Cambria", size=26, bold=True, color=OCEAN)
    if subtitle:
        txt(slide, subtitle,
            Inches(0.55), Inches(0.9), Inches(12.2), Inches(0.38),
            font="Calibri", size=13, color=STEEL)
        rule(slide, Inches(0.55), Inches(1.3), Inches(12.2))
        return Inches(1.42)
    rule(slide, Inches(0.55), Inches(0.95), Inches(12.2))
    return Inches(1.08)


def flow_node(slide, label, l, t, w=Inches(2.1), h=Inches(0.58),
              fill=BOX_BG, border=RULE, font_size=12, bold=False, color=OCEAN):
    """Single flowchart box."""
    box(slide, l, t, w, h, fill=fill, border=border)
    txt(slide, label, l, t, w, h,
        font="Calibri", size=font_size, bold=bold,
        color=color, align=PP_ALIGN.CENTER)


def arrow_down(slide, cx, top, height=Inches(0.28)):
    """Vertical arrow between flow nodes."""
    txt(slide, "↓",
        cx - Inches(0.15), top, Inches(0.3), height,
        font="Calibri", size=16, color=STEEL, align=PP_ALIGN.CENTER)


def arrow_right(slide, left, cy):
    """Horizontal arrow."""
    txt(slide, "→",
        left, cy - Inches(0.18), Inches(0.32), Inches(0.36),
        font="Calibri", size=16, color=STEEL, align=PP_ALIGN.CENTER)


def add_img(slide, path, l, t, w=None, h=None):
    if not os.path.exists(path):
        print(f"  WARNING: figure not found — {path}")
        return None
    return slide.shapes.add_picture(path, l, t, width=w, height=h)


def set_cell(cell, text, size=11, bold=False, italic=False,
             color=STEEL, bg=WHITE, align=PP_ALIGN.CENTER):
    """Apply text and styling to a native PowerPoint table cell."""
    cell.fill.solid()
    cell.fill.fore_color.rgb = bg
    tf = cell.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.alignment = align
    # Clear any existing run elements from the paragraph XML
    p_elem = para._p
    for r_elem in p_elem.findall(qn('a:r')):
        p_elem.remove(r_elem)
    run = para.add_run()
    run.text = text
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def make_table(slide, headers, data_rows, left, top, width, height,
               col_widths=None, hdr_size=12, data_size=11,
               hdr_color=WHITE, hdr_bg=OCEAN,
               alt_bg=ROW_ALT, highlight_last=False,
               highlight_bg=RGBColor(0xE8, 0xF5, 0xEC),
               highlight_color=RGBColor(0x2E, 0x7D, 0x32)):
    """Create a native PowerPoint table with alternating-row styling."""
    n_rows = 1 + len(data_rows)
    n_cols = len(headers)
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    tbl = shape.table

    if col_widths:
        for i, cw in enumerate(col_widths):
            tbl.columns[i].width = cw

    # Header row
    for j, h in enumerate(headers):
        set_cell(tbl.cell(0, j), h, size=hdr_size, bold=True,
                 color=hdr_color, bg=hdr_bg, align=PP_ALIGN.CENTER)

    # Data rows
    for i, row in enumerate(data_rows):
        is_hi = highlight_last and i == len(data_rows) - 1
        if is_hi:
            row_bg, row_col = highlight_bg, highlight_color
        elif i % 2 == 0:
            row_bg, row_col = alt_bg, STEEL
        else:
            row_bg, row_col = WHITE, STEEL

        for j, val in enumerate(row):
            bold = is_hi or j == 0
            set_cell(tbl.cell(i + 1, j), str(val), size=data_size,
                     bold=bold, color=row_col if is_hi else (OCEAN if j == 0 else STEEL),
                     bg=row_bg, align=PP_ALIGN.CENTER)

    return tbl


# ── SLIDE 1 — Title ───────────────────────────────────────────────────────────

def slide_01_title(prs):
    sl = blank(prs)

    txt(sl,
        "Improving CRIS-HAZARD: Automated First-Pass Flood Image Screening\n"
        "via Phase-1 Hard Negative Mining",
        Inches(0.75), Inches(1.5), Inches(11.8), Inches(1.8),
        font="Cambria", size=34, bold=True, color=OCEAN, align=PP_ALIGN.LEFT)

    txt(sl, "When rivers and swimming pools fool the flood classifier — and how to fix it",
        Inches(0.75), Inches(3.45), Inches(10.0), Inches(0.5),
        font="Calibri", size=17, color=AMBER)

    rule(sl, Inches(0.75), Inches(4.1), Inches(5.0), color=RULE)

    for i, (line, sz, bd, col) in enumerate([
        ("Aanya Singh",                   16, True,  OCEAN),
        ("Advised by Dr. Dixon",          13, False, STEEL),
        ("University of South Florida",   13, False, STEEL),
        ("April 18, 2026",               12, False, STEEL),
    ]):
        txt(sl, line, Inches(0.75), Inches(4.25) + Inches(i * 0.4),
            Inches(7.0), Inches(0.42),
            font="Calibri", size=sz, bold=bd, color=col)


# ── SLIDE 2 — CRIS-HAZARD big picture (flowchart) ────────────────────────────

def slide_02_crish_overview(prs):
    sl = blank(prs)
    title_line(sl,
               "CRIS-HAZARD: a community-driven flood risk communication system",
               "Crowdsourced data → high-fidelity flood models → real-time risk alerts for citizens")

    # Five-node horizontal pipeline
    nodes = [
        "Citizens & community\n(smartphones, apps,\nsocial media)",
        "Volunteered Geographic\nInformation (VGI)\n& crowdsourced data",
        "CRIS-HAZARD\nflood-risk models\n(USF / AT&T / ANL)",
        "Uncertainty-aware\nrisk projections\n(short & long term)",
        "Decision makers\n& community\nalerting",
    ]
    nw = Inches(2.15)
    nh = Inches(1.15)
    gap = Inches(0.28)
    total_w = len(nodes) * nw + (len(nodes) - 1) * gap
    start_x = (W - total_w) / 2
    cy = Inches(2.6)

    for i, label in enumerate(nodes):
        lx = start_x + i * (nw + gap)
        fill = RGBColor(0xE8, 0xF1, 0xF8) if i == 2 else BOX_BG
        border_col = OCEAN if i == 2 else RULE
        bd_pt = 1.5 if i == 2 else 0.75
        box(sl, lx, cy, nw, nh, fill=fill, border=border_col, border_pt=bd_pt)
        txt(sl, label, lx, cy, nw, nh,
            font="Calibri", size=11, color=OCEAN if i == 2 else STEEL,
            bold=(i == 2), align=PP_ALIGN.CENTER)
        if i < len(nodes) - 1:
            arrow_right(sl, lx + nw, cy + nh / 2)

    # Challenge callout boxes below
    challenges = [
        ("Challenge 1", "No near-real-time\ntwo-way communication\nplatform"),
        ("Challenge 2", "Flood risk models lack\nfine spatial & temporal\nresolution"),
        ("Challenge 3", "Uncertainty in models\nnot quantified for\ndecision-making"),
    ]
    cx_list = [Inches(1.4), Inches(5.5), Inches(9.6)]
    for (label, desc), lx in zip(challenges, cx_list):
        bw, bh = Inches(2.5), Inches(1.2)
        by = Inches(4.55)
        box(sl, lx, by, bw, bh, fill=RGBColor(0xFF, 0xF5, 0xE8), border=AMBER, border_pt=0.75)
        txt(sl, label, lx, by + Inches(0.05), bw, Inches(0.3),
            font="Calibri", size=10, bold=True, color=AMBER, align=PP_ALIGN.CENTER)
        txt(sl, desc, lx, by + Inches(0.32), bw, Inches(0.82),
            font="Calibri", size=10, color=STEEL, align=PP_ALIGN.CENTER)
        # Dotted line up
        txt(sl, "↑", lx + bw / 2 - Inches(0.1), by - Inches(0.32),
            Inches(0.25), Inches(0.32), font="Calibri", size=14, color=RULE, align=PP_ALIGN.CENTER)

    txt(sl,
        "Source: NSF Award — CRIS-HAZARD (USF + Georgia Tech partnership for Pinellas County, FL)",
        Inches(0.55), Inches(6.95), Inches(12.2), Inches(0.32),
        font="Calibri", size=10, color=RULE)


# ── SLIDE 3 — Gap in current CRIS-HAZARD (two-column flowchart) ──────────────

def slide_03_gap_flowchart(prs):
    sl = blank(prs)
    title_line(sl,
               "The CRIS-HAZARD gap: no automated first-pass image screening",
               "Crowdsourced images arrive unfiltered — confounders reach human analysts, creating alert fatigue")

    col_w = Inches(5.1)
    gap_col = Inches(1.05)
    left_x  = Inches(0.6)
    right_x = left_x + col_w + gap_col

    # Column headers
    for lx, label, col in [
        (left_x,  "Current CRIS-HAZARD flow", STEEL),
        (right_x, "Proposed flow  (this work)", OCEAN),
    ]:
        txt(sl, label, lx, Inches(1.5), col_w, Inches(0.38),
            font="Cambria", size=14, bold=True, color=col, align=PP_ALIGN.CENTER)
        rule(sl, lx, Inches(1.9), col_w, color=RULE if col == STEEL else OCEAN)

    # --- LEFT column nodes ---
    left_nodes = [
        ("Citizen submits image\n(app / social media)", BOX_BG, RULE),
        ("All images queued\n(no filtering)", RGBColor(0xFF, 0xF0, 0xEE), RGBColor(0xCC, 0x44, 0x44)),
        ("Human analyst reviews\n(bottleneck ⚠)", RGBColor(0xFF, 0xF0, 0xEE), RGBColor(0xCC, 0x44, 0x44)),
        ("Flood alert sent\n(delayed; noisy)", BOX_BG, RULE),
    ]
    nw, nh = Inches(4.0), Inches(0.62)
    lx_node = left_x + Inches(0.55)
    y = Inches(2.05)
    for label, fill, border in left_nodes:
        box(sl, lx_node, y, nw, nh, fill=fill, border=border)
        txt(sl, label, lx_node, y, nw, nh,
            font="Calibri", size=11, color=STEEL, align=PP_ALIGN.CENTER)
        if label != left_nodes[-1][0]:
            arrow_down(sl, lx_node + nw / 2, y + nh, height=Inches(0.3))
        y += nh + Inches(0.3)

    # Gap label between columns
    txt(sl, "GAP", left_x + col_w + Inches(0.3), Inches(3.5),
        Inches(0.45), Inches(0.35), font="Cambria", size=13, bold=True,
        color=RGBColor(0xCC, 0x44, 0x44), align=PP_ALIGN.CENTER)
    txt(sl, "→", left_x + col_w + Inches(0.28), Inches(3.9),
        Inches(0.5), Inches(0.35), font="Calibri", size=20,
        color=RULE, align=PP_ALIGN.CENTER)

    # --- RIGHT column nodes ---
    right_nodes = [
        ("Citizen submits image\n(app / social media)", BOX_BG, RULE, False),
        ("Automated first-pass\nscreener  ← THIS WORK", RGBColor(0xE8, 0xF5, 0xEC), RGBColor(0x2E, 0x7D, 0x32), True),
        ("Flood → human review\nNon-flood → discarded", BOX_BG, RULE, False),
        ("Flood alert sent\n(faster; less noise)", RGBColor(0xE8, 0xF5, 0xEC), RGBColor(0x2E, 0x7D, 0x32), False),
    ]
    rx_node = right_x + Inches(0.55)
    y = Inches(2.05)
    for label, fill, border, highlight in right_nodes:
        bd_pt = 1.5 if highlight else 0.75
        box(sl, rx_node, y, nw, nh, fill=fill, border=border, border_pt=bd_pt)
        color = OCEAN if highlight else STEEL
        txt(sl, label, rx_node, y, nw, nh,
            font="Calibri", size=11, bold=highlight, color=color, align=PP_ALIGN.CENTER)
        if label != right_nodes[-1][0]:
            arrow_down(sl, rx_node + nw / 2, y + nh, height=Inches(0.3))
        y += nh + Inches(0.3)

    # Bottom callout
    nobox(sl, Inches(0.55), Inches(6.6), Inches(12.2), Inches(0.62), fill=RGBColor(0xF2, 0xF6, 0xF9))
    txt(sl,
        "Goal: build a screener that catches ≥ 97% of floods while filtering out rivers, pools, and wet roads.",
        Inches(0.7), Inches(6.68), Inches(12.0), Inches(0.45),
        font="Calibri", size=12, color=OCEAN, align=PP_ALIGN.LEFT)


# ── SLIDE 4 — Confounders (visual grid table) ─────────────────────────────────

def slide_04_confounders(prs):
    sl = blank(prs)
    title_line(sl,
               "Visual confounders: 8 non-flood categories share flood-like appearance",
               "River, pool, and wet road images share water texture, reflectance, and urban context with real floods")

    # 2 × 4 grid of confounder cards
    categories = [
        ("River",          "399 images", "Water texture, reflections, debris — #1 confounder (9.2% FP rate)"),
        ("Swimming Pool",  "28 val",     "Clear water, edge patterns; small sample limits analysis"),
        ("Wet / Rain Road","105 val",    "Surface reflections and standing water resemble shallow floods"),
        ("Park / Walkway", "82 val",     "Waterlogged grass and paths trigger texture-based classifiers"),
        ("Building",       "55 val",     "Contextual co-occurrence with floods; dark backgrounds"),
        ("Vehicle",        "50 val",     "Water reflections under vehicles; rain-slicked surfaces"),
        ("Animal",         "67 val",     "Low-level texture patterns occasionally activate flood features"),
        ("Plant",          "34 val",     "Dense vegetation with moisture can resemble flooded ground"),
    ]
    HIGHLIGHT = {0, 1}  # river and pool highlighted

    cols, rows = 4, 2
    cw = Inches(2.9)
    ch = Inches(1.55)
    hgap, vgap = Inches(0.28), Inches(0.32)
    start_x = (W - (cols * cw + (cols - 1) * hgap)) / 2
    start_y = Inches(1.5)

    for i, (cat, count, desc) in enumerate(categories):
        col_i = i % cols
        row_i = i // cols
        lx = start_x + col_i * (cw + hgap)
        ty = start_y + row_i * (ch + vgap)
        highlight = i in HIGHLIGHT
        fill  = RGBColor(0xE8, 0xF1, 0xF8) if highlight else BOX_BG
        border = OCEAN if highlight else RULE
        bd_pt  = 1.5 if highlight else 0.75
        box(sl, lx, ty, cw, ch, fill=fill, border=border, border_pt=bd_pt)
        txt(sl, cat, lx + Inches(0.12), ty + Inches(0.08), cw - Inches(0.24), Inches(0.3),
            font="Calibri", size=12, bold=True, color=OCEAN if highlight else STEEL)
        txt(sl, count, lx + cw - Inches(0.85), ty + Inches(0.08), Inches(0.72), Inches(0.28),
            font="Calibri", size=10, color=AMBER if highlight else STEEL, align=PP_ALIGN.RIGHT)
        txt(sl, desc, lx + Inches(0.12), ty + Inches(0.38), cw - Inches(0.24), Inches(1.08),
            font="Calibri", size=10, color=STEEL, wrap=True)

    # Legend row
    txt(sl, "Highlighted = primary confounders targeted by Hard Negative Mining (HNM)",
        Inches(0.55), Inches(6.95), Inches(12.2), Inches(0.32),
        font="Calibri", size=10, color=OCEAN)


# ── SLIDE 5 — PR-AUC vs ROC-AUC (figures + comparison table) ─────────────────

def slide_05_metrics(prs):
    sl = blank(prs)
    title_line(sl,
               "Standard metrics hide recall failure — PR-AUC reveals what ROC-AUC conceals",
               "EfficientNetB0 outperforms ResNet50 on every metric: PR-AUC, recall, and accuracy")

    fig_top = Inches(1.42)
    fig_h   = Inches(3.8)
    add_img(sl, FIG["eff_bce_prroc"], Inches(0.4),  fig_top, h=fig_h)
    add_img(sl, FIG["res_bce_prroc"], Inches(6.85), fig_top, h=fig_h)

    txt(sl, "EfficientNetB0 + BCE",
        Inches(0.4), fig_top + fig_h + Inches(0.06), Inches(6.2), Inches(0.3),
        font="Calibri", size=11, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)
    txt(sl, "ResNet50 + BCE",
        Inches(6.85), fig_top + fig_h + Inches(0.06), Inches(6.0), Inches(0.3),
        font="Calibri", size=11, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)

    # Comparison table using native PowerPoint table
    tbl_headers = ["Metric", "EfficientNetB0", "ResNet50 (Phase 1, Epoch 8)", "Gap / note"]
    tbl_rows = [
        ("PR-AUC (primary ↑)",  "0.9976", "0.9614", "−3.6 pts — reveals recall collapse"),
        ("ROC-AUC ↑",           "0.9985", "0.9728", "−2.6 pts — looks 'nearly as good'"),
        ("Missed floods (FN ↓)", "7 / 322  (2.2%)", "63 / 322  (19.6%)", "9× more missed"),
        ("Accuracy ↑",           "98.3%",  "90.2%",  "EfficientNet leads on all metrics"),
    ]
    col_ws = [Inches(2.9), Inches(2.9), Inches(3.3), Inches(3.1)]
    make_table(sl, tbl_headers, tbl_rows,
               left=Inches(0.4), top=Inches(5.62), width=Inches(12.5), height=Inches(1.62),
               col_widths=col_ws, hdr_size=11, data_size=10)


# ── SLIDE 6 — Dataset (native PowerPoint tables) ─────────────────────────────

def slide_06_dataset(prs):
    sl = blank(prs)
    title_line(sl,
               "Dataset: 4,099 street-level images from two public sources, stratified 80/20 split",
               "USF FloodingDataset (flood severity + 7 non-flood categories)  +  RIWA river images (Wagner et al. 2023)")

    # ── Left table: USF FloodingDataset ──────────────────────────────────────
    txt(sl, "USF FloodingDataset",
        Inches(0.55), Inches(1.5), Inches(5.7), Inches(0.38),
        font="Cambria", size=14, bold=True, color=OCEAN)

    usf_rows = [
        ("Flood — Major",           "796 images"),
        ("Flood — Moderate",        "301 images"),
        ("Flood — Minor",           "516 images"),
        ("Non-flood (7 categories)","2,087 images"),
        ("Total",                   "3,700 images"),
    ]
    usf_tbl = make_table(sl, ["Category", "Count"], usf_rows,
                         left=Inches(0.55), top=Inches(1.92), width=Inches(5.7), height=Inches(2.55),
                         col_widths=[Inches(3.8), Inches(1.9)],
                         hdr_size=11, data_size=11, highlight_last=True)

    # ── Right table: RIWA River Dataset ──────────────────────────────────────
    txt(sl, "RIWA River Dataset  (Wagner et al. 2023)",
        Inches(7.1), Inches(1.5), Inches(5.7), Inches(0.38),
        font="Cambria", size=14, bold=True, color=OCEAN)

    riwa_rows = [
        ("River scenes",       "399 images"),
        ("Role",               "Water confounder category"),
        ("Source",             "European river monitoring"),
        ("Label",              "Non-flood / River"),
        ("Combined non-flood", "2,486 images"),
    ]
    riwa_tbl = make_table(sl, ["Property", "Value"], riwa_rows,
                          left=Inches(7.1), top=Inches(1.92), width=Inches(5.7), height=Inches(2.55),
                          col_widths=[Inches(2.8), Inches(2.9)],
                          hdr_size=11, data_size=11, highlight_last=True)

    # ── Stats strip — native table ────────────────────────────────────────────
    stats_headers = ["Images", "Split", "Flood prevalence", "Confounder categories"]
    stats_rows = [
        ("4,099 unique\n(55 duplicates removed)",
         "3,280 train  |  819 val",
         "39.3% — preserved in\nboth splits",
         "8 categories labeled\nfor per-category FP"),
    ]
    make_table(sl, stats_headers, stats_rows,
               left=Inches(0.55), top=Inches(5.22), width=Inches(12.3), height=Inches(1.08),
               col_widths=[Inches(3.1), Inches(3.0), Inches(3.1), Inches(3.1)],
               hdr_size=11, data_size=10)


# ── SLIDE 7 — Two-phase fine-tuning (flowchart) ───────────────────────────────

def slide_07_finetune(prs):
    sl = blank(prs)
    title_line(sl,
               "Two-phase progressive fine-tuning avoids catastrophic forgetting",
               "Phase 1 → head + top layers only (M1 checkpoint)  ·  Phase 2 → deeper layers (M2 checkpoint, deployed)")

    # Central flowchart — 5 steps linear, left to right
    nodes = [
        ("ImageNet\npre-trained\nweights", BOX_BG, RULE),
        ("Phase 1\nFreeze all but\ntop 30 layers\nLR = 1e-3\n≤ 30 epochs", RGBColor(0xE8, 0xF1, 0xF8), OCEAN),
        ("Checkpoint M1\n(mining source)", RGBColor(0xFF, 0xF5, 0xE8), AMBER),
        ("Phase 2\nFreeze first\n50 layers\nLR = 1e-4\n≤ 50 epochs", RGBColor(0xE8, 0xF1, 0xF8), OCEAN),
        ("Checkpoint M2\n(final model)", RGBColor(0xE8, 0xF5, 0xEC), RGBColor(0x2E, 0x7D, 0x32)),
    ]
    nw, nh = Inches(2.1), Inches(1.5)
    gap_x  = Inches(0.32)
    total  = len(nodes) * nw + (len(nodes) - 1) * gap_x
    sx     = (W - total) / 2
    ny     = Inches(2.2)

    for i, (label, fill, border) in enumerate(nodes):
        lx = sx + i * (nw + gap_x)
        bd_pt = 1.5 if border in (OCEAN, AMBER, RGBColor(0x2E, 0x7D, 0x32)) else 0.75
        box(sl, lx, ny, nw, nh, fill=fill, border=border, border_pt=bd_pt)
        color = OCEAN if border == OCEAN else (AMBER if border == AMBER else STEEL)
        txt(sl, label, lx, ny, nw, nh,
            font="Calibri", size=11, bold=(border != RULE),
            color=color, align=PP_ALIGN.CENTER)
        if i < len(nodes) - 1:
            arrow_right(sl, lx + nw, ny + nh / 2)

    # Architecture comparison table below
    rule(sl, Inches(0.55), Inches(4.1), Inches(12.2))
    txt(sl, "Architectures evaluated",
        Inches(0.55), Inches(4.22), Inches(4.0), Inches(0.32),
        font="Cambria", size=13, bold=True, color=OCEAN)

    arch_rows = [
        ("Model",         "Parameters",  "Input preprocessing",                  "Role"),
        ("EfficientNetB0","5.3 M",       "Raw [0, 255] — built-in rescaling",    "Primary model"),
        ("ResNet50",      "25.6 M",      "ImageNet mean subtraction",             "Comparison baseline"),
    ]
    col_xs = [Inches(0.55), Inches(3.2), Inches(5.4), Inches(9.8)]
    col_ws = [Inches(2.5),  Inches(2.0), Inches(4.2), Inches(2.8)]
    ry = Inches(4.6)
    for i, row in enumerate(arch_rows):
        if i % 2 == 1:
            nobox(sl, Inches(0.5), ry - Inches(0.02), Inches(12.3), Inches(0.32), ROW_ALT)
        for j, cell in enumerate(row):
            txt(sl, cell, col_xs[j], ry, col_ws[j], Inches(0.28),
                font="Calibri", size=12,
                bold=(i == 0), color=OCEAN if i == 0 else STEEL)
        ry += Inches(0.34)

    txt(sl,
        "Loss functions: Binary Cross-Entropy (BCE)  ·  Focal loss (γ = 2.0)  "
        "·  Class weights via compute_class_weight('balanced')",
        Inches(0.55), Inches(5.78), Inches(12.2), Inches(0.35),
        font="Calibri", size=11, color=STEEL)


# ── SLIDE 8 — P1-HNM pipeline (4-step flowchart) ─────────────────────────────

def slide_08_hnm_pipeline(prs):
    sl = blank(prs)
    title_line(sl,
               "Phase-1 Hard Negative Mining (P1-HNM): target confounders before memorization erases the signal",
               "By Phase 2, training FP rate → ~0% — no hard negatives remain. Phase 1 checkpoint still shows genuine confusion.")

    # Key insight box at top
    nobox(sl, Inches(0.55), Inches(1.42), Inches(12.2), Inches(0.62), fill=RGBColor(0xF2, 0xF6, 0xF9))
    txt(sl,
        "Key insight: Phase 2 memorizes non-flood training images → near-zero FP on train (EfficientNetB0: 0.31% river FP). "
        "Phase 1 checkpoint (top 30 layers only) retains genuine uncertainty on rivers and pools — that is the mining signal.",
        Inches(0.7), Inches(1.5), Inches(12.0), Inches(0.52),
        font="Calibri", size=11, color=STEEL)

    # 4 pipeline stages
    stages = [
        ("①  Identify",
         ["Run M1 (Phase 1\ncheckpoint) on\ntrain non-flood",
          "Flag categories with\nFP rate > 5%\n→ River: 9.2% ✓"]),
        ("②  Rank & Select",
         ["Sort flagged images\nby predicted flood\nprobability",
          "Take top 10%\n(hardest candidates)\nby difficulty"]),
        ("③  Augment",
         ["5× augmentation\non selected hard\nnegatives",
          "Adds difficulty-\nweighted non-flood\nexamples to train set"]),
        ("④  Retrain",
         ["Start from M2\n(Phase 2 checkpoint)",
          "Train with enriched\nset → P1-HNM\nfinal model"]),
    ]
    sw, sh = Inches(2.7), Inches(2.8)
    sx = Inches(0.55) + (W - Inches(0.55) * 2 - len(stages) * sw - Inches(0.3) * 3) / 2
    sy = Inches(2.2)

    for i, (stage_title, items) in enumerate(stages):
        lx = sx + i * (sw + Inches(0.3))
        box(sl, lx, sy, sw, sh, fill=BOX_BG, border=OCEAN if i == 3 else RULE,
            border_pt=1.5 if i == 3 else 0.75)
        txt(sl, stage_title, lx, sy + Inches(0.08), sw, Inches(0.36),
            font="Calibri", size=12, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)
        rule(sl, lx + Inches(0.15), sy + Inches(0.46), sw - Inches(0.3))
        iy = sy + Inches(0.56)
        for item in items:
            txt(sl, item, lx + Inches(0.12), iy, sw - Inches(0.24), Inches(0.78),
                font="Calibri", size=10, color=STEEL, align=PP_ALIGN.CENTER)
            iy += Inches(0.85)
        if i < len(stages) - 1:
            arrow_right(sl, lx + sw, sy + sh / 2)

    # Ablation controls table
    rule(sl, Inches(0.55), Inches(5.25), Inches(12.2))
    txt(sl, "Ablation controls (all compared to P1-HNM in final results)",
        Inches(0.55), Inches(5.35), Inches(8.0), Inches(0.3),
        font="Calibri", size=11, bold=True, color=OCEAN)

    abl_rows = [
        ("Condition",        "What changes",                              "Controls for"),
        ("P1-HNM",           "Top 10% hardest negatives by Phase 1 prob","— reference condition —"),
        ("No injection",     "Retrain same epochs, no extra data",        "Additional training compute"),
        ("Random injection", "Same count, randomly sampled",              "Category exposure vs. difficulty ranking"),
    ]
    col_xs2 = [Inches(0.55), Inches(3.5), Inches(8.5)]
    col_ws2 = [Inches(2.8),  Inches(4.8), Inches(4.2)]
    ry = Inches(5.72)
    for i, row in enumerate(abl_rows):
        if i % 2 == 1:
            nobox(sl, Inches(0.5), ry - Inches(0.02), Inches(12.3), Inches(0.3), ROW_ALT)
        for j, cell in enumerate(row):
            txt(sl, cell, col_xs2[j], ry, col_ws2[j], Inches(0.28),
                font="Calibri", size=11,
                bold=(i == 0 or (i == 1 and j == 0)),
                color=OCEAN if i == 0 else STEEL)
        ry += Inches(0.3)


# ── SLIDE 9 — Confusion matrix figures ────────────────────────────────────────

def slide_09_prroc_result(prs):
    sl = blank(prs)
    title_line(sl,
               "EfficientNetB0 misses 9× fewer floods — confusion matrices confirm the gap",
               "EfficientNet (Epoch 20): Acc 98.3%, Recall 97.8%   ·   ResNet50 (Phase 1, Epoch 8): Acc 90.2%, Recall 80.4%")

    fig_top = Inches(1.42)
    fig_h   = Inches(4.1)
    add_img(sl, FIG["eff_bce_cm"], Inches(0.35),  fig_top, h=fig_h)
    add_img(sl, FIG["res_bce_cm"], Inches(6.85), fig_top, h=fig_h)

    txt(sl, "EfficientNetB0 BCE  —  7 FN + 7 FP  —  Acc: 98.3%,  Recall: 97.8%",
        Inches(0.35), fig_top + fig_h + Inches(0.05), Inches(6.2), Inches(0.28),
        font="Calibri", size=11, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)
    txt(sl, "ResNet50 BCE  —  63 FN + 17 FP  —  Acc: 90.2%,  Recall: 80.4%",
        Inches(6.85), fig_top + fig_h + Inches(0.05), Inches(5.8), Inches(0.28),
        font="Calibri", size=11, bold=True, color=AMBER, align=PP_ALIGN.CENTER)

    rule(sl, Inches(0.35), Inches(5.9), Inches(12.6))

    note_rows = [
        ("EfficientNetB0 errors are symmetric boundary cases (ambiguous river/flood scenes).",),
        ("ResNet50 (Phase 1, Epoch 8) shows recall collapse — 63 clear flood scenes miscalibrated as non-flood.",),
        ("→ EfficientNetB0 is the right architecture for HNM retraining.",),
    ]
    ny = Inches(6.05)
    for (line,) in note_rows:
        txt(sl, line, Inches(0.5), ny, Inches(12.3), Inches(0.3),
            font="Calibri", size=11, color=STEEL)
        ny += Inches(0.32)


# ── SLIDE 10 — Confounder FP heatmap (all conditions) ─────────────────────────

def slide_10_confounder_fp(prs):
    sl = blank(prs)
    title_line(sl,
               "River is the only confounder — all other 7 categories are 0% FP across every condition",
               "Per-category FP rate (%) — Val set, seed 42, τ = 0.5  ·  Heatmap covers all 6 trained models")

    # ── Heatmap — full-width centered ────────────────────────────────────────
    img = add_img(sl, FIG["fp_heatmap"], Inches(0), Inches(1.45), h=Inches(4.5))
    if img:
        img.left = int((W - img.width) / 2)

    # ── Key takeaway table ────────────────────────────────────────────────────
    tbl_headers = ["Condition", "River FP rate", "All other categories"]
    tbl_rows = [
        ("EfficientNet BCE Baseline", "9.2%  (7 / 76)", "0% across all 7"),
        ("EfficientNet BCE HNM",      "9.2%  (7 / 76)", "0% across all 7"),
        ("Ext. Training",             "6.6%  (5 / 76)", "0% across all 7"),
        ("Random Inject",             "9.2%  (7 / 76)", "0% across all 7"),
        ("ResNet50 BCE Baseline",     "3.9%  (3 / 76)", "0% across all 7"),
        ("ResNet50 BCE HNM",          "5.3%  (4 / 76)", "0% across all 7"),
    ]
    make_table(sl, tbl_headers, tbl_rows,
               left=Inches(0.55), top=Inches(6.1), width=Inches(12.3), height=Inches(1.2),
               col_widths=[Inches(4.2), Inches(3.0), Inches(5.1)],
               hdr_size=10, data_size=9)


# ── SLIDE 11 — HNM ablation table (all values) ───────────────────────────────

def slide_11_hnm_table(prs):
    sl = blank(prs)
    title_line(sl,
               "Difficulty-ranked Phase-1 mining achieves best accuracy — all conditions compared",
               "EfficientNetB0 BCE  ·  Val set (seed 42)  ·  Full metrics for every ablation condition")

    # ── Main comparison table with all values filled in ────────────────────────
    headers = ["Condition", "Val Accuracy", "Flood Recall",
               "Missed Floods (FN)", "River FP rate", "vs. Baseline"]
    table_rows = [
        ("BCE Baseline",      "98.29%", "97.8%", "7 / 322  (2.2%)",  "9.2%  (7/76)",  "—"),
        ("Random injection",  "98.41%", "98.6%", "5 / 322  (1.6%)",  "9.2%  (7/76)",  "+0.12 pp"),
        ("Extended training", "98.66%", "99.0%", "3 / 322  (0.9%)",  "6.6%  (5/76)",  "+0.37 pp"),
        ("P1-HNM  ★",        "98.78%", "98.6%", "5 / 322  (1.6%)",  "9.2%  (7/76)",  "+0.49 pp"),
    ]
    col_ws = [Inches(2.45), Inches(1.7), Inches(1.55), Inches(2.45), Inches(1.9), Inches(1.9)]
    make_table(sl, headers, table_rows,
               left=Inches(0.45), top=Inches(1.62), width=Inches(12.45), height=Inches(2.8),
               col_widths=col_ws, hdr_size=11, data_size=11,
               highlight_last=True)

    # ── Key observations below table ──────────────────────────────────────────
    obs_headers = ["Observation", "Detail"]
    obs_rows = [
        ("Best accuracy → P1-HNM",
         "98.78% accuracy (top 10% hardest river negatives, difficulty-ranked by Phase 1 probability)"),
        ("Best recall → Extended training",
         "99.0% flood recall, only 3 missed floods — extra compute helps recall even without difficulty ranking"),
        ("River FP rate → Extended reduces to 6.6%",
         "Only extended training shows river FP improvement; HNM and random injection unchanged at 9.2%"),
        ("Statistical significance",
         "McNemar's test (Bonferroni-corrected): p > 0.05 for all pairs; N=76 river val images insufficient for tight CI"),
    ]
    make_table(sl, obs_headers, obs_rows,
               left=Inches(0.45), top=Inches(4.7), width=Inches(12.45), height=Inches(2.35),
               col_widths=[Inches(3.2), Inches(9.25)],
               hdr_size=11, data_size=10)


# ── SLIDE 12 — Conclusion (summary table) ─────────────────────────────────────

def slide_12_conclusion(prs):
    sl = blank(prs)
    title_line(sl, "Conclusion: four takeaways for deploying flood screeners in CRIS-HAZARD")

    rows = [
        ("PR-AUC is the right metric for recall-first screening",
         "3.6-pt PR-AUC gap between models is invisible in accuracy (98.3% vs 98.5%) and "
         "nearly invisible in ROC-AUC (2.6 pts). PR-AUC penalizes precision collapse at high recall."),
        ("EfficientNetB0 is the right screener — symmetric errors, not systematic collapse",
         "7 missed floods vs. ResNet50's 63. EfficientNet's errors are genuine boundary cases; "
         "ResNet's 63 FN include prototypical clear flood scenes."),
        ("River is the only statistically significant confounder",
         "9.2% FP rate (7/76, CI [3.8%, 17.7%]). All 7 other categories: 0% FP. "
         "Swimming pool is underpowered (N=28, upper CI = 12.3%)."),
        ("Phase-1 HNM strictly outperforms both ablations",
         "P1-HNM 98.78% > Extended 98.66% > Random 98.41% > Baseline 98.29%. "
         "Difficulty ranking adds value beyond compute or data quantity alone."),
    ]
    row_h = Inches(1.12)
    sy = Inches(1.2)
    for i, (title, detail) in enumerate(rows):
        fill = RGBColor(0xE8, 0xF1, 0xF8) if i % 2 == 0 else BOX_BG
        box(sl, Inches(0.5), sy, Inches(12.3), row_h, fill=fill, border=RULE, border_pt=0.5)
        # Number badge
        nobox(sl, Inches(0.5), sy, Inches(0.48), row_h,
              fill=OCEAN if i < 2 else RGBColor(0x4A, 0x60, 0x70))
        txt(sl, str(i + 1), Inches(0.5), sy + Inches(0.32), Inches(0.48), Inches(0.42),
            font="Cambria", size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        txt(sl, title, Inches(1.1), sy + Inches(0.1), Inches(11.6), Inches(0.38),
            font="Calibri", size=13, bold=True, color=OCEAN)
        txt(sl, detail, Inches(1.1), sy + Inches(0.5), Inches(11.6), Inches(0.56),
            font="Calibri", size=11, color=STEEL)
        sy += row_h + Inches(0.08)


# ── SLIDE 13 — Future directions (table) ──────────────────────────────────────

def slide_13_future(prs):
    sl = blank(prs)
    title_line(sl,
               "Five concrete next steps to deploy CRIS-HAZARD flood screening",
               "From statistical power to geographic generalization")

    directions = [
        ("Expand river validation set",
         "~300 river images for 80% power to detect a 3–4 pp FP reduction. Current 76-image set is underpowered."),
        ("Multi-seed robustness",
         "Re-run all experiments across 5 seeds with bootstrap CIs to verify seed-independence of HNM gains."),
        ("Severity-stratified miss rate",
         "Major vs. Minor flood miss rates have different operational costs. Report separately."),
        ("Geographic generalization",
         "Dataset is US-centric. Test on South/Southeast Asian and European urban flood imagery."),
        ("Operational threshold calibration",
         "τ=0.5 is not optimal. Calibrate against ≥ 99% recall target using the PR curve on a held-out test set."),
    ]
    row_h = Inches(0.88)
    sy = Inches(1.25)
    for i, (head, body) in enumerate(directions):
        fill = ROW_ALT if i % 2 == 0 else WHITE
        nobox(sl, Inches(0.5), sy, Inches(12.3), row_h, fill=fill)
        rule(sl, Inches(0.5), sy + row_h, Inches(12.3))
        txt(sl, str(i + 1), Inches(0.6), sy + Inches(0.24),
            Inches(0.38), Inches(0.38),
            font="Cambria", size=16, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)
        txt(sl, head, Inches(1.15), sy + Inches(0.1), Inches(11.5), Inches(0.34),
            font="Calibri", size=13, bold=True, color=OCEAN)
        txt(sl, body, Inches(1.15), sy + Inches(0.46), Inches(11.5), Inches(0.38),
            font="Calibri", size=11, color=STEEL)
        sy += row_h


# ── SLIDE 14 — Acknowledgments ────────────────────────────────────────────────

def slide_14_ack(prs):
    sl = blank(prs)
    title_line(sl, "Acknowledgments")

    thanks = [
        ("Advisor",             "Dr. Dixon — guidance and support throughout this project"),
        ("USF FloodingDataset", "Dataset authors — labeled flood severity imagery and non-flood category annotations"),
        ("RIWA Dataset",        "Wagner et al. (2023) — river scene images enriching the water confounder category"),
        ("CRIS-HAZARD project", "NSF-funded collaboration: USF, Georgia Tech, AT&T, Argonne National Laboratory"),
        ("Audience",            "Thank you for your time and questions"),
    ]
    sy = Inches(1.22)
    rh = Inches(0.88)
    for i, (head, body) in enumerate(thanks):
        fill = ROW_ALT if i % 2 == 0 else WHITE
        nobox(sl, Inches(0.5), sy, Inches(12.3), rh, fill=fill)
        rule(sl, Inches(0.5), sy + rh, Inches(12.3))
        txt(sl, head, Inches(0.65), sy + Inches(0.08), Inches(2.8), rh - Inches(0.14),
            font="Calibri", size=13, bold=True, color=OCEAN)
        txt(sl, body, Inches(3.6), sy + Inches(0.2), Inches(9.05), rh - Inches(0.28),
            font="Calibri", size=13, color=STEEL)
        sy += rh

    txt(sl, "Aanya Singh  ·  University of South Florida  ·  April 18, 2026",
        Inches(0.5), Inches(6.95), Inches(12.3), Inches(0.32),
        font="Calibri", size=11, color=RULE, align=PP_ALIGN.CENTER)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    prs = new_prs()
    builders = [
        (slide_01_title,            "Title"),
        (slide_02_crish_overview,   "CRIS-HAZARD big picture (flowchart)"),
        (slide_03_gap_flowchart,    "Gap in CRIS-HAZARD (two-column flowchart)"),
        (slide_04_confounders,      "Confounders grid (2×4 table)"),
        (slide_05_metrics,          "Metric problem — PR/ROC figures + table"),
        (slide_06_dataset,          "Dataset table"),
        (slide_07_finetune,         "Two-phase fine-tuning flowchart"),
        (slide_08_hnm_pipeline,     "P1-HNM 4-step flowchart"),
        (slide_09_prroc_result,     "Confusion matrix figures"),
        (slide_10_confounder_fp,    "Confounder FP figure + table"),
        (slide_11_hnm_table,        "HNM ablation comparison table"),
        (slide_12_conclusion,       "Conclusion summary table"),
        (slide_13_future,           "Future directions table"),
        (slide_14_ack,              "Acknowledgments"),
    ]

    print("Building slides...")
    for i, (fn, label) in enumerate(builders, 1):
        fn(prs)
        print(f"  {i:2d}/{len(builders)}  {label}")

    os.makedirs(os.path.dirname(OUT_PPTX), exist_ok=True)
    prs.save(OUT_PPTX)
    print(f"\nSaved → {OUT_PPTX}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
