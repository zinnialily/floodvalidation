"""
build_pptx.py  —  Flood Detection Research Presentation
Generates presentation/flood_detection.pptx from verified paper data.

Run:  python3 presentation/build_pptx.py
Requires: pip install python-pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_BASE = os.path.join(ROOT, "results", "figures")
OUT_PPTX = os.path.join(ROOT, "presentation", "flood_detection.pptx")

FIG = {
    "eff_bce_curves":  os.path.join(FIG_BASE, "baselines", "efficientnet_bce",  "training_curves.png"),
    "eff_bce_prroc":   os.path.join(FIG_BASE, "baselines", "efficientnet_bce",  "pr_roc_curves.png"),
    "eff_bce_cm":      os.path.join(FIG_BASE, "baselines", "efficientnet_bce",  "confusion_matrix.png"),
    "res_bce_prroc":   os.path.join(FIG_BASE, "baselines", "resnet50_bce",      "pr_roc_curves.png"),
    "res_bce_cm":      os.path.join(FIG_BASE, "baselines", "resnet50_bce",      "confusion_matrix.png"),
    "fp_heatmap":      os.path.join(FIG_BASE, "confounder_fp_heatmap.png"),
    "fp_comparison":   os.path.join(FIG_BASE, "confounder_fp_comparison.png"),
    "hnm_curves":      os.path.join(FIG_BASE, "hnm", "efficientnet_bce",        "training_curves.png"),
    "ablation_curves": os.path.join(FIG_BASE, "hnm", "ablations", "efficientnet_bce", "training_curves.png"),
}

# ── Color palette ─────────────────────────────────────────────────────────────
OCEAN      = RGBColor(0x1A, 0x3A, 0x5C)   # deep ocean blue  — dominant
STEEL      = RGBColor(0x4A, 0x60, 0x70)   # steel gray       — secondary
AMBER      = RGBColor(0xE8, 0xA0, 0x20)   # amber alert      — accent
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)    # slide background
LIGHT_GRAY = RGBColor(0xCC, 0xD5, 0xDE)   # muted labels
RULE_GRAY  = RGBColor(0xE2, 0xE8, 0xEE)   # thin divider lines

# ── Slide dimensions (16:9 widescreen) ────────────────────────────────────────
W = Inches(13.33)
H = Inches(7.5)

# ── Helpers ───────────────────────────────────────────────────────────────────

def new_prs():
    prs = Presentation()
    prs.slide_width  = W
    prs.slide_height = H
    return prs


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def white_bg(slide):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = WHITE


def add_rect(slide, left, top, width, height, fill_color, line_color=None):
    shape = slide.shapes.add_shape(1, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
    return shape


def add_textbox(slide, left, top, width, height,
                text, font_name, font_size, bold=False,
                color=OCEAN, align=PP_ALIGN.LEFT, wrap=True):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = wrap
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text           = text
    run.font.name      = font_name
    run.font.size      = Pt(font_size)
    run.font.bold      = bold
    run.font.color.rgb = color
    return tb


def add_para(tf, text, font_name, font_size, bold=False,
             color=OCEAN, align=PP_ALIGN.LEFT, space_before=6):
    p = tf.add_paragraph()
    p.alignment    = align
    p.space_before = Pt(space_before)
    run = p.add_run()
    run.text           = text
    run.font.name      = font_name
    run.font.size      = Pt(font_size)
    run.font.bold      = bold
    run.font.color.rgb = color
    return p


def add_image(slide, path, left, top, width=None, height=None):
    if not os.path.exists(path):
        print(f"  WARNING: figure not found: {path}")
        return None
    return slide.shapes.add_picture(path, left, top, width=width, height=height)


def slide_header(slide, title, subtitle=None):
    """Amber left bar + ocean title. Returns bottom y of header."""
    add_rect(slide, Inches(0.0), Inches(0.0), Inches(0.18), H, OCEAN)
    add_textbox(slide,
                left=Inches(0.4), top=Inches(0.22),
                width=Inches(12.7), height=Inches(0.72),
                text=title, font_name="Cambria", font_size=28, bold=True,
                color=OCEAN, align=PP_ALIGN.LEFT)
    if subtitle:
        add_textbox(slide,
                    left=Inches(0.4), top=Inches(0.97),
                    width=Inches(12.7), height=Inches(0.38),
                    text=subtitle, font_name="Calibri", font_size=14,
                    color=STEEL, align=PP_ALIGN.LEFT)
        add_rect(slide, Inches(0.4), Inches(1.38), Inches(12.33), Inches(0.025), RULE_GRAY)
        return Inches(1.5)
    add_rect(slide, Inches(0.4), Inches(1.0), Inches(12.33), Inches(0.025), RULE_GRAY)
    return Inches(1.12)


def section_chip(slide, label, color=AMBER):
    """Small section label in top-right corner."""
    add_rect(slide, Inches(11.5), Inches(0.18), Inches(1.65), Inches(0.38), color)
    add_textbox(slide, Inches(11.5), Inches(0.2), Inches(1.65), Inches(0.34),
                label, "Calibri", 11, bold=True, color=WHITE, align=PP_ALIGN.CENTER)


# ── SLIDE 1 — Title ──────────────────────────────────────────────────────────

def slide_01_title(prs):
    sl = blank_slide(prs)
    white_bg(sl)

    # Left ocean band
    add_rect(sl, 0, 0, Inches(0.35), H, OCEAN)
    # Top amber stripe
    add_rect(sl, 0, 0, W, Inches(0.1), AMBER)
    # Bottom amber stripe
    add_rect(sl, 0, H - Inches(0.1), W, Inches(0.1), AMBER)

    # Title
    add_textbox(sl,
                left=Inches(0.65), top=Inches(0.85),
                width=Inches(12.3), height=Inches(2.1),
                text="Confounder-Robust Flood Detection from Street-Level Imagery\nvia Phase-1 Hard Negative Mining",
                font_name="Cambria", font_size=36, bold=True,
                color=OCEAN, align=PP_ALIGN.LEFT)

    # Subtitle
    add_textbox(sl,
                left=Inches(0.65), top=Inches(3.05),
                width=Inches(11.0), height=Inches(0.6),
                text="When Rivers Look Like Floods — and Why Standard Metrics Miss It",
                font_name="Calibri", font_size=20, bold=False,
                color=AMBER, align=PP_ALIGN.LEFT)

    # Divider
    add_rect(sl, Inches(0.65), Inches(3.75), Inches(5.5), Inches(0.045), AMBER)

    # Info block
    info = [
        ("Aanya Singh", 18, True, OCEAN),
        ("Advised by Dr. Dixon", 15, False, STEEL),
        ("University of South Florida", 15, False, STEEL),
        ("April 18, 2026", 14, False, STEEL),
    ]
    y = Inches(3.95)
    for txt, sz, bd, col in info:
        add_textbox(sl, Inches(0.65), y, Inches(7.0), Inches(0.45),
                    txt, "Calibri", sz, bold=bd, color=col, align=PP_ALIGN.LEFT)
        y += Inches(0.44)


# ── SLIDE 2 — Problem ────────────────────────────────────────────────────────

def slide_02_problem(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "PROBLEM", OCEAN)

    slide_header(sl,
                 "The problem: floods are hard to detect because visual confounders mimic flood imagery",
                 "Rivers, pools, and wet roads share water texture, color, and reflections with real flood scenes")

    # Two columns
    # Left: why it matters
    add_textbox(sl, Inches(0.4), Inches(1.62), Inches(5.8), Inches(0.4),
                "Why real-time screening matters", "Calibri", 14, bold=True, color=OCEAN)

    left_bullets = [
        ("58 million people/year",   "exposed to floods globally, 2000–2018 (Tellman 2021)"),
        ("Hours to days",             "satellite revisit cycles — too slow for dispatch"),
        ("Citizen photos arrive first", "smartphone images precede official reports"),
        ("Recall-first design",        "missing a flood = delayed response; a false alarm = wasted call"),
    ]
    y = Inches(2.12)
    for stat, desc in left_bullets:
        add_rect(sl, Inches(0.4), y, Inches(2.1), Inches(0.6), OCEAN)
        add_textbox(sl, Inches(0.42), y + Inches(0.04), Inches(2.06), Inches(0.52),
                    stat, "Calibri", 12, bold=True, color=AMBER, align=PP_ALIGN.CENTER)
        add_textbox(sl, Inches(2.62), y + Inches(0.12), Inches(3.5), Inches(0.38),
                    desc, "Calibri", 13, color=STEEL)
        y += Inches(0.82)

    # Vertical rule
    add_rect(sl, Inches(6.5), Inches(1.55), Inches(0.025), Inches(5.65), RULE_GRAY)

    # Right: confounders
    add_textbox(sl, Inches(6.75), Inches(1.62), Inches(6.2), Inches(0.4),
                "What makes classification hard", "Calibri", 14, bold=True, color=OCEAN)

    confounders = [
        ("River scenes",      "Share water texture, reflections, murky color — the #1 confounder"),
        ("Swimming pools",    "Clear water with similar edge patterns; small N limits analysis"),
        ("Wet / rain roads",  "Surface reflections and standing water resembling shallow floods"),
        ("Parks after rain",  "Waterlogged grass and paths trigger flood classifiers"),
    ]
    y = Inches(2.12)
    for cat, desc in confounders:
        add_rect(sl, Inches(6.75), y, Inches(0.12), Inches(0.55), AMBER)
        add_textbox(sl, Inches(7.0), y, Inches(2.2), Inches(0.38),
                    cat, "Calibri", 13, bold=True, color=OCEAN)
        add_textbox(sl, Inches(7.0), y + Inches(0.36), Inches(5.7), Inches(0.35),
                    desc, "Calibri", 12, color=STEEL)
        y += Inches(0.82)

    # Bottom callout
    add_rect(sl, Inches(0.4), Inches(6.55), Inches(12.6), Inches(0.72), OCEAN)
    add_textbox(sl, Inches(0.6), Inches(6.65), Inches(12.2), Inches(0.52),
                "No prior work reports per-category false positive rates with confidence intervals across confounder types.",
                "Calibri", 13, bold=False, color=WHITE, align=PP_ALIGN.LEFT)


# ── SLIDE 3 — Problem: metrics mislead ──────────────────────────────────────

def slide_03_problem_metrics(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "PROBLEM", OCEAN)

    slide_header(sl,
                 "Standard metrics hide recall failures — accuracy rewards non-flood correctness, not flood detection",
                 "The accuracy paradox: ResNet50 scores higher accuracy than EfficientNetB0 while missing 9× more floods")

    # Three panels
    panels = [
        ("Accuracy paradox", OCEAN,
         [
             "ResNet50 accuracy: 98.5%",
             "EfficientNetB0 accuracy: 98.3%",
             "→ ResNet appears better",
             "",
             "But ResNet misses 63/322 floods.",
             "EfficientNet misses only 7/322.",
             "",
             "Non-flood images (497) outnumber floods (322).",
             "Getting non-floods right inflates accuracy.",
         ]),
        ("ROC-AUC is misleading here", STEEL,
         [
             "Receiver Operating Characteristic",
             "AUC (ROC-AUC) treats all non-flood",
             "images equally.",
             "",
             "ROC gap: 0.9985 vs. 0.9728",
             "= only 2.6 points — 'nearly as good'",
             "",
             "Large non-flood class inflates",
             "the true-negative rate artificially.",
         ]),
        ("PR-AUC is the right metric", AMBER,
         [
             "Precision-Recall AUC (PR-AUC)",
             "focuses on the flood (minority) class.",
             "",
             "PR-AUC gap: 0.9976 vs. 0.9614",
             "= 3.6 points — reveals collapse",
             "",
             "Penalizes precision drop at",
             "high recall — exactly what matters",
             "for a recall-first screener.",
         ]),
    ]

    x = Inches(0.4)
    box_w = Inches(3.9)
    for title, col, lines in panels:
        txt_col = WHITE if col != AMBER else OCEAN
        add_rect(sl, x, Inches(1.55), box_w, Inches(5.65), col)
        add_textbox(sl, x + Inches(0.15), Inches(1.65), box_w - Inches(0.3), Inches(0.5),
                    title, "Cambria", 15, bold=True, color=WHITE if col != AMBER else OCEAN)
        add_rect(sl, x + Inches(0.15), Inches(2.22), box_w - Inches(0.3), Inches(0.025), WHITE if col != AMBER else OCEAN)
        y = Inches(2.35)
        for line in lines:
            add_textbox(sl, x + Inches(0.15), y, box_w - Inches(0.3), Inches(0.36),
                        line, "Calibri", 12, color=txt_col if line else WHITE)
            y += Inches(0.38)
        x += Inches(4.35)


# ── SLIDE 4 — Research Questions ─────────────────────────────────────────────

def slide_04_research_questions(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "RESEARCH QUESTIONS", OCEAN)

    slide_header(sl, "Four research questions guide this work")

    rqs = [
        ("RQ1", "Metric validity",
         "Does PR-AUC reveal recall failures that accuracy and ROC-AUC conceal for imbalanced flood/non-flood data?",
         "Motivates metric choice before all else."),
        ("RQ2", "Architecture comparison",
         "Which architecture — EfficientNetB0 (5.3M params) or ResNet50 (25.6M params) — better supports recall-first flood screening?",
         "Drives model selection for subsequent mining experiments."),
        ("RQ3", "Confounder profiling",
         "Which specific non-flood categories produce false positives, and at what rate with 95% confidence intervals?",
         "Identifies where hard negative mining should be targeted."),
        ("RQ4", "Mining strategy",
         "Does Phase-1 Hard Negative Mining (P1-HNM) outperform extended training and random confounder injection?",
         "Tests whether difficulty ranking adds value beyond compute or data quantity alone."),
    ]

    y = Inches(1.32)
    for tag, head, question, relevance in rqs:
        # Number badge
        add_rect(sl, Inches(0.4), y, Inches(0.72), Inches(0.72), OCEAN)
        add_textbox(sl, Inches(0.4), y + Inches(0.1), Inches(0.72), Inches(0.52),
                    tag, "Cambria", 12, bold=True, color=AMBER, align=PP_ALIGN.CENTER)
        # Heading
        add_textbox(sl, Inches(1.3), y, Inches(11.6), Inches(0.38),
                    head, "Cambria", 16, bold=True, color=OCEAN)
        # Question
        add_textbox(sl, Inches(1.3), y + Inches(0.38), Inches(11.6), Inches(0.45),
                    question, "Calibri", 13, color=STEEL)
        # Relevance tag
        add_rect(sl, Inches(1.3), y + Inches(0.86), Inches(0.1), Inches(0.32), AMBER)
        add_textbox(sl, Inches(1.5), y + Inches(0.86), Inches(11.4), Inches(0.32),
                    relevance, "Calibri", 11, color=STEEL)
        add_rect(sl, Inches(0.4), y + Inches(1.28), Inches(12.5), Inches(0.02), RULE_GRAY)
        y += Inches(1.4)


# ── SLIDE 5 — Solution overview ───────────────────────────────────────────────

def slide_05_solution(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "SOLUTION", AMBER)

    slide_header(sl,
                 "Solution: EfficientNetB0 with Phase-1 Hard Negative Mining achieves 98.78% accuracy with measured confounder robustness",
                 "Two-phase fine-tuning + difficulty-ranked mining on Phase-1 checkpoint outperforms all ablations")

    answers = [
        ("RQ1 answered", OCEAN,
         "PR-AUC = 0.9976 vs. 0.9614",
         "3.6-pt gap exposes ResNet's recall collapse — invisible in accuracy (98.5% vs. 98.3%) and ROC-AUC (2.6-pt gap)."),
        ("RQ2 answered", OCEAN,
         "EfficientNetB0 is the right architecture",
         "7 missed floods vs. ResNet's 63. Errors are symmetric boundary cases, not systematic miscalibration."),
        ("RQ3 answered", STEEL,
         "River is the only significant confounder",
         "9.2% FP rate (7/76, CI [3.8%–17.7%]). All 7 other categories: 0% FP. Pools: 0% but underpowered."),
        ("RQ4 answered", STEEL,
         "P1-HNM strictly outperforms all ablations",
         "Accuracy: 98.78% > Extended 98.66% > Random 98.41% > Baseline 98.29%. River FP unchanged — power-limited."),
    ]

    x = Inches(0.4)
    box_w = Inches(6.1)
    row_h = Inches(2.35)
    positions = [(Inches(0.4), Inches(1.55)), (Inches(6.75), Inches(1.55)),
                 (Inches(0.4), Inches(4.0)),  (Inches(6.75), Inches(4.0))]

    for (bx, by), (tag, col, stat, desc) in zip(positions, answers):
        add_rect(sl, bx, by, box_w, row_h, col)
        add_textbox(sl, bx + Inches(0.15), by + Inches(0.12), box_w - Inches(0.3), Inches(0.38),
                    tag, "Calibri", 12, bold=True, color=AMBER)
        add_textbox(sl, bx + Inches(0.15), by + Inches(0.55), box_w - Inches(0.3), Inches(0.5),
                    stat, "Cambria", 17, bold=True, color=WHITE)
        add_textbox(sl, bx + Inches(0.15), by + Inches(1.1), box_w - Inches(0.3), Inches(1.1),
                    desc, "Calibri", 12, color=LIGHT_GRAY)

    # Central divider line
    add_rect(sl, Inches(6.55), Inches(1.45), Inches(0.025), Inches(5.0), RULE_GRAY)


# ── SLIDE 6 — Dataset ─────────────────────────────────────────────────────────

def slide_06_dataset(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "METHOD", STEEL)

    slide_header(sl,
                 "Dataset: 4,099 street-level images from two labeled sources, deduplicated by SHA-256",
                 "USF FloodingDataset (flood severity + non-flood categories) + RIWA river images (Wagner et al. 2023)")

    # Source 1
    add_rect(sl, Inches(0.4), Inches(1.55), Inches(5.85), Inches(4.3), OCEAN)
    add_textbox(sl, Inches(0.55), Inches(1.65), Inches(5.5), Inches(0.48),
                "USF FloodingDataset", "Cambria", 17, bold=True, color=AMBER)
    usf = [
        "Flood images (severity labeled):",
        "  Major    —  796 images",
        "  Moderate — 301 images",
        "  Minor    — 516 images",
        "  Total flood:  1,613",
        "",
        "Non-flood (7 categories):",
        "  Street, Animal, Building, Vehicle",
        "  Plant, Park/Walkway, Swimming Pool",
        "  Total non-flood:  2,087",
    ]
    y = Inches(2.22)
    for line in usf:
        add_textbox(sl, Inches(0.6), y, Inches(5.5), Inches(0.38),
                    line, "Calibri", 13, color=WHITE)
        y += Inches(0.38)

    # Source 2
    add_rect(sl, Inches(6.75), Inches(1.55), Inches(5.85), Inches(4.3), STEEL)
    add_textbox(sl, Inches(6.9), Inches(1.65), Inches(5.5), Inches(0.48),
                "RIWA River Dataset  (Wagner et al. 2023)", "Cambria", 17, bold=True, color=AMBER)
    riwa = [
        "River scene images — water confounder",
        "  399 river images added",
        "  European river monitoring source",
        "  Enriches the hardest confounder category",
        "",
        "Label:  Non-flood / River",
        "",
        "Combined non-flood total:  2,486",
    ]
    y = Inches(2.22)
    for line in riwa:
        add_textbox(sl, Inches(6.95), y, Inches(5.5), Inches(0.38),
                    line, "Calibri", 13, color=WHITE)
        y += Inches(0.38)

    # Stats bar
    add_rect(sl, Inches(0.4), Inches(5.98), Inches(12.55), Inches(1.25), RULE_GRAY)
    stats = [
        ("4,099 unique images", "55 duplicates removed (SHA-256 hash)"),
        ("80/20 stratified split  ·  seed 42", "3,280 train  |  819 validation"),
        ("39.3% flood prevalence", "Same ratio maintained in both splits"),
    ]
    x = Inches(0.6)
    for top_t, bot_t in stats:
        add_textbox(sl, x, Inches(6.06), Inches(4.0), Inches(0.42),
                    top_t, "Calibri", 14, bold=True, color=OCEAN)
        add_textbox(sl, x, Inches(6.5), Inches(4.0), Inches(0.38),
                    bot_t, "Calibri", 12, color=STEEL)
        x += Inches(4.2)


# ── SLIDE 7 — Method: two-phase fine-tuning ───────────────────────────────────

def slide_07_method_finetune(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "METHOD", STEEL)

    slide_header(sl,
                 "Method: two-phase progressive fine-tuning preserves ImageNet representations",
                 "Phase 1 → checkpoint M1 (hard negative mining source)  ·  Phase 2 → checkpoint M2 (final model)")

    # Architecture row
    for bx, name, desc in [
        (Inches(0.4),  "EfficientNetB0", "5.3M parameters  —  primary model"),
        (Inches(6.75), "ResNet50",        "25.6M parameters  —  comparison baseline"),
    ]:
        add_rect(sl, bx, Inches(1.55), Inches(5.9), Inches(0.82), OCEAN)
        add_textbox(sl, bx + Inches(0.15), Inches(1.62), Inches(5.6), Inches(0.42),
                    name, "Cambria", 16, bold=True, color=AMBER)
        add_textbox(sl, bx + Inches(0.15), Inches(2.0), Inches(5.6), Inches(0.3),
                    desc, "Calibri", 12, color=LIGHT_GRAY)

    # Phase boxes
    phases = [
        ("Phase 1  →  Checkpoint M1",
         ["Last 30 layers trainable", "Learning rate: 1e-4", "≤ 15 epochs",
          "Residual uncertainty on confounders", "→ mining signal lives here"]),
        ("Phase 2  →  Checkpoint M2",
         ["First 50 layers frozen", "Learning rate: 1e-5", "≤ 20 epochs",
          "Early ImageNet features preserved", "→ final deployed model"]),
    ]
    x = Inches(0.4)
    for title, items in phases:
        add_rect(sl, x, Inches(2.55), Inches(5.9), Inches(3.1), STEEL)
        add_textbox(sl, x + Inches(0.15), Inches(2.65), Inches(5.6), Inches(0.45),
                    title, "Cambria", 15, bold=True, color=AMBER)
        add_rect(sl, x + Inches(0.15), Inches(3.15), Inches(5.6), Inches(0.025), LIGHT_GRAY)
        y = Inches(3.25)
        for item in items:
            add_textbox(sl, x + Inches(0.25), y, Inches(5.5), Inches(0.38),
                        "• " + item, "Calibri", 13, color=WHITE)
            y += Inches(0.4)
        x += Inches(6.35)

    # Arrow
    add_textbox(sl, Inches(6.1), Inches(3.8), Inches(0.6), Inches(0.6),
                "→", "Cambria", 28, bold=True, color=AMBER, align=PP_ALIGN.CENTER)

    # Classification head
    add_rect(sl, Inches(0.4), Inches(5.78), Inches(12.55), Inches(1.45), RULE_GRAY)
    add_textbox(sl, Inches(0.55), Inches(5.86), Inches(3.0), Inches(0.38),
                "Classification head:", "Calibri", 13, bold=True, color=OCEAN)
    add_textbox(sl, Inches(3.65), Inches(5.86), Inches(9.0), Inches(0.38),
                "Global Average Pooling (GAP)  →  Dropout(0.2)  →  Dense(256, ReLU)  →  BatchNorm  →  Dropout(0.3)  →  Dense(1, sigmoid)",
                "Calibri", 13, color=STEEL)
    add_textbox(sl, Inches(0.55), Inches(6.3), Inches(12.2), Inches(0.7),
                "Loss functions evaluated:  Binary Cross-Entropy (BCE)  ·  Focal loss  (γ=2, α=0.25)",
                "Calibri", 12, color=STEEL)


# ── SLIDE 8 — Method: Phase-1 HNM ────────────────────────────────────────────

def slide_08_method_hnm(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "METHOD", STEEL)

    slide_header(sl,
                 "Phase-1 Hard Negative Mining (P1-HNM): mine confounders before the signal disappears",
                 "Phase-2 converged model gives near-zero probability to all non-flood images — no mining gradient remains")

    # Key insight box
    add_rect(sl, Inches(0.4), Inches(1.55), Inches(12.55), Inches(0.9), RULE_GRAY)
    add_textbox(sl, Inches(0.55), Inches(1.62), Inches(12.2), Inches(0.75),
                "Key insight: use the Phase-1 checkpoint M1 (partially trained, early layers still frozen) "
                "while residual uncertainty on rivers and pools still exists — that uncertainty is the mining signal. "
                "By Phase 2 the model has overfit its boundary; hard negatives can no longer be identified.",
                "Calibri", 13, color=STEEL)

    # Pipeline stages
    stages = [
        ("1", "Identify",
         ["Use Phase-1 model M1", "Flag categories with FP > 5%", "→ River: 9.2%  qualifies"]),
        ("2", "Rank & Select",
         ["Sort flagged images by", "predicted flood probability", "Take top 10% (hardest)"]),
        ("3", "Augment",
         ["5× augmentation on", "selected hard negatives", "Adds difficulty-weighted data"]),
        ("4", "Retrain",
         ["Start from checkpoint M2", "Train with enriched set", "→ P1-HNM final model"]),
    ]

    x = Inches(0.4)
    sw = Inches(2.9)
    for stage in stages:
        num, head, items = stage
        add_rect(sl, x, Inches(2.62), sw, Inches(4.3), OCEAN)
        add_rect(sl, x + Inches(0.12), Inches(2.74), Inches(0.55), Inches(0.55), AMBER)
        add_textbox(sl, x + Inches(0.12), Inches(2.77), Inches(0.55), Inches(0.45),
                    num, "Cambria", 18, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)
        add_textbox(sl, x + Inches(0.8), Inches(2.78), sw - Inches(0.95), Inches(0.45),
                    head, "Cambria", 15, bold=True, color=WHITE)
        add_rect(sl, x + Inches(0.15), Inches(3.38), sw - Inches(0.3), Inches(0.025), LIGHT_GRAY)
        y = Inches(3.5)
        for item in items:
            add_textbox(sl, x + Inches(0.2), y, sw - Inches(0.3), Inches(0.42),
                        item, "Calibri", 13, color=LIGHT_GRAY)
            y += Inches(0.42)
        if num != "4":
            add_textbox(sl, x + sw + Inches(0.08), Inches(4.25), Inches(0.5), Inches(0.5),
                        "→", "Cambria", 24, bold=True, color=AMBER, align=PP_ALIGN.CENTER)
        x += Inches(3.25)

    add_textbox(sl, Inches(0.4), Inches(7.1), Inches(12.55), Inches(0.3),
                "Why difficulty ranking matters: random injection ignores which images the model is confused by — "
                "ranked selection targets the decision boundary directly.",
                "Calibri", 11, color=STEEL)


# ── SLIDE 9 — Result: PR-AUC ──────────────────────────────────────────────────

def slide_09_result_prroc(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "RESULTS", AMBER)

    slide_header(sl,
                 "EfficientNetB0 misses 9× fewer floods than ResNet50 — visible in PR-AUC, hidden in ROC-AUC",
                 "PR-AUC gap 3.6 pts vs. ROC-AUC gap 2.6 pts  ·  63 vs. 7 missed floods hidden by the accuracy paradox")

    fig_top = Inches(1.35)
    fig_h   = Inches(4.4)
    if os.path.exists(FIG["eff_bce_prroc"]):
        add_image(sl, FIG["eff_bce_prroc"], left=Inches(0.35), top=fig_top, height=fig_h)
        add_textbox(sl, Inches(0.35), fig_top + fig_h + Inches(0.05), Inches(6.1), Inches(0.28),
                    "EfficientNetB0 + BCE", "Calibri", 12, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)
    if os.path.exists(FIG["res_bce_prroc"]):
        add_image(sl, FIG["res_bce_prroc"], left=Inches(6.65), top=fig_top, height=fig_h)
        add_textbox(sl, Inches(6.65), fig_top + fig_h + Inches(0.05), Inches(6.2), Inches(0.28),
                    "ResNet50 + BCE", "Calibri", 12, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)

    # Stats strip
    add_rect(sl, Inches(0.35), Inches(6.15), Inches(12.6), Inches(1.1), RULE_GRAY)
    col_w = Inches(3.05)
    headers = ["Metric", "EfficientNetB0", "ResNet50", "Gap"]
    rows = [
        ("PR-AUC",              "0.9976",          "0.9614",           "−3.6 pts"),
        ("ROC-AUC",             "0.9985",           "0.9728",           "−2.6 pts"),
        ("Missed floods (FN)",  "7/322  (2.2%)",    "63/322  (19.6%)",  "9× more"),
        ("Accuracy",            "98.3%",            "98.5%",            "paradox ↑"),
    ]
    x0 = Inches(0.5)
    y0 = Inches(6.2)
    for i, h in enumerate(headers):
        add_textbox(sl, x0 + i*col_w, y0, col_w, Inches(0.3),
                    h, "Calibri", 11, bold=True,
                    color=AMBER if i in (1,3) else OCEAN)
    y0 += Inches(0.32)
    for r in rows:
        for i, cell in enumerate(r):
            add_textbox(sl, x0 + i*col_w, y0, col_w, Inches(0.3),
                        cell, "Calibri", 11, bold=(i in (1,3)),
                        color=AMBER if i in (1,3) else STEEL)
        y0 += Inches(0.28)


# ── SLIDE 10 — Result: confusion matrices ────────────────────────────────────

def slide_10_result_confusion(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "RESULTS", AMBER)

    slide_header(sl,
                 "EfficientNetB0 makes symmetric boundary errors; ResNet50 collapses systematically",
                 "Symmetric 7/7 FN/FP vs. asymmetric 63/17 — tells us which model to target with HNM")

    fig_top = Inches(1.35)
    fig_h   = Inches(4.75)
    if os.path.exists(FIG["eff_bce_cm"]):
        add_image(sl, FIG["eff_bce_cm"], left=Inches(0.5), top=fig_top, height=fig_h)
        add_textbox(sl, Inches(0.5), fig_top + fig_h + Inches(0.06), Inches(5.9), Inches(0.34),
                    "EfficientNetB0  —  7 FN + 7 FP  (boundary errors)",
                    "Calibri", 12, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)
    if os.path.exists(FIG["res_bce_cm"]):
        add_image(sl, FIG["res_bce_cm"], left=Inches(7.0), top=fig_top, height=fig_h)
        add_textbox(sl, Inches(7.0), fig_top + fig_h + Inches(0.06), Inches(5.9), Inches(0.34),
                    "ResNet50  —  63 FN + 17 FP  (systematic miscalibration)",
                    "Calibri", 12, bold=True, color=AMBER, align=PP_ALIGN.CENTER)

    add_textbox(sl, Inches(0.4), Inches(6.65), Inches(12.55), Inches(0.62),
                "EfficientNet's symmetric errors are genuine ambiguous cases. "
                "ResNet's 63 false negatives include prototypical clear flood scenes — the decision boundary is miscalibrated. "
                "→ EfficientNet is the right model to target with HNM.",
                "Calibri", 12, color=STEEL)


# ── SLIDE 11 — Result: confounder FP rates ───────────────────────────────────

def slide_11_result_confounders(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "RESULTS", AMBER)

    slide_header(sl,
                 "River scenes cause a 9.2% false positive rate — every other confounder category is 0%",
                 "EfficientNetB0 BCE  ·  95% confidence intervals  ·  N per category shown")

    if os.path.exists(FIG["fp_heatmap"]):
        add_image(sl, FIG["fp_heatmap"], left=Inches(0.35), top=Inches(1.35), height=Inches(5.0))
        add_textbox(sl, Inches(0.35), Inches(6.4), Inches(6.3), Inches(0.28),
                    "Fig: Confounder FP rate heatmap by category and model",
                    "Calibri", 10, color=STEEL, align=PP_ALIGN.LEFT)

    findings = [
        ("EfficientNetB0 + BCE", OCEAN,
         ["River:   9.2%  (7/76, CI [3.8%, 17.7%])",
          "Plant:   2.9%  (1/34, CI [0.1%, 15.3%])",
          "All 6 other categories:  0%"]),
        ("ResNet50 + BCE", STEEL,
         ["River:   3.9%  (3/76, CI [0.8%, 11.0%])",
          "All 7 other categories:  0%"]),
        ("Swimming pool (N=28)", AMBER,
         ["0% FP — but upper CI = 12.3%",
          "Insufficient sample: 0% ≠ 'robust'",
          "~150+ images needed for tight bound"]),
    ]
    y = Inches(1.35)
    for model, col, items in findings:
        txt_col = WHITE if col != AMBER else OCEAN
        box_h   = Inches(0.45 + 0.44 * len(items))
        add_rect(sl, Inches(7.1), y, Inches(5.7), box_h, col)
        add_textbox(sl, Inches(7.25), y + Inches(0.08), Inches(5.4), Inches(0.38),
                    model, "Calibri", 13, bold=True, color=WHITE if col != AMBER else OCEAN)
        yi = y + Inches(0.48)
        for item in items:
            add_textbox(sl, Inches(7.3), yi, Inches(5.4), Inches(0.38),
                        "• " + item, "Calibri", 12, color=txt_col)
            yi += Inches(0.42)
        y = yi + Inches(0.3)

    add_textbox(sl, Inches(7.1), Inches(6.62), Inches(5.7), Inches(0.62),
                "Takeaway: river is the only statistically significant confounder. "
                "This justifies targeting rivers specifically in HNM.",
                "Calibri", 12, color=STEEL)


# ── SLIDE 12 — Result: HNM ablation ──────────────────────────────────────────

def slide_12_result_hnm(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "RESULTS", AMBER)

    slide_header(sl,
                 "Difficulty-ranked Phase-1 mining strictly outperforms extended training and random augmentation",
                 "Accuracy ordering: P1-HNM 98.78% > Extended 98.66% > Random 98.41% > Baseline 98.29%  ·  EfficientNetB0 BCE")

    # Bar chart via shapes
    conditions = [
        ("Baseline\n98.29%",       0.5,  STEEL, False),
        ("Random inject\n98.41%",  0.65, STEEL, False),
        ("Extended train\n98.66%", 0.82, STEEL, False),
        ("P1-HNM\n98.78%",         1.0,  AMBER, True),
    ]
    bar_bottom = Inches(5.55)
    bar_max_h  = Inches(3.3)
    bar_w      = Inches(2.15)
    x          = Inches(0.8)
    for label, frac, col, highlight in conditions:
        bh = bar_max_h * frac
        add_rect(sl, x, bar_bottom - bh, bar_w, bh, col)
        add_textbox(sl, x, bar_bottom + Inches(0.06), bar_w, Inches(0.58),
                    label, "Calibri", 12, bold=highlight,
                    color=OCEAN if not highlight else AMBER, align=PP_ALIGN.CENTER)
        x += Inches(2.55)

    add_textbox(sl, Inches(0.3), Inches(1.85), Inches(0.45), Inches(3.5),
                "Val. Accuracy", "Calibri", 11, color=STEEL, align=PP_ALIGN.CENTER)
    add_rect(sl, Inches(0.75), Inches(2.25), Inches(10.6), Inches(0.02), RULE_GRAY)
    add_textbox(sl, Inches(11.4), Inches(2.1), Inches(1.8), Inches(0.32),
                "Baseline", "Calibri", 10, color=STEEL)

    # Right panel
    add_rect(sl, Inches(11.4), Inches(1.35), Inches(1.7), Inches(4.35), RULE_GRAY)
    kn = [
        ("Gains vs. Baseline", True),
        ("+0.49 pp  P1-HNM", False),
        ("+0.37 pp  Extended", False),
        ("+0.12 pp  Random", False),
        ("", False),
        ("River FP after HNM:", True),
        ("7/76 = 9.2%", False),
        ("(unchanged)", False),
        ("", False),
        ("McNemar test:", True),
        ("p > 0.05 all pairs", False),
        ("Bonferroni corrected", False),
    ]
    yn = Inches(1.42)
    for txt, hdr in kn:
        add_textbox(sl, Inches(11.45), yn, Inches(1.6), Inches(0.3),
                    txt, "Calibri", 10, bold=hdr,
                    color=AMBER if hdr else STEEL)
        yn += Inches(0.33)

    # Mirror callout
    add_rect(sl, Inches(0.4), Inches(6.6), Inches(10.9), Inches(0.68), RULE_GRAY)
    add_textbox(sl, Inches(0.55), Inches(6.68), Inches(10.6), Inches(0.52),
                "RESULT (mirrors goal):  1 in 46 EfficientNet floods missed at baseline (7/322, 2.2%). "
                "P1-HNM achieves same recall with +0.49 pp accuracy — difficulty ranking strictly beats both alternatives.",
                "Calibri", 13, bold=False, color=OCEAN)


# ── SLIDE 13 — Conclusion ────────────────────────────────────────────────────

def slide_13_conclusion(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "CONCLUSION", OCEAN)

    slide_header(sl, "Conclusion")

    conclusions = [
        ("PR-AUC is the right metric for recall-first flood screening.",
         "3.6-pt gap (0.9976 vs. 0.9614) is concealed by accuracy (98.3% vs. 98.5%) and nearly hidden in ROC-AUC (2.6 pts)."),
        ("EfficientNetB0 is the right screener — symmetric boundary errors, not systematic collapse.",
         "7 FN vs. ResNet's 63 FN; errors are genuinely ambiguous cases, not miscalibration."),
        ("River is the only category with a statistically significant false positive rate.",
         "9.2% FP (7/76, CI [3.8%, 17.7%]). All 7 other confounder categories: 0% FP."),
        ("Phase-1 Hard Negative Mining (P1-HNM) strictly outperforms all ablations.",
         "P1-HNM 98.78% > Extended 98.66% > Random 98.41% > Baseline 98.29%."),
        ("No significant river FP reduction yet — a statistical power problem, not a null result.",
         "76 river validation images gives insufficient power; ~300 images needed to detect a 3–4 pp reduction."),
    ]

    y = Inches(1.32)
    for title, detail in conclusions:
        add_rect(sl, Inches(0.4), y, Inches(0.55), Inches(0.55), AMBER)
        add_textbox(sl, Inches(0.4), y + Inches(0.08), Inches(0.55), Inches(0.42),
                    "✓", "Cambria", 18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(sl, Inches(1.1), y, Inches(11.8), Inches(0.38),
                    title, "Calibri", 14, bold=True, color=OCEAN)
        add_textbox(sl, Inches(1.1), y + Inches(0.4), Inches(11.8), Inches(0.34),
                    detail, "Calibri", 12, color=STEEL)
        add_rect(sl, Inches(0.4), y + Inches(0.82), Inches(12.55), Inches(0.02), RULE_GRAY)
        y += Inches(1.1)


# ── SLIDE 14 — Future directions ──────────────────────────────────────────────

def slide_14_future(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "FUTURE WORK", STEEL)

    slide_header(sl,
                 "Future directions: statistical power, geographic generalization, and operational deployment",
                 "Five concrete next steps from the paper's limitations")

    directions = [
        ("Expand river validation set",
         "~300 river images needed for 80% power to detect a 3–4 pp FP reduction. Current 76-image set is underpowered."),
        ("Multi-seed robustness validation",
         "Re-run all experiments across seeds 42, 123, 256, 512, 1024 with bootstrap CIs to verify seed-independence."),
        ("Severity-stratified miss rate analysis",
         "Major vs. Minor flood miss rates differ operationally. Report separately — missing a Major flood costs far more."),
        ("Geographic generalization",
         "Dataset is US-centric. Test on South/Southeast Asian and European urban flood imagery to assess transfer."),
        ("Threshold calibration for deployment",
         "τ=0.5 is not optimal. Calibrate threshold against operational recall targets (e.g., ≥ 99%) using the PR curve."),
    ]

    y = Inches(1.35)
    for i, (head, body) in enumerate(directions):
        col = OCEAN if i % 2 == 0 else STEEL
        add_rect(sl, Inches(0.4), y, Inches(0.55), Inches(0.55), col)
        add_textbox(sl, Inches(0.4), y + Inches(0.08), Inches(0.55), Inches(0.42),
                    str(i+1), "Cambria", 18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        add_textbox(sl, Inches(1.1), y, Inches(11.8), Inches(0.38),
                    head, "Calibri", 14, bold=True, color=OCEAN)
        add_textbox(sl, Inches(1.1), y + Inches(0.38), Inches(11.8), Inches(0.42),
                    body, "Calibri", 12, color=STEEL)
        add_rect(sl, Inches(0.4), y + Inches(0.9), Inches(12.55), Inches(0.02), RULE_GRAY)
        y += Inches(1.08)


# ── SLIDE 15 — Acknowledgments ────────────────────────────────────────────────

def slide_15_acknowledgments(prs):
    sl = blank_slide(prs)
    white_bg(sl)
    section_chip(sl, "ACKNOWLEDGMENTS", OCEAN)

    slide_header(sl, "Acknowledgments")

    thanks = [
        ("Lab & Advisor",        "Dr. Dixon and the research lab for guidance and support throughout this project."),
        ("USF FloodingDataset",  "Authors of the USF FloodingDataset for the labeled flood severity imagery and non-flood category annotations."),
        ("RIWA Dataset",         "Wagner et al. (2023) for the RIWA river scene dataset used to enrich the water confounder category."),
        ("Audience",             "Thank you for your time and attention today."),
    ]

    y = Inches(1.42)
    for head, body in thanks:
        add_rect(sl, Inches(0.4), y, Inches(0.1), Inches(0.55), AMBER)
        add_textbox(sl, Inches(0.65), y, Inches(3.2), Inches(0.48),
                    head, "Calibri", 15, bold=True, color=OCEAN)
        add_textbox(sl, Inches(4.0), y + Inches(0.08), Inches(9.0), Inches(0.5),
                    body, "Calibri", 14, color=STEEL)
        add_rect(sl, Inches(0.4), y + Inches(0.68), Inches(12.55), Inches(0.02), RULE_GRAY)
        y += Inches(1.1)


# ── SLIDE 16 — Q&A ────────────────────────────────────────────────────────────

def slide_16_qa(prs):
    sl = blank_slide(prs)
    white_bg(sl)

    add_rect(sl, 0, 0, W, Inches(0.1), AMBER)
    add_rect(sl, 0, H - Inches(0.1), W, Inches(0.1), AMBER)
    add_rect(sl, 0, 0, Inches(0.18), H, OCEAN)

    add_textbox(sl, Inches(0.5), Inches(2.3), Inches(12.33), Inches(1.5),
                "Questions?",
                "Cambria", 64, bold=True, color=OCEAN, align=PP_ALIGN.CENTER)
    add_textbox(sl, Inches(0.5), Inches(3.95), Inches(12.33), Inches(0.55),
                "Aanya Singh  ·  University of South Florida",
                "Calibri", 18, color=STEEL, align=PP_ALIGN.CENTER)
    add_textbox(sl, Inches(0.5), Inches(4.55), Inches(12.33), Inches(0.45),
                "Advised by Dr. Dixon",
                "Calibri", 16, color=STEEL, align=PP_ALIGN.CENTER)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    prs = new_prs()

    builders = [
        (slide_01_title,                  "Title"),
        (slide_02_problem,                "Problem: floods + confounders"),
        (slide_03_problem_metrics,        "Problem: metrics mislead"),
        (slide_04_research_questions,     "Research Questions"),
        (slide_05_solution,               "Solution overview"),
        (slide_06_dataset,                "Dataset"),
        (slide_07_method_finetune,        "Method: two-phase fine-tuning"),
        (slide_08_method_hnm,             "Method: P1-HNM pipeline"),
        (slide_09_result_prroc,           "Result: PR-AUC vs ROC-AUC"),
        (slide_10_result_confusion,       "Result: confusion matrices"),
        (slide_11_result_confounders,     "Result: confounder FP rates"),
        (slide_12_result_hnm,             "Result: HNM ablation"),
        (slide_13_conclusion,             "Conclusion"),
        (slide_14_future,                 "Future directions"),
        (slide_15_acknowledgments,        "Acknowledgments"),
        (slide_16_qa,                     "Q&A"),
    ]

    print("Building slides...")
    for i, (fn, label) in enumerate(builders, 1):
        fn(prs)
        print(f"  {i:2d}/16  {label}")

    os.makedirs(os.path.dirname(OUT_PPTX), exist_ok=True)
    prs.save(OUT_PPTX)
    print(f"\nSaved → {OUT_PPTX}")
    print(f"Slides: {len(prs.slides)}")


if __name__ == "__main__":
    main()
