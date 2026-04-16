#!/usr/bin/env python3
"""
Build flood detection presentation (15 slides) for Computers & Geosciences audience.
Output: presentation/flood_detection.pptx
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = '/Users/aanyasingh/Projects/imagevalidation2'
FIGS = BASE + '/results/figures'
OUT  = BASE + '/presentation/flood_detection.pptx'

# ── Color palette ──────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x1A, 0x3A, 0x5C)   # Deep navy — dark bg, titles
ICE    = RGBColor(0xEE, 0xF5, 0xFB)   # Light ice blue — content bg
AMBER  = RGBColor(0xE0, 0x7B, 0x39)   # Amber-orange — emphasis
TEAL   = RGBColor(0x2E, 0x7D, 0x9C)   # Teal — section headers
CHAR   = RGBColor(0x2C, 0x2C, 0x2C)   # Dark charcoal — body text
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)   # White — text on dark slides
GRAY   = RGBColor(0x55, 0x55, 0x55)   # Caption gray
LGRAY  = RGBColor(0xBB, 0xCC, 0xDD)   # Light separator
RED    = RGBColor(0xC0, 0x30, 0x30)   # Red (bad metric)
GREEN  = RGBColor(0x1A, 0x7A, 0x3A)   # Green (good metric)
LNAVY  = RGBColor(0x9B, 0xB8, 0xD4)   # Light navy for subtitle on dark slides

# ── Presentation setup ─────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK_LAYOUT = prs.slide_layouts[6]


# ── Primitive helpers ──────────────────────────────────────────────────────────

def new_slide():
    return prs.slides.add_slide(BLANK_LAYOUT)


def rect(sl, l, t, w, h, fill, line_color=None):
    """Add a filled rectangle (inches)."""
    shp = sl.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(l), Inches(t), Inches(w), Inches(h)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line_color:
        shp.line.color.rgb = line_color
    else:
        shp.line.fill.background()
    return shp


def txt(sl, text, l, t, w, h, sz=16, bold=False, italic=False,
        color=CHAR, align=PP_ALIGN.LEFT, font='Calibri'):
    """Add a plain textbox (inches)."""
    tb = sl.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    lines = text.split('\n')
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
    return tb


def stxt(sl, text, l, t, w, h, fill, sz=15, bold=True, color=WHITE,
         font='Calibri', v_anchor='ctr'):
    """Colored shape with vertically-centered text."""
    shp = sl.shapes.add_shape(
        1, Inches(l), Inches(t), Inches(w), Inches(h)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.fill.background()
    tf = shp.text_frame
    tf.word_wrap = True
    bodyPr = tf._txBody.find(qn('a:bodyPr'))
    if bodyPr is not None:
        bodyPr.set('anchor', v_anchor)
    lines = text.split('\n')
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.name = font
        r.font.size = Pt(sz)
        r.font.bold = bold
        r.font.color.rgb = color
    return shp


def img(sl, path, l, t, w, h=None):
    """Add a picture; prints warning if file missing."""
    if not os.path.exists(path):
        print(f'  MISSING: {path}')
        return None
    if h:
        return sl.shapes.add_picture(path, Inches(l), Inches(t),
                                     Inches(w), Inches(h))
    return sl.shapes.add_picture(path, Inches(l), Inches(t), Inches(w))


def set_notes(sl, text):
    sl.notes_slide.notes_text_frame.text = text


# ── Composite helpers ──────────────────────────────────────────────────────────

def dark_bg(sl):
    """Full-slide deep navy background with amber accent bar at bottom."""
    rect(sl, 0, 0, 13.33, 7.5, NAVY)
    rect(sl, 0, 7.1, 13.33, 0.07, AMBER)


def light_bg(sl):
    """Full-slide ice-blue background."""
    rect(sl, 0, 0, 13.33, 7.5, ICE)


def slide_header(sl, title, subtitle=None):
    """Navy header bar with teal title text (for light slides)."""
    rect(sl, 0, 0, 13.33, 1.05, NAVY)
    rect(sl, 0, 1.05, 13.33, 0.05, AMBER)
    txt(sl, title, 0.4, 0.08, 12.5, 0.88,
        sz=26, bold=True, color=WHITE, font='Cambria', align=PP_ALIGN.LEFT)
    if subtitle:
        txt(sl, subtitle, 0.4, 0.78, 12.5, 0.35,
            sz=13, color=LNAVY, font='Calibri')


def takeaway_box(sl, text, l=0.4, t=6.65, w=12.5, h=0.6):
    """Teal takeaway strip at bottom of slide."""
    rect(sl, l, t, w, h, TEAL)
    txt(sl, text, l + 0.15, t + 0.08, w - 0.3, h - 0.15,
        sz=13, bold=True, color=WHITE, font='Calibri',
        align=PP_ALIGN.LEFT)


def caption(sl, text, l, t, w, h=0.3):
    txt(sl, text, l, t, w, h, sz=11, italic=True, color=GRAY,
        align=PP_ALIGN.CENTER)


def slide_number(sl, n):
    txt(sl, str(n), 12.6, 7.15, 0.5, 0.3, sz=11, color=GRAY,
        align=PP_ALIGN.RIGHT)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title (dark)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 1: Title')
s1 = new_slide()
dark_bg(s1)

txt(s1,
    'Confounder-Robust Flood Detection\nfrom Street-Level Imagery\nvia Phase-1 Hard Negative Mining',
    0.7, 0.9, 11.9, 3.0,
    sz=38, bold=True, color=WHITE, font='Cambria', align=PP_ALIGN.CENTER)

rect(s1, 3.2, 4.05, 6.9, 0.06, AMBER)

txt(s1,
    'Computers & Geosciences  ·  2026',
    0.7, 4.3, 11.9, 0.55,
    sz=20, color=LNAVY, font='Calibri', align=PP_ALIGN.CENTER)

txt(s1,
    'Aanya Singh  ·  University of South Florida',
    0.7, 5.0, 11.9, 0.5,
    sz=16, color=LNAVY, font='Calibri', align=PP_ALIGN.CENTER)

txt(s1,
    'Abbreviations: PR-AUC = Precision-Recall Area Under Curve  ·  ROC-AUC = Receiver Operating Characteristic AUC\n'
    'HNM = Hard Negative Mining  ·  P1-HNM = Phase-1 Hard Negative Mining  ·  FP = False Positive  ·  FN = False Negative  ·  BCE = Binary Cross-Entropy',
    0.7, 5.8, 11.9, 0.9,
    sz=11, color=GRAY, font='Calibri', align=PP_ALIGN.CENTER)

set_notes(s1,
    "Welcome. This talk is about a specific and practical problem: automated screening "
    "of street-level images for flood detection. We built a two-phase fine-tuned classifier "
    "and found two problems that had to be fixed before the system was fit for purpose. "
    "First, the standard evaluation metrics — accuracy and ROC-AUC — actively misled us. "
    "Second, visual confounders (rivers, retention ponds) caused most of the false alarms. "
    "This talk covers both problems and the Phase-1 Hard Negative Mining approach we used to address them.")

slide_number(s1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Floods demand fast, reliable automated screening (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 2: Motivation')
s2 = new_slide()
light_bg(s2)
slide_header(s2, 'Floods demand fast, reliable automated screening')

# Big stat callout
stxt(s2, '~10,000\ndeaths/year', 0.5, 1.35, 4.1, 2.0, NAVY, sz=28, bold=True)
txt(s2, 'from floods globally\n(EM-DAT, CRED)',
    0.5, 3.45, 4.1, 0.6, sz=13, color=GRAY, align=PP_ALIGN.CENTER)

# Three motivation points
bullets = [
    ('Citizen science platforms receive thousands of images after flood events',
     'Human review of every submission does not scale at disaster speed.'),
    ('Street-level imagery is now ubiquitous',
     'Smartphones + dashcams provide ground-truth that satellite imagery misses under clouds.'),
    ('First-pass screening enables triage',
     'Route flood images to risk models; route non-floods to archive. Saves analyst time.'),
]
for i, (head, body) in enumerate(bullets):
    y = 1.35 + i * 1.45
    rect(s2, 4.9, y, 8.0, 1.25, WHITE)
    rect(s2, 4.9, y, 0.08, 1.25, TEAL)
    txt(s2, head, 5.1, y + 0.08, 7.6, 0.45,
        sz=15, bold=True, color=NAVY, font='Calibri')
    txt(s2, body, 5.1, y + 0.52, 7.6, 0.6,
        sz=13, color=CHAR, font='Calibri')

takeaway_box(s2,
    'A reliable screener must maximise recall — a missed flood means a neighbourhood goes unreported.')

set_notes(s2,
    "Floods kill roughly 10,000 people per year globally and cause hundreds of billions in "
    "economic damage. After a flood event, citizen science platforms receive thousands of "
    "street-level photos from community members. A first-pass screener routes flood images "
    "to risk models and human review, and archives the rest. But the screener must be "
    "recall-first: a missed flood means a neighborhood never gets flagged. A false alarm "
    "costs a reviewer one click. These are not equivalent failure modes.")

slide_number(s2, 2)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Visual confounders are the dominant failure mode (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 3: Confounders')
s3 = new_slide()
light_bg(s3)
slide_header(s3, 'Visual confounders are the dominant failure mode',
             subtitle='EfficientNetB0 baseline — false positives and false negatives')

img(s3, FIGS + '/baselines/efficientnet_bce/fp_fn_examples.png',
    0.4, 1.2, 12.5, 5.2)

takeaway_box(s3,
    'All 7 FPs were river scenes. At close range, reflective water texture is visually identical to a flooded road.')

set_notes(s3,
    "This is the key failure mode. EfficientNet's 7 false positives in the baseline were all "
    "river photographs — taken from bridges or embankments where the reflective surface, ripple "
    "pattern, and cropped horizon are visually indistinguishable from a flooded street. "
    "The model isn't broken. It's responding to the correct visual features, just in the wrong context. "
    "Other potential confounders — swimming pools, wet roads, park fountains — produced zero false "
    "positives, though we'll see why 'zero' needs careful interpretation.")

slide_number(s3, 3)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Standard metrics overstate performance (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 4: Metrics')
s4 = new_slide()
light_bg(s4)
slide_header(s4, 'Standard metrics overstate performance on imbalanced data')

# Two metric formula boxes
for label, formula, note, x, bc, fc in [
    ('Accuracy', 'Acc = (TP + TN) / N',
     'Dominated by the large non-flood majority.\nA model that labels everything non-flood scores ~84%.',
     0.4, WHITE, CHAR),
    ('PR-AUC', 'Precision × Recall trade-off\nintegrated over all thresholds',
     'Penalises missed floods (recall axis)\nand false alarms (precision axis).',
     6.9, TEAL, WHITE),
]:
    rect(s4, x, 1.35, 5.9, 3.2, bc, line_color=TEAL)
    txt(s4, label, x + 0.2, 1.45, 5.5, 0.5,
        sz=18, bold=True, color=TEAL if bc == WHITE else WHITE, font='Cambria')
    txt(s4, formula, x + 0.2, 2.05, 5.5, 0.85,
        sz=15, bold=True, color=NAVY if bc == WHITE else WHITE, font='Calibri',
        align=PP_ALIGN.CENTER)
    rect(s4, x + 0.2, 2.95, 5.5, 0.04, AMBER)
    txt(s4, note, x + 0.2, 3.1, 5.5, 1.2,
        sz=13, color=CHAR if bc == WHITE else WHITE, font='Calibri')

txt(s4, 'vs.', 6.2, 2.55, 0.6, 0.55,
    sz=20, bold=True, color=AMBER, align=PP_ALIGN.CENTER)

# ROC-AUC note
rect(s4, 0.4, 4.85, 12.5, 1.2, WHITE, line_color=LGRAY)
txt(s4,
    'ROC-AUC also misleads: it averages performance over all thresholds including low-FP-rate regions '
    'that rarely matter in practice. A 2.6-pt ROC-AUC gap between EfficientNet and ResNet50 '
    'hid 56 missed flood reports that a 3.6-pt PR-AUC gap correctly flagged.',
    0.6, 4.95, 12.1, 1.0,
    sz=13, color=CHAR, font='Calibri')

takeaway_box(s4,
    'PR-AUC is the appropriate primary metric for a recall-first flood screener.')

set_notes(s4,
    "Accuracy is the default metric for classification, but it's wrong here. "
    "With roughly 84% non-flood images in validation, a model that labels everything non-flood "
    "achieves 84% accuracy. The two-class imbalance means accuracy is dominated by the majority class. "
    "ROC-AUC integrates over all operating thresholds, including very low false positive rate regions "
    "that don't reflect real deployment. PR-AUC integrates precision against recall — both axes "
    "directly relevant to flood detection. The 3.6-point PR-AUC gap between EfficientNet and ResNet50 "
    "corresponds to 56 missed floods in a single 819-image validation set. That's the metric we use.")

slide_number(s4, 4)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Goal (dark)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 5: Goal')
s5 = new_slide()
dark_bg(s5)

txt(s5, 'Research Question', 0.7, 0.85, 11.9, 0.6,
    sz=20, bold=True, color=AMBER, font='Cambria', align=PP_ALIGN.CENTER)

stxt(s5,
    'Can targeted hard negative mining from an\nintermediate training checkpoint reduce\nfalse alarms from visual confounders\n(rivers, pools) without sacrificing flood recall?',
    1.2, 1.6, 10.9, 3.2,
    NAVY, sz=28, bold=True, color=WHITE, font='Cambria')

# Three sub-questions
for i, q in enumerate([
    'Which architecture and loss function best preserves recall?',
    'Which evaluation metric correctly ranks models for this task?',
    'Does difficulty-ranked mining outperform extended training and random injection?',
]):
    rect(s5, 1.2, 5.1 + i * 0.62, 10.9, 0.52, RGBColor(0x22, 0x4E, 0x72))
    txt(s5, f'{i+1}.  {q}', 1.45, 5.15 + i * 0.62, 10.4, 0.42,
        sz=14, color=LNAVY, font='Calibri')

set_notes(s5,
    "Three sub-questions structure the paper. First, architecture and loss function: "
    "EfficientNetB0 vs ResNet50, BCE vs Focal loss — which combination preserves recall on "
    "imbalanced flood data? Second, evaluation metrics: which metric correctly ranks models "
    "for a recall-first screener? Third, the core HNM question: does mining from the Phase-1 "
    "checkpoint outperform just training longer or adding random data from the same categories? "
    "The ablation controls are key — they isolate what's actually driving the improvement.")

slide_number(s5, 5)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Dataset (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 6: Dataset')
s6 = new_slide()
light_bg(s6)
slide_header(s6, 'Dataset: 4,099 street-level images across 11 fine-grained categories')

# Dataset stats
stats = [
    ('Total images', '4,099'),
    ('Flood (positive)', '1,657  (40.4%)'),
    ('Non-flood (negative)', '2,442  (59.6%)'),
    ('Train / Val / Test', '2,870 / 819 / 410'),
    ('Image source', 'CRIS-HAZARD + open datasets'),
]
for i, (label, val) in enumerate(stats):
    y = 1.35 + i * 0.72
    rect(s6, 0.4, y, 3.8, 0.62, WHITE if i % 2 == 0 else ICE)
    txt(s6, label, 0.6, y + 0.1, 3.4, 0.42, sz=13, bold=True, color=NAVY)
    rect(s6, 4.2, y, 4.2, 0.62, WHITE if i % 2 == 0 else ICE)
    txt(s6, val,   4.4, y + 0.1, 3.8, 0.42, sz=13, color=CHAR)

# Category breakdown table
cats = [
    ('Flood categories (7)',    ['Urban flood', 'Rural flood', 'Highway flood',
                                  'Building flood', 'Car flood', 'Infrastructure', 'Debris']),
    ('Water confounders (2)',   ['River / stream', 'Swimming pool / pond']),
    ('Street confounders (2)',  ['Wet road', 'Street (dry)']),
]
y0 = 1.35
for col, (group_label, items) in enumerate(cats):
    x = 8.7 + col * 0.0   # stacked vertically in right column
    if col == 0:
        x, y0 = 8.7, 1.35
    elif col == 1:
        x, y0 = 8.7, 3.75
    else:
        x, y0 = 8.7, 5.35
    rect(s6, x, y0, 4.2, 0.4, TEAL)
    txt(s6, group_label, x + 0.12, y0 + 0.07, 3.9, 0.3,
        sz=12, bold=True, color=WHITE)
    for j, item in enumerate(items):
        rect(s6, x, y0 + 0.4 + j * 0.38, 4.2, 0.37,
             WHITE if j % 2 == 0 else ICE)
        txt(s6, '•  ' + item, x + 0.2, y0 + 0.45 + j * 0.38, 3.8, 0.28,
            sz=12, color=CHAR)

takeaway_box(s6,
    'Two water-confounder categories (river, pool) are the primary evaluation targets for false positive analysis.')

set_notes(s6,
    "The dataset has 4,099 street-level images. Roughly 40% are flood-positive, which is more "
    "balanced than a real deployment scenario but reflects intentional curation for training. "
    "The 11 fine-grained categories span 7 flood subtypes and 4 non-flood subtypes. "
    "The two water confounders — rivers/streams and swimming pools/ponds — are the primary "
    "concern for false positives because they share visual features with flood scenes. "
    "The two street confounders (wet road, dry street) are included as controls.")

slide_number(s6, 6)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Two-phase training + training curves (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 7: Training curves')
s7 = new_slide()
light_bg(s7)
slide_header(s7, 'Two-phase progressive fine-tuning preserves mining signal at Phase-1')

# Phase description boxes
for label, detail, x in [
    ('Phase 1', 'Top 30 layers unfrozen\nLR = 1×10⁻⁴  ·  ≤15 epochs\nClassification head + upper backbone', 0.4),
    ('Phase 2', 'All layers unfrozen\nLR = 1×10⁻⁵  ·  ≤15 epochs\nFull fine-tuning to convergence', 3.55),
    ('P1-HNM', 'Mine top-10% FP candidates from M₁\n5× augmentation per candidate\nRetrain from Phase-2 checkpoint', 6.7),
]:
    rect(s7, x, 1.2, 2.85, 1.55, WHITE if label != 'P1-HNM' else RGBColor(0xFF, 0xF3, 0xE6))
    rect(s7, x, 1.2, 2.85, 0.38, TEAL if label != 'P1-HNM' else AMBER)
    txt(s7, label, x + 0.12, 1.25, 2.6, 0.3,
        sz=15, bold=True, color=WHITE, font='Cambria')
    txt(s7, detail, x + 0.12, 1.66, 2.6, 1.0, sz=12, color=CHAR)

txt(s7, '→', 3.3, 1.75, 0.4, 0.45, sz=22, bold=True, color=AMBER,
    align=PP_ALIGN.CENTER)
txt(s7, '→', 6.45, 1.75, 0.4, 0.45, sz=22, bold=True, color=AMBER,
    align=PP_ALIGN.CENTER)

# Training curves figure
img(s7, FIGS + '/baselines/efficientnet_bce/training_curves.png',
    0.4, 2.9, 9.55, 3.9)

# Key insight box
rect(s7, 10.1, 2.9, 2.85, 3.9, WHITE, line_color=TEAL)
txt(s7,
    'Key insight\n\nBy Phase-2 convergence, all non-flood images have near-zero flood probability.\n\nPhase-1 retains genuine uncertainty on river/pool images.\n\nThat uncertainty is the mining signal.',
    10.25, 3.05, 2.55, 3.6, sz=12, color=NAVY, font='Calibri')

takeaway_box(s7,
    'Mine hard negatives from Phase-1 checkpoint (M₁), not the converged model — the signal disappears at convergence.')

set_notes(s7,
    "The two-phase training strategy is standard progressive fine-tuning. In Phase 1 we unfreeze "
    "the top 30 layers and the classification head, training at learning rate 1e-4. The model "
    "improves quickly but retains uncertainty about visually ambiguous categories. In Phase 2 "
    "we unfreeze all layers and reduce the learning rate to 1e-5, driving the model to convergence. "
    "The key insight: by the end of Phase 2, every non-flood training image has been assigned "
    "near-zero flood probability. There are no candidates to mine. Phase 1 is the right moment "
    "to collect hard negatives — the model has learned enough to identify difficult examples "
    "but hasn't yet resolved its uncertainty about rivers and pools.")

slide_number(s7, 7)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — EfficientNet vs ResNet confusion matrices (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 8: Confusion matrices')
s8 = new_slide()
light_bg(s8)
slide_header(s8, 'EfficientNetB0 detects 9× more floods than ResNet50 at equal accuracy')

# Stat callouts
for label, val, note, x, vc in [
    ('EfficientNetB0\nAccuracy', '98.3%', '7 FN, 7 FP', 0.4, CHAR),
    ('ResNet50\nAccuracy',       '98.5%', '63 FN, 17 FP', 4.65, RED),
    ('Flood recall gap',         '17.4 pp', 'EfficientNet: 97.8%\nResNet50: 80.4%', 8.9, TEAL),
]:
    rect(s8, x, 1.2, 3.9, 1.4, WHITE, line_color=LGRAY)
    txt(s8, val, x + 0.1, 1.28, 3.7, 0.72,
        sz=40, bold=True, color=vc, font='Cambria', align=PP_ALIGN.CENTER)
    txt(s8, label, x + 0.1, 1.92, 3.7, 0.4,
        sz=12, bold=True, color=GRAY, align=PP_ALIGN.CENTER)
    txt(s8, note, x + 0.1, 2.3, 3.7, 0.25,
        sz=11, color=GRAY, align=PP_ALIGN.CENTER)

# Confusion matrices side by side
img(s8, FIGS + '/baselines/efficientnet_bce/confusion_matrix.png',
    0.6, 2.75, 5.8, 3.85)
img(s8, FIGS + '/baselines/resnet50_bce/confusion_matrix.png',
    7.0, 2.75, 5.8, 3.85)

caption(s8, 'EfficientNetB0 (BCE)', 0.6, 6.62, 5.8)
caption(s8, 'ResNet50 (BCE)', 7.0, 6.62, 5.8)

takeaway_box(s8,
    'Same accuracy, 9× more missed floods. Accuracy actively inverted the model ranking. Use EfficientNetB0.')

set_notes(s8,
    "Both models achieve roughly 98% accuracy. But EfficientNet has 7 false negatives — 7 missed "
    "floods — while ResNet50 has 63. That's the same accuracy masking a 17.4 percentage-point "
    "gap in flood recall: 97.8% versus 80.4%. The large non-flood majority dominates accuracy's "
    "denominator, making ResNet50 appear slightly better (98.5% vs 98.3%) despite missing nearly "
    "9 times as many floods. This is why accuracy is the wrong metric — it actively inverted "
    "the ranking. For the ablation experiments, we use EfficientNetB0 with BCE loss throughout.")

slide_number(s8, 8)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — PR-AUC exposes 56 missed floods (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 9: PR/ROC curves')
s9 = new_slide()
light_bg(s9)
slide_header(s9, 'PR-AUC exposes 56 missed floods that accuracy and ROC-AUC hide')

# PR-ROC curves side by side
img(s9, FIGS + '/baselines/efficientnet_bce/pr_roc_curves.png',
    0.4, 1.25, 6.2, 5.1)
img(s9, FIGS + '/baselines/resnet50_bce/pr_roc_curves.png',
    6.75, 1.25, 6.2, 5.1)

caption(s9, 'EfficientNetB0 (BCE)  ·  PR-AUC = 0.9976  ·  ROC-AUC = 0.9985',
        0.4, 6.35, 6.2)
caption(s9, 'ResNet50 (BCE)  ·  PR-AUC = 0.9614  ·  ROC-AUC = 0.9728',
        6.75, 6.35, 6.2)

takeaway_box(s9,
    'PR-AUC gap: 3.6 pts = 56 more missed floods. ROC-AUC gap: 2.6 pts. PR-AUC is the right metric.')

set_notes(s9,
    "Look at the two panels. Left is EfficientNet, right is ResNet50. The ROC curves look "
    "reasonably similar — both hug the top-left corner. The ROC-AUC gap is 2.6 points. "
    "The PR curves tell a different story: EfficientNet's precision-recall curve stays near "
    "1.0 across most of the recall range. ResNet50's drops steeply. The PR-AUC gap is 3.6 points. "
    "That 3.6-point difference corresponds to 56 missed flood reports in a single 819-image "
    "validation set. The PR-AUC correctly identified EfficientNet as the better model for "
    "this task. ROC-AUC gave the right answer here too, but the gap looks smaller and the "
    "intuitive connection to missed floods is harder to make. PR-AUC should be the standard "
    "metric for any flood screening evaluation.")

slide_number(s9, 9)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — River is the only confounder (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 10: Confounder heatmap')
s10 = new_slide()
light_bg(s10)
slide_header(s10, 'River is the only confounder that produces false alarms')

img(s10, FIGS + '/confounder_fp_heatmap.png',
    0.4, 1.2, 9.0, 5.25)

# CI callout box on right
rect(s10, 9.55, 1.2, 3.4, 2.7, WHITE, line_color=TEAL)
txt(s10, 'River FP rate\n(EfficientNet BCE)', 9.7, 1.3, 3.1, 0.6,
    sz=13, bold=True, color=TEAL, font='Cambria')
txt(s10, '9.2%\n[3.8%, 17.7%]', 9.7, 1.95, 3.1, 1.2,
    sz=22, bold=True, color=AMBER, font='Cambria', align=PP_ALIGN.CENTER)
txt(s10, 'Clopper-Pearson 95% CI\n7 FP / 76 river images', 9.7, 3.15, 3.1, 0.6,
    sz=11, color=GRAY)

rect(s10, 9.55, 4.1, 3.4, 2.35, WHITE, line_color=LGRAY)
txt(s10, 'All other categories\n(both models)', 9.7, 4.2, 3.1, 0.55,
    sz=13, bold=True, color=CHAR, font='Cambria')
txt(s10, '0.0%', 9.7, 4.8, 3.1, 0.75,
    sz=30, bold=True, color=GREEN, align=PP_ALIGN.CENTER)
txt(s10,
    '"Zero" on pools (N=28):\n95% CI upper bound = 12.3%\nNot confirmed robust.',
    9.7, 5.6, 3.1, 0.75, sz=11, color=RED)

takeaway_box(s10,
    'River: 9.2% FP [3.8%, 17.7%]. Pool "zero" at N=28 is uninformative — upper CI bound = 12.3%.')

set_notes(s10,
    "The heatmap shows false positive rates by category and model condition. River scenes "
    "are the only category producing false positives at the 0.5 decision threshold. "
    "EfficientNet BCE: 9.2% river FP rate, with a Clopper-Pearson 95% CI of [3.8%, 17.7%]. "
    "That CI is wide because we have 76 river images — 7 false positives. "
    "All other categories show zero false positives. But 'zero' on swimming pools, with "
    "only 28 images, gives a 95% CI upper bound of 12.3%. We cannot conclude the model "
    "is robust to pools — we lack statistical power. Before deployment, expanding "
    "water-confounder validation sets is a data collection priority, not a modeling priority.")

slide_number(s10, 10)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — HNM ablation results (light, 4-panel)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 11: Ablation / HNM results')
s11 = new_slide()
light_bg(s11)
slide_header(s11, 'HNM outperforms extended training and random injection')

# 4-panel 2×2 grid of PR-ROC curves
panel_specs = [
    ('Baseline (EfficientNet BCE)',
     FIGS + '/baselines/efficientnet_bce/pr_roc_curves.png',
     0.4, 1.25),
    ('Extended training (+epochs)',
     FIGS + '/ablations/extended_baseline/efficientnet_bce/pr_roc_curves.png',
     6.7, 1.25),
    ('Random injection (same N)',
     FIGS + '/ablations/random_inject/efficientnet_bce/pr_roc_curves.png',
     0.4, 4.15),
    ('P1-HNM ★ (difficulty-ranked)',
     FIGS + '/hnm/efficientnet_bce/pr_roc_curves.png',
     6.7, 4.15),
]
for label, path, x, y in panel_specs:
    img(s11, path, x, y, 5.9, 2.7)
    is_hnm = '★' in label
    caption(s11, label, x, y + 2.72, 5.9)
    if is_hnm:
        rect(s11, x, y, 5.9, 0.28, AMBER)
        txt(s11, label, x + 0.1, y + 0.03, 5.7, 0.22,
            sz=11, bold=True, color=WHITE)

# Ordering summary strip
rect(s11, 0.4, 7.0, 12.5, 0.35, WHITE, line_color=LGRAY)
txt(s11,
    'Accuracy gain:  Baseline → Random (+0.12 pp) → Extended (+0.37 pp) → P1-HNM (+0.49 pp)  '
    '|  Each step isolates one variable — data volume, category exposure, difficulty ranking',
    0.55, 7.05, 12.2, 0.27, sz=11, bold=False, color=CHAR)

takeaway_box(s11,
    'Strict ordering holds: P1-HNM (+0.49 pp) > Extended (+0.37 pp) > Random (+0.12 pp) > Baseline. Difficulty ranking matters.')

set_notes(s11,
    "The four panels show the PR-ROC curves for baseline, extended training, random injection, "
    "and P1-HNM. The strict ordering in accuracy improvement: baseline to random injection "
    "adds 0.12 percentage points — some benefit from exposure to the confounder categories. "
    "Extended training adds 0.37 pp — more compute helps beyond more data. "
    "P1-HNM adds 0.49 pp — difficulty ranking adds something beyond category exposure and "
    "training budget alone. Each comparison isolates one variable. "
    "Random injection versus extended: is it the data or the epochs? Both get the same epoch budget. "
    "P1-HNM versus random: same count, same categories, same budget — only the difficulty "
    "ranking differs. The ordering is consistent with the hypothesis that Phase-1 uncertainty "
    "identifies genuinely harder examples.")

slide_number(s11, 11)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Conclusion (dark)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 12: Conclusion')
s12 = new_slide()
dark_bg(s12)

txt(s12, 'Conclusions', 0.7, 0.3, 11.9, 0.75,
    sz=32, bold=True, color=WHITE, font='Cambria', align=PP_ALIGN.CENTER)
rect(s12, 3.2, 1.05, 6.9, 0.06, AMBER)

conclusions = [
    ('1', 'Use EfficientNetB0',
     '97.8% flood recall vs 80.4% for ResNet50 — same accuracy masks 9× more missed floods.'),
    ('2', 'Use PR-AUC as primary metric',
     '3.6-pt gap = 56 missed floods in one validation set. Accuracy and ROC-AUC actively mislead.'),
    ('3', 'Phase-1 HNM reduces confounder FPs',
     'Strict ordering: P1-HNM (+0.49 pp) > Extended (+0.37 pp) > Random (+0.12 pp) > Baseline.'),
    ('4', 'River remains the hard problem; pool data is too sparse to conclude',
     'River: 9.2% FP [3.8, 17.7]. Pool "zero" at N=28 → CI upper bound 12.3%.'),
]
for i, (num, head, body) in enumerate(conclusions):
    y = 1.35 + i * 1.4
    stxt(s12, num, 0.5, y, 0.65, 1.1, AMBER, sz=22, bold=True)
    rect(s12, 1.3, y, 11.5, 1.1, RGBColor(0x22, 0x4E, 0x72))
    txt(s12, head, 1.5, y + 0.1, 11.1, 0.42,
        sz=16, bold=True, color=WHITE, font='Cambria')
    txt(s12, body, 1.5, y + 0.58, 11.1, 0.45,
        sz=13, color=LNAVY, font='Calibri')

set_notes(s12,
    "Four conclusions. One: use EfficientNetB0. The recall gap versus ResNet50 is 17 percentage "
    "points; accuracy hid this. Two: use PR-AUC. It correctly ranks models for recall-first "
    "screening and quantifies the gap in terms of missed floods. Three: Phase-1 HNM works — "
    "it strictly outperforms extended training and random injection. Four: river remains the "
    "dominant confounder; pool robustness cannot be confirmed at N=28. These findings have "
    "direct implications for how flood screening systems should be trained and evaluated.")

slide_number(s12, 12)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Future directions (light)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 13: Future directions')
s13 = new_slide()
light_bg(s13)
slide_header(s13, 'Outstanding questions & future directions')

future = [
    ('Expand water-confounder validation',
     'Target N ≥ 300 river images; expand pool/lake sets to N ≥ 100.\n'
     'Current N=76 river images give wide CIs; deployment decisions cannot be made at N=28 pools.'),
    ('Per-category confidence intervals as deployment gates',
     'Flag any category with FP rate upper CI > 5% as requiring additional data before deployment.\n'
     'Clopper-Pearson CIs should be standard reporting for all category-level evaluations.'),
    ('Iterative HNM across training phases',
     'P1-HNM was applied once. Iterative mining at multiple checkpoints may compound gains.\n'
     'Investigate optimal mining fraction (current: 10%) and augmentation factor (current: 5×).'),
    ('Temporal and geographic generalisation',
     'Dataset is Pinellas County, FL. Evaluate on post-Helene imagery and other coastal geographies.\n'
     'Confounder distribution will differ — rivers vs estuaries vs retention ponds.'),
    ('Threshold calibration per deployment context',
     'Τ=0.5 default threshold. For high-stakes recall-first scenarios, calibrate threshold\n'
     'on held-out deployment data using PR curve operating point selection.'),
]
for i, (head, body) in enumerate(future):
    y = 1.25 + i * 1.18
    rect(s13, 0.4, y, 12.5, 1.08, WHITE if i % 2 == 0 else ICE)
    rect(s13, 0.4, y, 0.1, 1.08, TEAL)
    txt(s13, head, 0.65, y + 0.08, 11.9, 0.38,
        sz=14, bold=True, color=NAVY, font='Calibri')
    txt(s13, body, 0.65, y + 0.5, 11.9, 0.52,
        sz=12, color=CHAR, font='Calibri')

set_notes(s13,
    "Five directions. First and most urgent: expand validation data for water confounders. "
    "This is a data problem, not a modeling problem. Second: make per-category CIs a standard "
    "part of evaluation reporting — flag categories where the upper bound exceeds 5%. "
    "Third: investigate iterative HNM across multiple checkpoints. We only applied it once. "
    "Fourth: test geographic generalisation — Pinellas County rivers look different from "
    "Pacific Northwest rivers or UK canals. Fifth: calibrate the decision threshold. "
    "We used 0.5; for a recall-first deployment you'd choose the threshold on the PR curve.")

slide_number(s13, 13)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Acknowledgments (dark)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 14: Acknowledgments')
s14 = new_slide()
dark_bg(s14)

txt(s14, 'Acknowledgments', 0.7, 0.5, 11.9, 0.75,
    sz=32, bold=True, color=WHITE, font='Cambria', align=PP_ALIGN.CENTER)
rect(s14, 3.2, 1.25, 6.9, 0.06, AMBER)

ack_items = [
    'CRIS-HAZARD platform team for the curated street-level imagery dataset',
    'University of South Florida, School of Geosciences',
    'Open-source contributors: PyTorch, HuggingFace Datasets, python-pptx',
    'Reviewers at Computers & Geosciences for constructive feedback',
]
for i, item in enumerate(ack_items):
    y = 1.55 + i * 0.85
    rect(s14, 0.8, y, 11.7, 0.65, RGBColor(0x22, 0x4E, 0x72))
    txt(s14, '•  ' + item, 1.1, y + 0.12, 11.1, 0.42,
        sz=15, color=LNAVY, font='Calibri')

txt(s14,
    'Funding: [Grant agency / award number]',
    0.7, 5.2, 11.9, 0.45, sz=14, color=GRAY, align=PP_ALIGN.CENTER)

txt(s14,
    'Data and code: github.com/[username]/flood-detection-hnm',
    0.7, 5.75, 11.9, 0.45, sz=14, color=LNAVY, align=PP_ALIGN.CENTER,
    italic=True)

set_notes(s14, "Acknowledge collaborators and funding as appropriate.")

slide_number(s14, 14)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Q&A (dark)
# ══════════════════════════════════════════════════════════════════════════════
print('Slide 15: Q&A')
s15 = new_slide()
dark_bg(s15)

txt(s15, 'Q & A', 0.7, 1.5, 11.9, 1.5,
    sz=72, bold=True, color=WHITE, font='Cambria', align=PP_ALIGN.CENTER)

rect(s15, 2.5, 3.3, 8.3, 0.07, AMBER)

txt(s15,
    'Key numbers to remember',
    0.7, 3.6, 11.9, 0.5, sz=16, bold=True, color=AMBER,
    align=PP_ALIGN.CENTER, font='Cambria')

quick_refs = [
    ('EfficientNet recall', '97.8%  vs  ResNet50 80.4%'),
    ('PR-AUC gap', '3.6 pp  =  56 missed floods'),
    ('River FP rate', '9.2%  [3.8%, 17.7%]  (N=76)'),
    ('HNM accuracy gain', '+0.49 pp > Extended +0.37 pp > Random +0.12 pp'),
]
for i, (label, val) in enumerate(quick_refs):
    x = 0.7 + (i % 2) * 6.2
    y = 4.2 + (i // 2) * 0.85
    txt(s15, label + ':  ', x, y, 3.0, 0.6, sz=13, bold=True, color=GRAY,
        align=PP_ALIGN.RIGHT)
    txt(s15, val, x + 3.0, y, 2.9, 0.6, sz=13, bold=True, color=WHITE)

set_notes(s15,
    "Quick reference numbers during Q&A:\n"
    "- EfficientNet flood recall 97.8% vs ResNet50 80.4% (both ~98.5% accuracy)\n"
    "- PR-AUC gap 3.6 pts = 56 missed floods in 819-image validation set\n"
    "- River FP: 9.2% [3.8%, 17.7%] Clopper-Pearson 95% CI, N=76\n"
    "- Accuracy gains: P1-HNM +0.49 pp, Extended +0.37 pp, Random +0.12 pp\n"
    "- Pool 'zero' at N=28 → upper CI bound 12.3% — not robust\n"
    "- Phase-1 mining fraction: top 10% by flood probability, 5× augmentation each")

slide_number(s15, 15)


# ── Save ────────────────────────────────────────────────────────────────────────
print(f'\nSaving to {OUT} ...')
os.makedirs(os.path.dirname(OUT), exist_ok=True)
prs.save(OUT)
print(f'Done. {len(prs.slides)} slides saved.')
