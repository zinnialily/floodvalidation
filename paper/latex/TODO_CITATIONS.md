# TODO: Citations — Unresolved Items

## Status as of 2026-04-05

---

## [CITATION NEEDED] — FloodingDataset2 / USF FloodingDataset2

**Status:** NO FORMAL PAPER FOUND

**What was searched:**
- "FloodingDataset2 USF flood binary dataset street-level image classification paper"
- "FloodingDataset2 Hugging Face flood detection binary classification USF University South Florida"
- scholar.usf.edu, Google Drive, GitHub, arXiv

**What was found:**
The dataset is distributed informally via Google Drive from USF:
https://drive.google.com/drive/folders/1PXc9VTeQgV5WeNxa2NOC9kbRpnQtg20o

It contains street-level images with severity labels:
`StreetFloodClasses/{MajorFlood, MinorFlood, ModerateFlood, NoFlood, parks_walkways}`

No associated peer-reviewed paper, arXiv preprint, or DOI was found.

**Current resolution:** Cited as `@misc{FloodingDataset2}` in main.bib with the
Google Drive URL. Before submission, contact USF authors to confirm whether a
formal publication exists (technical report, data paper, or workshop paper).

**Action required before submission:**
1. Contact USF research team to ask for a citable publication
2. Check USF Scholar Commons (digitalcommons.usf.edu) for any data papers
3. If no paper exists, update the `@misc` entry with the most authoritative
   available URL (e.g., Hugging Face Hub link if dataset was uploaded there)

---

## RIWA — River Water Segmentation Dataset

**Status:** RESOLVED — cite as @Wagner2023 (peer-reviewed JAG paper)

**Citation added to main.bib:**
```
@article{Wagner2023,
  author = {Wagner, Franz and Eltner, Anette and Maas, Hans-Gerd},
  title  = {River water segmentation in surveillance camera images: ...},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  year   = {2023}, volume = {119}, pages = {103305},
  doi    = {10.1016/j.jag.2023.103305},
}
```

A companion Kaggle dataset release also exists:
- Blanch Gorriz, Wagner, Eltner (2023), DOI: 10.34740/kaggle/dsv/4901781
- URL: https://www.kaggle.com/datasets/franzwagner/river-water-segmentation-dataset

**Action:** None needed — citation is complete.

---

## AlleyFloodNet — Author correction

**Status:** RESOLVED — authors corrected from "Lee et al." to "Lee, O. and Joo, H."

**Verified from MDPI Electronics page:**
- Lee, Onhee and Joo, Hyunjin (2025)
- DOI: 10.3390/electronics14102082

---

## All other citations in main.bib

All entries are based on the verified reference list in `paper/paper.md`,
which was compiled from `results/literature/literature_review_flood_screening.md`
(53 papers with confirmed DOIs). No additional verification needed.

---

## Citation counts

| Status | Count |
|--------|-------|
| Fully resolved (BibTeX entry in main.bib) | 19 |
| Partially resolved (@misc, needs formal paper) | 1 (FloodingDataset2) |
| Unresolved | 0 |
