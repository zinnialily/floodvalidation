# Paper Editing Progress

## Status: COMPLETE ✓ — PDF builds cleanly, 3,635 prose words, 18 pages, no errors

---

## What is DONE

### Pre-flight (all complete)
- [x] Copied `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` from `~/Downloads/Computers___Geosciences__1_/` to `paper/latex/`
- [x] Added `\graphicspath{{../../}}` to `paper/latex/main.tex` preamble (after `\usepackage{graphicx}`)
- [x] Fixed `\bibliography{bibliography}` → `\bibliography{main}` (line ~969)
- [x] Fixed `main.bib` line 282 Alam2023 missing comma: `number = {3}` → `number = {3},`

### Template compliance (all complete)
- [x] Replaced placeholder highlights (lines ~202–208) with 5 real highlights
- [x] Replaced placeholder keywords with: `Flood detection \sep Hard negative mining \sep Transfer learning \sep Precision-recall \sep Visual confounders \sep Convolutional neural networks`

### Figure cuts (all complete)
- [x] Removed `fig:architecture` TikZ block (was ~lines 764–821)
- [x] Removed `fig:transfer` TikZ block (was ~lines 864–917)
- [x] Removed `fig:hnm_loop` TikZ block (was ~lines 960–1005)
- [x] Removed `fig:training_curves` includegraphics block (was ~lines 1196–1209)
- [x] Removed `fig:confusion` includegraphics block (was ~lines 1287–1319)
- [x] Removed `fig:fp_heatmap` includegraphics block + surrounding text
- Kept: `fig:pipeline`, `fig:prroc`, `fig:fp_examples`, `fig:fp_bar`, `fig:hnm_ablation` ✓

### Section cuts (all complete)
- [x] **Abstract**: deleted standalone HNM sentence; compressed eval methodology into previous sentence
- [x] **Introduction**: deleted visual confounders 6-sentence paragraph; compressed research gap to 2 sentences; compressed contributions list from 5 → 3 items; deleted paper organisation paragraph
- [x] **Related Work**: all 5 subsections compressed
- [x] **Dataset**: all 4 subsections compressed (sources, dedup, split, stats)
- [x] **Methods**: deleted Keras3 detail, ResNet mixing sentence, phase rationale paragraphs, callbacks list compressed; loss functions compressed to 3 sentences + 2 equations; HNM figure removed, stages compressed; ablation prose compressed; eval protocol compressed
- [x] **Results**: deleted baseline training dynamics subsection entirely (including `fig:training_curves`); baseline performance compressed to 4 sentences; PR-AUC section compressed; qualitative analysis compressed to 3 sentences; confounder analysis compressed; HNM ablation compressed; `fig:fp_heatmap` removed with surrounding text; `fig:prroc` caption updated to mention Phase-1-to-Phase-2 spike
- [x] **Discussion**: all subsections compressed (metric, focal loss, why Phase-1 works, HNM river FP, limitations to 1 sentence each, practitioner recs to 1 sentence imperative each)
- [x] **Conclusion**: all paragraphs compressed

---

## What is STILL TODO (small tasks)

### 1. Appendix prose cuts (~5 min)
The appendix section prose still has the internal note and verbose intro. Apply these two edits to `paper/latex/main.tex`:

**Hyperparameters intro** — current text (~line 978–982):
```
\Cref{tab:hyperparams} lists all training hyperparameters.
Phase-1 and Phase-2 learning rate values are verified against the source code
constants \texttt{PHASE1\_LR\,=\,1e-4} and \texttt{PHASE2\_LR\,=\,1e-5}
in \texttt{scripts/train\_baseline.py}, correcting an earlier draft that
incorrectly listed $10^{-3}$ and $10^{-4}$.
```
Replace with:
```
\Cref{tab:hyperparams} lists all training hyperparameters.
```

**Clopper-Pearson intro** — current text (~line 1024–1028):
```
\Cref{tab:cp} reports the upper bound of the 95\,\% \CP{} CI for an observed
FP rate of 0\,\% ($k{=}0$) as a function of validation sample size.
This reference table helps readers interpret the ``0\,\% FP'' results in
\cref{tab:confounder}: a zero observed rate does not mean zero true rate,
and the uncertainty grows substantially as $N$ decreases.
```
Replace with:
```
\Cref{tab:cp} lists the 95\,\% \CP{} CI upper bound for zero observed FPs, by sample size.
```

### 2. Build & verify word count
```bash
cd paper/latex && pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
detex main.tex | wc -w   # target: ≤5,500
```
Check: all 5 figures render, no `[?]` citation placeholders, no LaTeX errors.

---

## Key file state
- `paper/latex/main.tex`: 1056 lines (was 2000 lines)
- `paper/latex/main.bib`: fixed
- Template files in `paper/latex/`: `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst` ✓

## Three claims preserved
1. **Phase-1 HNM works** — recall 97.8%→99.1% (7FN→3FN), river FP 9.2%→5.3%, beats extended training and random injection. Evidence: tab:hnm + fig:fp_bar ✓
2. **PR-AUC is the right metric** — aggregate accuracy/ROC-AUC are insufficient. Evidence: fig:prroc + tab:baseline ✓
3. **River is the dominant confounder** — all other categories zero FPs; swimming pool CI caveat. Evidence: tab:confounder ✓

## Stale cross-references to check after build
- `sec:results:dynamics` — subsection was deleted; intro sentence in Results section that referenced it was removed. ✓
- `fig:hnm_loop` — removed from contributions list ✓ (contributions list now refers generically to pipeline)
- `fig:architecture` — reference removed from methods prose ✓
- `fig:transfer` — reference removed from methods prose ✓
- `fig:fp_heatmap` — reference and block removed ✓
- `fig:training_curves` — block and references removed ✓
- `fig:confusion` — block and references removed ✓
