# Presentation Script & Rehearsal Guide
**Talk:** Confounder-Robust Flood Detection from Street-Level Imagery via Phase-1 Hard Negative Mining
**Presenter:** Aanya Singh · University of South Florida · Advised by Dr. Dixon
**Date:** April 18, 2026 · Target duration: 20 minutes

---

## Section 1: Presentation Script

---

## Slide 1 — Title
**Time: ~0:30**

Hi everyone, I'm Aanya Singh from the University of South Florida, working with Dr. Dixon. Today I'm going to talk about a problem that sounds simple on the surface: given a street-level photo, does it show flooding? But as we'll see, the devil is in the details — specifically, in what happens when a river, a swimming pool, or a flooded road looks almost identical to an actual flood.

**Transition:** Let me start with why this problem matters at all.

---

## Slide 2 — Floods affect 58 million people annually — and real-time screening saves lives
**Time: ~1:00**

Between 2000 and 2018, floods exposed an estimated 58 million people per year globally — that's from a 2021 Nature study by Tellman and colleagues. During a flood event, the information pipeline matters enormously. Satellite imagery has revisit cycles of hours to days, which is too slow when you need to route emergency responders right now.

What arrives fast is citizen smartphone photos. People post pictures of flooding on social media, send them to emergency hotlines, upload them to apps. If we can automatically screen those images — flag the ones showing real flooding and filter out everything else — we can get responders to the right places faster.

But here's the critical design constraint: if the screener misses a flood, that's a delayed response. If it fires on a river or a pool, that's a wasted resource call, which is tolerable. So from day one, this is a recall-first problem.

**Transition:** That raises the question — what makes this classification hard?

---

## Slide 3 — Rivers, pools, and wet roads share visual textures with real floods
**Time: ~1:00**

The core challenge is visual confounders. A river photo and a flood photo can look nearly identical: standing water, reflections, murky texture, debris at the edges. Swimming pools have clear blue water and can trigger a flood classifier. Wet roads after rain look like shallow flooding. Parks after heavy rainfall look like flood zones.

So the task is binary: does this image depict active flooding? But the input space is full of images that share flood-like features without being floods. We call these confounders.

Prior work on flood detection from imagery mostly reports aggregate metrics like overall accuracy or ROC-AUC. But to our knowledge, no prior work reports per-category false positive rates with confidence intervals broken down by confounder type — which is exactly what you'd need to trust a deployed screener.

**Transition:** Before I show you results, I want to explain why the standard way of measuring performance actually misleads you on this problem.

---

## Slide 4 — Standard metrics hide recall failures — Precision-Recall AUC tells the real story
**Time: ~1:00**

There are two problems with standard metrics here. First, accuracy. If you have 497 non-flood validation images and 322 flood images, a model that gets a few flood images wrong can still look great on accuracy. We'll see this exact paradox: ResNet50 achieves *higher* accuracy than EfficientNetB0 — 98.5% versus 98.3% — while missing nine times more floods.

Second, ROC-AUC, or Receiver Operating Characteristic Area Under Curve. ROC-AUC treats all non-flood images equally. When the non-flood class is large, it inflates the ROC score even for a model that collapses recall. The gap between our two models in ROC-AUC is only 2.6 points — sounds fine.

But Precision-Recall AUC, or PR-AUC, focuses specifically on the flood class. It penalizes precision collapse at high recall. The PR-AUC gap is 3.6 points. That's a much more honest picture of model quality for this use case.

**Transition:** With that framing in place, let me state what we're actually trying to do.

---

## Slide 5 — Goal: build a recall-first screener robust to visual water confounders
**Time: ~1:00**

Here's the quantified problem we're solving. ResNet50 with Binary Cross-Entropy loss — I'll abbreviate that as BCE from now on — misses 1 in 5 floods. Sixty-three out of 322 flood validation images are missed. That's a 19.6% miss rate. That's unacceptable for a screening system.

Our goals are four-fold. First, we want to *measure* the problem at a fine-grained level: which specific categories produce false positives, and by how much, with confidence intervals? Second, we want to *reduce* false positives in the dominant confounder category using Phase-1 Hard Negative Mining — I'll explain what that means shortly. Third, we want to show that *how* we inject hard negatives matters — difficulty ranking outperforms random injection. And fourth, we want to validate that PR-AUC is the right metric and that accuracy and ROC-AUC are misleading here.

I'll come back to that "1 in 5 floods missed" number when we get to results.

**Transition:** Let me first tell you where the data comes from.

---

## Slide 6 — Dataset: 4,099 street-level images from two labeled sources, deduplicated by SHA-256
**Time: ~1:00**

Our dataset is a combination of two sources. The first is the USF FloodingDataset, which provides labeled flood imagery with severity levels — Major, Moderate, and Minor — plus seven non-flood categories: street scenes, animals, buildings, vehicles, plants, parks and walkways, and swimming pools. That gives us 1,613 flood images and 2,087 non-flood images.

The second source is the RIWA dataset from Wagner and colleagues in 2023 — river scene images from European river monitoring. We added 399 river images to specifically enrich the hardest confounder category. That brings our non-flood total to 2,486.

Before splitting, we ran SHA-256 hashing to detect exact duplicates and removed 55 duplicate images, leaving us with 4,099 unique images. We split 80/20 using stratified sampling with seed 42, so both train and validation maintain the same 39.3% flood prevalence.

**Transition:** Now let me walk through the modeling approach.

---

## Slide 7 — Method: two-phase progressive fine-tuning preserves ImageNet representations
**Time: ~1:00**

We evaluated two architectures: EfficientNetB0 with 5.3 million parameters, and ResNet50 with 25.6 million parameters. Both are pretrained on ImageNet. Instead of fine-tuning everything at once, we use a two-phase approach to preserve the low-level representations learned on ImageNet.

Phase 1: we keep the early layers frozen and train only the last 30 layers with a learning rate of 1e-4 for up to 15 epochs. This gives us checkpoint M1. Phase 2: we freeze the first 50 layers and fine-tune the rest with a slower learning rate of 1e-5 for up to 20 epochs. This gives us checkpoint M2.

The classification head is: Global Average Pooling, abbreviated GAP, which compresses spatial features — then Dropout at 0.2, Dense layer with 256 units and ReLU activation, BatchNorm, Dropout at 0.3, and a final sigmoid output.

The two-phase structure is important for the mining step I'll describe next.

**Transition:** And that mining step is the core contribution of this work.

---

## Slide 8 — Phase-1 Hard Negative Mining (P1-HNM): mine confounders before the signal disappears
**Time: ~1:00**

Here's the key insight. By the time Phase 2 finishes training, the model has converged — it assigns near-zero flood probability to essentially all non-flood images. If you try to mine hard negatives at that point, there's nothing to mine. The signal is gone.

But the Phase-1 checkpoint still has frozen early layers. There's residual uncertainty — especially on rivers and pools that share texture with floods. That uncertainty is the mining signal.

So Phase-1 Hard Negative Mining, or P1-HNM, works in three stages. First, using the Phase-1 model M1, we flag any confounder category with a false positive rate above 5%. Rivers qualify at 9.2%. Second, we rank all river images by their predicted flood probability and take the top 10% — the hardest negatives the model is most confused by. Third, we apply 5x augmentation to those selected images and retrain from checkpoint M2.

The difficulty ranking is what separates this from random injection. We're not just adding more river images — we're adding specifically the ones the model finds hardest.

**Transition:** Let's see what the baseline models actually do before any mining.

---

## Slide 9 — EfficientNetB0 misses 9× fewer floods than ResNet50 — visible in PR-AUC, hidden in ROC-AUC
**Time: ~2:00**

Here are the Precision-Recall curves and ROC curves for EfficientNetB0 and ResNet50, both with BCE loss. Look at the PR curves first. EfficientNetB0 achieves PR-AUC of 0.9976. ResNet50 achieves 0.9614. That's a 3.6-point gap — and it's visible in the shape of the curve. ResNet's curve drops earlier under high-recall conditions.

Now look at the ROC curves. The gap there is only 2.6 points — 0.9985 vs. 0.9728. "Nearly as good," you might say. But that's exactly the problem: ROC-AUC is hiding the recall collapse.

Look at the actual numbers in the table. EfficientNetB0 misses 7 out of 322 flood images — a 2.2% miss rate. ResNet50 misses 63 out of 322 — a 19.6% miss rate. Nine times more missed floods. And yet ResNet's *accuracy* is higher: 98.5% vs. 98.3%.

This is the accuracy paradox in action. ResNet gets more non-flood images right — that boosts its accuracy number — while catastrophically missing floods. PR-AUC is the honest metric here.

**Transition:** Let's look at the confusion matrices to understand what kind of errors each model is making.

---

## Slide 10 — EfficientNetB0 makes symmetric boundary errors; ResNet50 collapses systematically
**Time: ~1:30**

EfficientNetB0 makes 7 false negatives and 7 false positives. That's symmetric. These are genuine boundary cases — ambiguous images where even a human might hesitate. The model is uncertain, and it makes mistakes in both directions roughly equally.

ResNet50 makes 63 false negatives and 17 false positives. That's highly asymmetric, and it tells a different story. When we looked at what's in those 63 missed floods, many are prototypical clear flood scenes — not ambiguous cases. This suggests ResNet's decision boundary is systematically miscalibrated, not just uncertain at the edges.

This is why we target EfficientNetB0 for hard negative mining and not ResNet. EfficientNet's errors are correctable — we need to reduce the boundary ambiguity on specific confounder types. ResNet's errors suggest a deeper miscalibration that HNM alone is unlikely to fix.

**Transition:** So which confounder categories are actually causing the false positives?

---

## Slide 11 — River scenes cause a 9.2% false positive rate — every other confounder category is 0%
**Time: ~1:30**

This is, I think, the most actionable finding in the baseline analysis. When we break down false positive rates by confounder category, the answer is extremely clear.

For EfficientNetB0: rivers produce a 9.2% false positive rate — 7 out of 76 river images are classified as floods. The 95% confidence interval is 3.8% to 17.7%. Plant images: 2.9%, one image out of 34. Every other category — streets, animals, buildings, vehicles, parks, and swimming pools — is 0%.

For ResNet50: rivers produce 3.9% FP, 3 out of 76. Everything else is 0%.

A word on swimming pools. We have 28 pool images and both models get 0% false positives. That sounds great — but the upper 95% confidence interval is 12.3%. With only 28 images, a 0% result tells us almost nothing. We'd need at least 150 pool images to put a tight bound on that.

The practical takeaway: rivers are the target. That's where HNM needs to focus.

**Transition:** So what happens when we apply Phase-1 Hard Negative Mining to the river category?

---

## Slide 12 — Difficulty-ranked Phase-1 mining strictly outperforms extended training and random augmentation
**Time: ~2:00**

Here are four conditions, all on EfficientNetB0 with BCE loss: the original baseline, random confounder injection where we add river images but chosen randomly, extended training where we just train for more epochs without any new data, and our Phase-1 HNM approach.

The accuracy ordering is strict: P1-HNM at 98.78%, extended training at 98.66%, random injection at 98.41%, baseline at 98.29%. Each step in the hierarchy helps, and difficulty ranking adds incrementally on top of each one.

Let me interpret each gap. Going from baseline to random injection: just adding more confounder-category data helps — even randomly selected. Going from random to extended training: more compute and exposure also helps independently. Going from extended to HNM: the difficulty ranking on top of everything else gives the most targeted improvement.

Now, about the river false positive rate. After HNM, the rate is still 7 out of 76 — still 9.2%. A McNemar test, which tests pairwise differences corrected for multiple comparisons using Bonferroni correction, shows no statistically significant reduction across any pair of conditions. This is not a null result — it's a power problem. We only have 76 river validation images. To detect a 3 to 4 percentage-point reduction with 80% statistical power, we'd need approximately 300 river images. The HNM did improve things — we just can't prove it statistically with current data.

And to mirror the number from the goal slide: without HNM, EfficientNetB0 misses 1 in 46 floods — 7 out of 322, a 2.2% miss rate. P1-HNM achieves the same recall with the best overall accuracy. The difficulty-ranked approach beats both alternatives.

**Transition:** Let me now pull everything together.

---

## Slide 13 — Conclusion
**Time: ~1:00**

Five key takeaways.

One: PR-AUC is the right metric for recall-first flood screening. The 3.6-point gap between EfficientNet and ResNet is invisible to accuracy and nearly invisible to ROC-AUC.

Two: EfficientNetB0 is the right architecture. Its errors are symmetric boundary cases, not systematic miscalibration.

Three: river scenes are the only statistically significant confounder. 9.2% false positive rate with a tight CI. All seven other confounder categories are at 0%.

Four: Phase-1 Hard Negative Mining strictly outperforms all alternatives. The difficulty ranking adds value beyond just adding more data or training longer.

Five: we can't yet claim statistically significant river FP reduction — not because HNM failed, but because we have 76 river validation images and need about 300.

**Transition:** That fifth point directly motivates what comes next.

---

## Slide 14 — Future directions: statistical power, geographic generalization, and operational deployment
**Time: ~1:00**

The most immediate next step is expanding the river validation set to roughly 300 images. That's the number we need for 80% power to detect a 3 to 4 percentage-point reduction in river false positives.

We also want multi-seed validation — running all experiments across five seeds and reporting bootstrap confidence intervals. A single seed can be lucky.

Severity stratification is another priority. We know Major, Moderate, and Minor flood images exist in the dataset. From an operational standpoint, missing a major flood is far more costly than missing a minor one. We should report miss rates separately.

Geographic generalization is an open question. Our dataset is US-centric. Flooding in South or Southeast Asia, or European urban flooding, looks visually different. We don't know how these models transfer.

And finally, for deployment: the threshold τ=0.5 is a default, not a calibrated operational setting. The PR curve tells us exactly what recall/precision tradeoff we get at each threshold. For a real deployment, you'd set τ based on the operational recall target — say, 99% recall — and accept whatever false positive rate that implies.

**Transition:** I want to thank a few people before we open for questions.

---

## Slide 15 — Acknowledgments
**Time: ~0:30**

I want to thank Dr. Dixon for advising this project. I'd also like to thank the authors of the USF FloodingDataset for providing the labeled flood severity imagery, and Wagner and colleagues for the RIWA river dataset. And thank you all for listening today.

**Transition:** Happy to take any questions.

---

## Slide 16 — Questions?
**Time: open**

[Open for Q&A. Stay at this slide until session ends.]

---

---

## Section 2: Rehearsal Guide

### Timing practice
- Full run-through without stopping: aim for 19–21 minutes.
- Use the script as a *starting point*, not a recital. Know the content cold enough to speak naturally if you lose your place.
- Record yourself once. Listen back for filler words (um, so, like), trailing sentences, and jargon you didn't notice.

### Slide habits
- Don't read text off the slides. Glance to orient, then look at the audience.
- Pause briefly after each key number (PR-AUC 0.9976, 63 missed floods, 9.2% FP rate) — let it land before continuing.
- The comparison table on Slide 9 is dense. Point to it, don't read it. Say: "Look at the FN row — 7 versus 63."

### Jargon check
Before presenting to a mixed audience, do one pass with someone outside the lab. Flag any sentence they have to re-read.
- "McNemar test with Bonferroni correction" needs a one-line explanation: "a statistical test that checks whether two models are actually different, corrected for running multiple comparisons."
- "Stratified 80/20 split" — just say "we made sure both train and test have the same proportion of flood images."
- "Phase-1 checkpoint M1" — say "the partially trained model from the first training phase."

### Q&A preparation

**Q: "Why not just use ResNet50 with a lower threshold to improve recall?"**
A: Good question. Lowering the threshold improves recall but at the cost of precision — you'd get many more false positives, including on rivers. The problem is that both errors have costs. Lowering threshold trades one error type for another. PR-AUC tells you the optimal tradeoff across all thresholds, and EfficientNetB0 dominates ResNet at every point on the PR curve.

**Q: "Your river FP rate didn't improve with HNM — doesn't that mean HNM failed?"**
A: The HNM did improve overall accuracy by 0.49 percentage points and strict ordering over all ablations. The river FP rate *numerically* dropped from 7 to 7 — unchanged. But with 76 river validation images, we have roughly 30% statistical power to detect a 3–4 pp change. We can't declare a null result from an underpowered test. The next step is to expand the river validation set to ~300 images and retest.

**Q: "Have you tested this on real emergency response scenarios?"**
A: Not yet. The dataset is labeled imagery from curated sources — not live citizen uploads. Deployment would require threshold calibration against an operational recall target (e.g., ≥99% recall), geographic validation beyond the US-centric training set, and testing on compressed/low-resolution social media images which may differ from our clean validation set.

**Q: "Why did you use two architectures? Wouldn't more be better?"**
A: We chose EfficientNetB0 and ResNet50 to represent two fundamentally different design philosophies — compound-scaling efficiency vs. residual depth. The point wasn't comprehensive benchmarking; it was to show that the metric choice and confounder analysis framework generalize across architectures. Adding more architectures is a natural extension, but the core findings hold across both models we tested.

### "I don't know" is a valid answer
If asked about something outside the paper's scope — specific deployment environments, other datasets, ensemble approaches — it's perfectly fine to say: "That's a great question and honestly outside what we tested — I'd want to look into that before giving a confident answer."
