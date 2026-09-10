# XGBoost

## 1. What is XGBoost?

XGBoost (eXtreme Gradient Boosting) is a scalable, open-source decision-tree ensemble built on gradient boosting (xgboost_1, xgb_2:37). It combines many shallow regression trees additively, each new tree correcting the errors of all previous ones, into a single strong model. The original paper's selling point is scalability: it handles sparse data, parallelizes tree building, and scales to billions of examples on one machine (bilions via out-of-core/compression/sharding), running ~10x faster than scikit-learn's GBM (xgboost_1:785). It won 17 of 29 Kaggle 2015 challenges.

## 2. Architecture

The model is a sum of K regression trees (CART), where each tree maps an input to a leaf and every leaf holds a continuous score (xgboost_1, Eq. 1). So the prediction is:
ŷᵢ = f₁(xᵢ) + f₂(xᵢ) + ... + f_K(xᵢ)
Key architectural pieces (xgboost_1, Sec. 2–4):

- **Regularized objective:** L = Σ loss(ŷᵢ, yᵢ) + Σ Ω(fₖ), where Ω(f) = γT + ½λ‖w‖² penalizes the number of leaves T and leaf weights w to stop overfitting (xgb_2 Eq. 6 calls γT a "pre-pruning" — higher γ → simpler trees). The loss part, Σ loss(ŷᵢ, yᵢ), simply measures "how wrong is the prediction ŷ vs. the true value y" — summed over all data points. Lower is better.

But if a model only cares about being right, it overfits. It memorizes the training data perfectly but fails on new data. So XGBoost adds a second term, Σ Ω(fₖ), a "complexity tax" on each tree:
Ω(f) = γT + ½λ‖w‖²

- T = number of leaves in the tree. Bigger T = more complicated tree. γ·T charges a fee per leaf, so the tree won't grow extra leaves unless they genuinely help.
- w = the numbers stored in those leaves (the actual predictions). ½λ‖w‖² charges a fee proportional to the size of those numbers — it nudges leaves toward small, moderate values instead of huge ones.

**Put simply:** XGBoost wants trees that are accurate but also simple — few leaves, modest values. The γT part is called "pre-pruning": if a split does not reduce error by at least γ, the tree just doesn't add that leaf (it prunes itself while growing, hence "pre"). Raise γ → fewer leaves → simpler, safer, shallower trees.
λ is the user set knob that controls how strong the penalty is.

- **Shrinkage:** each tree's contribution is scaled by learning rate η (like SGD learning rate). Boosting builds an ensemble one tree at a time, each new tree trying to fix what the previous ones got wrong. Shrinkage says: don't let any single tree have full effect.
Instead of adding a whole tree's prediction, only add a fraction of it:
new prediction = old prediction + η × (new tree's output)
with η typically 0.05 or 0.1 (your forecaster uses learning_rate=0.05, xgboost_forecaster.py:74).
Why? It's insurance against overfitting. If one tree is allowed to fully correct the errors right now, the model latches onto quirks of this particular training data too strongly. By only moving a small step each time, the model is forced to fix errors gradually across many trees — like SGD's learning rate, where you take small steps toward the minimum instead of one giant jump. The classic analogy: small steps = more stable, generalizes better; it just needs more steps (trees) to converge. That's why your forecaster uses a small learning_rate but a large n_estimators=300

- **Column and row subsampling:**
  - **column subsampling:** when building each tree, don't consider all features at every split. Only look at a random subset of columns when picking a split. colsample_bytree=0.8 in your forecaster (xgboost_forecaster.py:144) = each tree only considers 80% of the lag/calendar features.
  - **row subsampling:** don't train each tree on all rows; give it a random 80% of them (subsample=0.8, xgboost_forecaster.py:143). Each tree sees a slightly different slice of history.
  - **Why it's done:**  if every tree sees all features and all rows, they all learn the same patterns and vote identically — the ensemble is just one big tree. By jittering features and rows per tree, each tree learns a different view, and their combined opinion (summed prediction) is smoother and generalizes better. This is exactly how Random Forest works (random features + bootstrap rows), which is why XGBoost borrows the idea

- **Pre-sorted column blocks:** Finding a good split means scanning a feature, checking "what if I split here?" at every value. Sorting is the expensive part. Normal libraries re-sort per node — wasteful. XGBoost sorts each feature once upfront and stores it as a ready column (= CSC format). From then on, split-finding is just a quick scan of the sorted column. Splitting is actually a sorting problem. For eg: for feature lag_48, if I cut at value 5000 vs 5001 vs ... — which single point separates lazy days from peak days best?" To evaluate a cut, you need to know the statistics (gradient sums) of points left of the cut vs right of the cut, for every possible cut. If data is in *unsorted* order, checking cut-at-value-5000 means scanning all rows (is each one ≤ 5000 or > 5000?). Checking the next candidate cut at 5001 means rescanning everything again. That's O(n) per candidate = catastrophically slow. If data is *sorted by the feature value*, you just walk down the sorted list once, adding each point to the "left side" as you pass it:

- **Parallel split-finding:** Since each feature column is independent, XGBoost searches for the best split in multiple columns at the same time on different CPU threads. That's why your forecaster's n_jobs=4 (xgboost_forecaster.py:145) makes training faster.

- **Sparsity-aware (handles missing values)** Real data has gaps (your demand series has NaN lags). XGBoost doesn't ignore them — at each node it *learns a "default direction"*: if the value is missing, send it left or right automatically, whichever worked best during training. No manual imputation needed.

## How does it work

XGBoost = train one tiny tree, learn from its mistakes, train another tree on those mistakes, repeat. Final answer = sum of all trees.

**Our mini-example:** predict today's demand y from one feature lag_48 (demand 24h earlier). 4 training rows:
x (lag_48)   y (demand)
4800         4900
5000         4950
5200         5300
5300         5350

**Step 0 — Start dumb:** Predict the average for everyone: mean = 5125. So all four forecasts are 5125, and the errors are:
actual  forecast  error (actual − forecast)
4900    5125      -225
4950    5125      -175
5300    5125      +175
5350    5125      +225

**Step 1 — Measure what needs fixing:** XGBoost turns each error into two numbers, the gradient g (direction/size of the mistake — roughly "negative error") and h (the second derivative, =1 for squared error). We just need g:
g values:  +225, +175, -175, -225
           (we were too high → need to come DOWN)
           (we were too low  → need to come UP)

**Step 2 — Grow a tree that fixes these:** Try a split: "x ≤ 5100" puts rows 1–2 on the left, rows 3–4 on the right.
Gain of the split = ½[GL²/(HL+λ) + GR²/(HR+λ) − G²/(H+λ)] − γ (xgboost_1, Eq. 7):

- Left: GL = 225+175 = 400, HL = 2 → 400²/2 = 80,000
- Right: GR = −175−225 = −400, HR = 2 → 80,000
- Parent: G = 0 → term is 0
- Gain = ½[80,000+80,000−0] − γ = 80,000 − γ
Huge gain → this split clearly separates "low-demand" rows (left) from "high-demand" rows (right). If the gain hadn't beaten γ, the tree would refuse to split (that's the pre-pruning from before).

**Step 3 — Decide how much to say.** Each leaf's value: w = −G/(H+λ):

- Left leaf: −400/2 = −200 → "if x ≤ 5100, subtract 200"
- Right leaf: −(−400)/2 = +200 → "if x > 5100, add 200"

**Step 4 — Shrink it.** Don't apply the full tree, only η = 0.1 × it. New predictions:
5125 + 0.1×(−200) = 5105   for rows 1–2   (moving 5125 → toward 4900/4950)
5125 + 0.1×(+200) = 5145   for rows 3–4   (moving 5125 → toward 5300/5350)
Small step — but notice the errors did shrink in the right direction.

**Step 5 — Repeat.** Recompute g from the new predictions (everyone got a bit closer), add another tree, shrink again. After 300 rounds, the 300 little corrections stack up and the sum lands close to the true values.

**Prediction time.** New row with x = 5150 falls into the right side of our first tree, right side of some trees, left of others (deeper splits) — it accumulates all 300 leaves' contributions, and the total is the forecast. No training, just routing each row down the trees and summing.

## 4. How it connects to the 4 adaptation arms

The contract is adapter(changepoints, model, data) -> updated model (adaptation/base.py:3), and Step 5 requires four arms, each a small Adapter subclass:

- **Arm 1 — never retrain:** XGBoost is fit once on the 2018–2019 window and rolled forward untouched. adapt() returns the model unchanged. Cheapest, but inert under COVID (2020-03) and 5MS cutover (2021-10) drift.
- **Arm 2 — retrain on a schedule:** adapt() refits XGBoost on a fixed cadence (e.g. every N days) regardless of changepoints — XGBoost's fit() is incremental-friendly here since our fit just rebuilds feature+target and calls xgb.XGBRegressor.fit, so retraining = calling model.fit(X, y) again.
- **Arm 3 — retrain on drift, full history:** when changepoints is non-empty, refit XGBoost on all data up to that point (data), i.e. full history including previous drift segments. The model re-learns lags/levels from scratch over everything.
- **Arm 4 — retrain on drift, recent window:** same trigger, but fit on only data.tail(window) (e.g. 60 days * 48 rows), as in adaptation/base.py:18-28. Drops old COVID-era patterns that no longer describe the new regime.

## References

1. Chen, T. and Guestrin, C. (2016). XGBoost: a Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining - KDD ’16, 1(1), pp.785–794. doi:10.1145/2939672.2939785.
2. Bentéjac, C., Csörgő, A. and Martínez-Muñoz, G. (2020). A Comparative Analysis of Gradient Boosting Algorithms. Artificial Intelligence Review, [online] 54(3). doi:10.1007/s10462-020-09896-5.
