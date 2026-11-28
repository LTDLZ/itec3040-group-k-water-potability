"""Water Potability Explorer - interactive dashboard (ITEC 3040, Group K).

Uses the team's saved models (models/*.joblib) and test-set predictions (results/predictions.csv).
Run with:  python -m streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score

import model as M

st.set_page_config(page_title="Water Potability Explorer", page_icon="💧", layout="wide")

POTABLE_COLOR, NOT_COLOR = "#1f77b4", "#d95f02"
CLASS_NAMES = {0: "Not potable", 1: "Potable"}
BASELINE = "Majority-Class Baseline"
sns.set_theme(style="whitegrid", context="notebook")

missing = M.missing_files()
if missing:
    st.error("Missing project files: " + ", ".join(missing) +
             ". Keep the folder structure from the zip (data/, models/, results/).")
    st.stop()


# ----------------------------------------------------------------- data / models
@st.cache_data
def get_data() -> pd.DataFrame:
    return M.load_data()


@st.cache_resource(show_spinner="Loading saved models...")
def get_results():
    df_ = M.load_data()
    X_train, X_test, y_train, y_test = M.split_data(df_)
    models = M.load_models()
    pred = M.load_predictions()
    proba = {name: pred[pcol].to_numpy() for name, (_, _, pcol) in M.MODEL_FILES.items()}
    return {
        "models": models, "pred": pred, "proba": proba, "y_test": pred["Actual"].to_numpy(),
        "X_train": X_train, "y_train": y_train, "X_test": X_test,
        "check": M.consistency_check(models, X_test, pred),
    }


@st.cache_data(show_spinner="Running 5-fold cross-validation...")
def get_cv() -> pd.DataFrame:
    r = get_results()
    return M.cv_accuracy(r["models"], r["X_train"], r["y_train"])


def label(col: str) -> str:
    name, unit, _ = M.FEATURE_INFO[col]
    return f"{name} ({unit})" if unit else name


df = get_data()
res = get_results()
model_names = list(res["models"])
scored = [n for n in model_names if n != BASELINE]

# ----------------------------------------------------------------- sidebar
st.sidebar.title("💧 Water Potability")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Explore the data", "Model performance", "Predict a sample", "Cohort explorer"],
)
st.sidebar.caption(
    "ITEC 3040 Final Project - Group K\n\n"
    "Educational tool only. Never use it to decide whether real drinking water is safe."
)


# ================================================================= OVERVIEW
def page_overview():
    st.title("Water Potability Explorer")
    st.write(
        "Can nine physicochemical measurements tell us whether a water sample is potable "
        "(safe to drink)? This dashboard lets you explore the data, compare the team's classification "
        "models, and try your own sample."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples", f"{len(df):,}")
    c2.metric("Features", len(M.FEATURES))
    c3.metric("Potable samples", f"{df[M.TARGET].mean():.1%}")
    c4.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    y = res["y_test"]
    acc = {n: M.metrics_at(y, res["proba"][n])["Accuracy"] for n in model_names}
    best = max(scored, key=lambda n: acc[n])
    st.subheader("Headline result")
    st.write(
        f"On the {len(y)} held-out test samples, **{best}** has the highest accuracy "
        f"(**{acc[best]:.1%}**) against **{acc[BASELINE]:.1%}** for the majority-class baseline, which just "
        "always answers 'not potable'. Accuracy alone is misleading here - see **Model performance** for "
        "precision, recall and the confusion matrix."
    )
    for n in scored:
        if M.metrics_at(y, res["proba"][n])["TP"] + M.metrics_at(y, res["proba"][n])["FP"] == 0:
            st.warning(
                f"**{n}** never predicts 'potable' at the default 0.5 threshold, so its accuracy equals the "
                "baseline. Its probabilities are all below 0.5 - try a lower threshold on the Model performance page."
            )
    st.info(
        "These measurements only weakly separate potable from non-potable samples, so even the "
        "best model makes many mistakes. Treat predictions as an illustration of the method, "
        "not a safety test."
    )

    st.subheader("What each page does")
    st.markdown(
        "- **Explore the data** - class balance, missing values, distributions, correlations.\n"
        "- **Model performance** - compare the saved models, move the decision threshold, see the confusion matrix, ROC curves, feature importance and cross-validation.\n"
        "- **Predict a sample** - enter water measurements and get a predicted probability from a saved model.\n"
        "- **Cohort explorer** - filter samples by measurement ranges, see how the potable share changes and how the models do on that cohort."
    )
    with st.expander("Data dictionary"):
        rows = [{"Field": c, "Name": v[0], "Unit": v[1] or "-", "Description": v[2]} for c, v in M.FEATURE_INFO.items()]
        rows.append({"Field": M.TARGET, "Name": "Potability", "Unit": "-", "Description": "Target: 1 = potable, 0 = not potable."})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with st.expander("Files used by this dashboard"):
        st.markdown(
            "- `models/*.joblib` - fitted baseline, logistic regression and random forest pipelines\n"
            "- `results/predictions.csv` - their predicted classes and probabilities on the 656 test samples\n"
            "- `scripts/train_models.py` - the script that produced both (80/20 stratified split, `random_state=42`)\n"
            "- `data/water_potability.csv` - the dataset"
        )
        ok = all(res["check"].values())
        st.write(("✅ " if ok else "⚠️ ") +
                 "Consistency check: re-predicting the test split with the saved models "
                 + ("reproduces predictions.csv exactly." if ok else f"does NOT match predictions.csv: {res['check']}"))


# ================================================================= EDA
def page_eda():
    st.title("Explore the data")
    tab1, tab2, tab3, tab4 = st.tabs(["Class balance", "Missing values", "Distributions", "Correlations"])

    with tab1:
        counts = df[M.TARGET].map(CLASS_NAMES).value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.bar(counts.index, counts.values, color=[NOT_COLOR if i == "Not potable" else POTABLE_COLOR for i in counts.index])
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,} ({v / len(df):.1%})", ha="center", va="bottom")
        ax.set_ylabel("Samples")
        ax.set_ylim(0, counts.max() * 1.15)
        st.pyplot(fig)
        st.caption("Moderately imbalanced: a model that always predicts 'not potable' would be right about 61% of the time.")

    with tab2:
        miss = df.isna().sum()
        miss = miss[miss > 0]
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.barh([label(c) for c in miss.index], miss.values, color="#7f7f7f")
        for i, v in enumerate(miss.values):
            ax.text(v, i, f" {v} ({v / len(df):.1%})", va="center")
        ax.set_xlabel("Missing values")
        ax.set_xlim(0, miss.max() * 1.25)
        st.pyplot(fig)
        st.caption(
            "Only ph, Sulfate and Trihalomethanes have gaps. In the saved models they are filled with the "
            "training-set median inside the pipeline (SimpleImputer), so nothing leaks from the test data."
        )

    with tab3:
        col = st.selectbox("Feature", M.FEATURES, format_func=label)
        kind = st.radio("View", ["Histogram", "Box plot"], horizontal=True)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        plot_df = df[[col, M.TARGET]].dropna().assign(Class=lambda d: d[M.TARGET].map(CLASS_NAMES))
        palette = {"Not potable": NOT_COLOR, "Potable": POTABLE_COLOR}
        if kind == "Histogram":
            sns.histplot(data=plot_df, x=col, hue="Class", palette=palette, bins=30,
                         element="step", stat="density", common_norm=False, ax=ax)
            ax.set_xlabel(label(col))
        else:
            sns.boxplot(data=plot_df, x="Class", y=col, palette=palette, hue="Class", legend=False, ax=ax)
            ax.set_ylabel(label(col))
        st.pyplot(fig)
        summary = df.groupby(df[M.TARGET].map(CLASS_NAMES))[col].agg(["count", "mean", "median", "std"]).round(2)
        st.dataframe(summary)
        st.caption(M.FEATURE_INFO[col][2] + " The two classes overlap heavily for every feature.")

    with tab4:
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    square=True, cbar_kws={"shrink": 0.8}, ax=ax)
        st.pyplot(fig)
        top = corr[M.TARGET].drop(M.TARGET).abs().sort_values(ascending=False)
        st.caption(
            f"Correlations are all weak. The largest with Potability is {label(top.index[0])} "
            f"(|r| = {top.iloc[0]:.2f}). Correlation does not imply causation."
        )


# ================================================================= MODEL PERFORMANCE
def page_models():
    st.title("Model performance")
    st.caption(f"Scored on the {len(res['y_test'])} held-out test samples in results/predictions.csv.")
    left, right = st.columns([1, 2])
    with left:
        chosen = st.selectbox("Model", model_names, index=model_names.index("Random Forest"))
        thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.01,
                        help="Predict 'potable' when the predicted probability is above this value.")
        st.caption("Lower threshold -> more samples called potable (higher recall, lower precision). "
                   "Higher threshold -> stricter (higher precision, lower recall). 0.50 reproduces the saved predictions.")
        p = res["proba"][chosen]
        st.caption(f"Probability range for {chosen}: {p.min():.2f} to {p.max():.2f}.")

    y = res["y_test"]
    m = M.metrics_at(y, res["proba"][chosen], thr)

    with right:
        c = st.columns(5)
        for ctl, k in zip(c, ["Accuracy", "Precision", "Recall", "F1", "ROC AUC"]):
            ctl.metric(k, f"{m[k]:.3f}")
        cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Pred: not potable", "Pred: potable"],
                    yticklabels=["Actual: not potable", "Actual: potable"])
        ax.set_title(f"{chosen} - confusion matrix")
        st.pyplot(fig)

    if m["TP"] + m["FP"] == 0:
        st.warning(f"At threshold {thr:.2f} this model predicts no sample as potable, so it behaves like the baseline. "
                   "Lower the threshold to see it call some samples potable.")
    else:
        st.markdown(
            f"**Reading this:** of {m['TP'] + m['FN']} truly potable samples the model finds {m['TP']} "
            f"(recall {m['Recall']:.0%}); of the {m['TP'] + m['FP']} it calls potable, {m['TP']} are correct "
            f"(precision {m['Precision']:.0%}). **{m['FP']} non-potable samples were wrongly labelled potable** - "
            "in a real safety setting this is the costly mistake, which argues for a higher threshold."
        )

    st.subheader("All models at this threshold")
    table = pd.DataFrame({n: M.metrics_at(y, res["proba"][n], thr) for n in model_names}).T
    st.dataframe(table.round(3).style.highlight_max(
        axis=0, subset=["Accuracy", "Precision", "Recall", "F1", "ROC AUC"], color="#cfe8cf"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("ROC curves")
        fig, ax = plt.subplots(figsize=(5, 4))
        for n in model_names:
            fpr, tpr = M.roc_points(y, res["proba"][n])
            auc = M.metrics_at(y, res["proba"][n], thr)["ROC AUC"]
            ax.plot(fpr, tpr, lw=2 if n == chosen else 1, label=f"{n} ({auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=7, loc="lower right")
        st.pyplot(fig)
    with c2:
        st.subheader("Feature importance")
        imp = M.feature_importance(res["models"][chosen])
        if imp.empty:
            st.info("The baseline has no features to rank.")
        else:
            is_forest = hasattr(M._final_step(res["models"][chosen]), "feature_importances_")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.barh([label(c).split(" (")[0] for c in imp.index[::-1]], imp.values[::-1], color=POTABLE_COLOR)
            ax.set_xlabel("Importance" if is_forest else "|Standardised coefficient|")
            st.pyplot(fig)
            st.caption("Importance shows what the model relies on, not what causes potability.")

    st.subheader("Cross-validation (training data only)")
    if st.checkbox("Run 5-fold cross-validation (a few seconds the first time)"):
        cv = get_cv()
        st.dataframe((cv * 100).round(2).rename(columns=lambda c: c + " %"))
        st.caption("Stratified 5-fold CV on the 2,620 training samples, as in scripts/train_models.py. "
                   "The test set is not used here.")


# ================================================================= PREDICT
def page_predict():
    st.title("Predict a sample")
    st.write("Enter water measurements to see the predicted probability that the sample is labelled potable.")

    for col in M.FEATURES:
        st.session_state.setdefault(f"in_{col}", float(df[col].median()))

    top = st.columns([2, 1, 1])
    chosen = top[0].selectbox("Model", scored, index=scored.index("Random Forest"), key="pred_model")
    thr = top[1].slider("Threshold", 0.05, 0.95, 0.50, 0.01, key="pred_thr")
    if top[2].button("Reset to medians"):
        for col in M.FEATURES:
            st.session_state[f"in_{col}"] = float(df[col].median())
        st.rerun()

    cols = st.columns(3)
    sample = {}
    for i, col in enumerate(M.FEATURES):
        lo, hi = float(df[col].min()), float(df[col].max())
        step = (hi - lo) / 200
        sample[col] = cols[i % 3].slider(label(col), lo, hi, step=step, key=f"in_{col}", help=M.FEATURE_INFO[col][2])

    pipe = res["models"][chosen]
    p = float(pipe.predict_proba(pd.DataFrame([sample])[M.FEATURES])[0, 1])
    verdict = "Potable" if p > thr else "Not potable"
    pmax = float(res["proba"][chosen].max())

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Predicted class", verdict)
        st.metric("Probability potable", f"{p:.1%}")
        st.progress(min(max(p, 0.0), 1.0))
        st.caption(f"Model: {chosen}. Threshold: {thr:.2f}.")
        if pmax <= thr:
            st.caption(f"Note: on the test set {chosen} never gave a probability above {pmax:.2f}, "
                       "so at this threshold it will almost never say 'potable'. Lower the threshold to see a difference.")
    with r2:
        st.write("**Where this sample sits** compared with the training data (dashed line = your value):")
        pick = st.multiselect("Features to compare", M.FEATURES, default=["ph", "Chloramines", "Turbidity"],
                              format_func=lambda c: label(c).split(" (")[0])
        if pick:
            fig, axes = plt.subplots(1, len(pick), figsize=(3.2 * len(pick), 2.8), squeeze=False)
            train = res["X_train"].join(res["y_train"])
            for ax, col in zip(axes[0], pick):
                for cls, colr in [(0, NOT_COLOR), (1, POTABLE_COLOR)]:
                    sns.kdeplot(train.loc[train[M.TARGET] == cls, col].dropna(), ax=ax, color=colr,
                                fill=True, alpha=0.25, label=CLASS_NAMES[cls], warn_singular=False)
                ax.axvline(sample[col], color="k", ls="--")
                ax.set_xlabel(label(col).split(" (")[0])
                ax.set_ylabel("")
                ax.set_yticks([])
            axes[0][0].legend(fontsize=7)
            plt.tight_layout()
            st.pyplot(fig)
    st.warning(
        "Illustration only. Model accuracy is modest, and real water safety requires laboratory "
        "testing against official standards."
    )


# ================================================================= COHORT EXPLORER
def page_cohort():
    st.title("Cohort explorer")
    st.write("Filter samples by measurement range, compare the cohort with the whole dataset, "
             "and see how the saved models perform on it.")
    chosen = st.multiselect("Filter on", M.FEATURES, default=["ph", "Chloramines"], format_func=label)
    mask = pd.Series(True, index=df.index)
    cols = st.columns(3)
    for i, col in enumerate(chosen):
        lo, hi = float(df[col].min()), float(df[col].max())
        rng = cols[i % 3].slider(label(col), lo, hi, (lo, hi), key=f"flt_{col}")
        # rows missing this value are excluded only once the filter is narrowed
        if rng != (lo, hi):
            mask &= df[col].between(*rng)
    sub = df[mask]

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples in cohort", f"{len(sub):,}", f"{len(sub) / len(df):.0%} of all")
    if len(sub):
        share, overall = sub[M.TARGET].mean(), df[M.TARGET].mean()
        c2.metric("Potable share", f"{share:.1%}", f"{(share - overall) * 100:+.1f} pts vs overall")
    else:
        c2.metric("Potable share", "-")
    c3.metric("Overall potable share", f"{df[M.TARGET].mean():.1%}")

    if len(sub) == 0:
        st.warning("No samples match these filters - widen a range.")
        return
    if len(sub) < 30:
        st.warning("Fewer than 30 samples: the potable share is very uncertain.")

    view = st.selectbox("Plot feature", M.FEATURES, format_func=label)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    sns.kdeplot(df[view].dropna(), ax=ax, color="#888", label="All samples", warn_singular=False)
    if sub[view].nunique() > 1:
        sns.kdeplot(sub[view].dropna(), ax=ax, color=POTABLE_COLOR, label="Cohort", warn_singular=False)
    ax.set_xlabel(label(view))
    ax.legend()
    st.pyplot(fig)

    st.subheader("How do the saved models do on this cohort?")
    pred = res["pred"]
    tr = pred[pred["sample_index"].isin(sub.index)]
    if len(tr) == 0:
        st.info("None of the cohort's samples are in the 656-sample test set, so there is nothing to score fairly.")
    else:
        rows = {}
        for n, (_, ccol, _) in M.MODEL_FILES.items():
            rows[n] = {
                "Test samples": len(tr),
                "Accuracy": accuracy_score(tr["Actual"], tr[ccol]),
                "Predicted potable": int(tr[ccol].sum()),
            }
        st.dataframe(pd.DataFrame(rows).T.round(3))
        st.caption(f"Only the {len(tr)} cohort samples that are in the held-out test set are scored (predictions at the "
                   "saved 0.5 threshold). " + ("With so few samples these numbers are noisy." if len(tr) < 30 else ""))

    with st.expander("Show cohort rows"):
        st.dataframe(sub)
    st.download_button("Download cohort as CSV", sub.to_csv(index=False).encode(), "cohort.csv", "text/csv")


{"Overview": page_overview, "Explore the data": page_eda, "Model performance": page_models,
 "Predict a sample": page_predict, "Cohort explorer": page_cohort}[page]()
