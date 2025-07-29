import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title("⚖️ Bias Dashboard")

st.markdown(
    """
    Upload a CSV with columns for **ground truth**, **predictions**, and a **demographic/group column** (e.g., gender, hospital).
    We'll compute accuracy and fairness metrics per group.
    """
)

uploaded = st.file_uploader("Upload evaluation CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(df.head())

    # Let user pick columns
    with st.form("bias_form"):
        all_cols = list(df.columns)
        y_true_col = st.selectbox("Ground truth column", all_cols)
        y_pred_col = st.selectbox("Prediction column", all_cols)
        group_col = st.selectbox("Group column (e.g., gender/hospital)", all_cols)
        submit = st.form_submit_button("Analyze")

    if submit:
        # Normalize labels to 0/1 if needed
        def to_binary(x):
            if isinstance(x, str):
                return 1 if x.lower() in ["tumor", "1", "positive", "yes"] else 0
            return int(x)

        df["y_true_bin"] = df[y_true_col].apply(to_binary)
        df["y_pred_bin"] = df[y_pred_col].apply(to_binary)

        groups = df[group_col].unique()
        rows = []
        for g in groups:
            dfg = df[df[group_col] == g]
            y_true = dfg["y_true_bin"]
            y_pred = dfg["y_pred_bin"]
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)  # TPR
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # For FPR we need TN/FP
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            fpr = fp / (fp + tn + 1e-9)

            rows.append({
                group_col: g,
                "n": len(dfg),
                "accuracy": acc,
                "precision": prec,
                "recall_TPR": rec,
                "FPR": fpr,
                "f1": f1,
            })

        result_df = pd.DataFrame(rows).sort_values(by="accuracy", ascending=False)
        st.subheader("Per-group metrics")
        st.dataframe(result_df, use_container_width=True)

        # Fairness gaps (max difference between best and worst groups)
        for metric in ["accuracy", "recall_TPR", "FPR"]:
            gap = result_df[metric].max() - result_df[metric].min()
            st.write(f"**{metric} gap:** {gap:.3f}")

        st.bar_chart(result_df.set_index(group_col)[["accuracy", "recall_TPR", "FPR"]])
else:
    st.info("Please upload a CSV to see bias metrics.")