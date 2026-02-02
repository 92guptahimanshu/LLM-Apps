import streamlit as st
import pandas as pd
import json
import re
from groq import Groq

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="AI Data Quality Engine", page_icon="✅", layout="wide")
st.title("🧠 AI-Powered Data Quality Engine")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    groq_key = st.text_input("Groq API Key", type="password")
    st.markdown("Supported rules:\n- between X and Y\n- greater than X\n- valid email\n- not null\n- starts with VALUE")

# ----------------------------
# Upload CSV
# ----------------------------
file = st.file_uploader("Upload CSV File", type=["csv"])

if not file:
    st.stop()

df = pd.read_csv(file)
st.subheader("📄 Preview Data")
st.dataframe(df.head())

columns = df.columns.tolist()

# ----------------------------
# User Rule Input
# ----------------------------
st.subheader("✍️ Accuracy Rules (Plain English)")
rules_input = {}

for col in columns:
    rules_input[col] = st.text_input(f"Rule for `{col}` (optional)")

# ----------------------------
# LLM Rule Conversion
# ----------------------------
def llm_parse_rules(rules):
    client = Groq(api_key=groq_key)

    prompt = f"""
Convert the following English data quality rules into JSON.
Only output valid JSON. No explanation.

Rules:
{json.dumps(rules, indent=2)}

JSON format:
column:
  type: range|min_value|regex|not_null|starts_with
  values if needed
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# ----------------------------
# Accuracy Evaluation Engine
# ----------------------------
def apply_rule(series, rule):
    if rule["type"] == "range":
        return series.between(rule["min"], rule["max"], inclusive="both")
    if rule["type"] == "min_value":
        return series >= rule["value"]
    if rule["type"] == "not_null":
        return series.notna()
    if rule["type"] == "starts_with":
        return series.astype(str).str.startswith(rule["value"])
    if rule["type"] == "regex":
        return series.astype(str).str.match(rule["pattern"])
    return pd.Series([True] * len(series))

# ----------------------------
# Run Checks
# ----------------------------
if st.button("🚀 Run Data Quality Checks"):

    if not groq_key:
        st.error("Groq API key required")
        st.stop()

    # -------- Completeness --------
    st.subheader("1️⃣ Completeness")
    completeness = (1 - df.isnull().mean()) * 100
    st.dataframe(completeness.rename("Completeness %"))

    # -------- Accuracy --------
    st.subheader("2️⃣ Accuracy")

    with st.spinner("Interpreting rules using AI..."):
        parsed_rules = llm_parse_rules({k: v for k, v in rules_input.items() if v})

    accuracy_scores = {}
    failed_rows = pd.DataFrame()

    for col, rule in parsed_rules.items():
        mask = apply_rule(df[col], rule)
        accuracy_scores[col] = round(mask.mean() * 100, 2)
        failed = df.loc[~mask]
        if not failed.empty:
            failed["Failed_Column"] = col
            failed_rows = pd.concat([failed_rows, failed])

    st.dataframe(pd.Series(accuracy_scores, name="Accuracy %"))

    # -------- Scorecard --------
    st.subheader("3️⃣ DQ Scorecard")
    dq_score = round(
        (completeness.mean() + pd.Series(accuracy_scores).mean()) / 2, 2
    )
    st.metric("Overall DQ Score", dq_score)

    # -------- Failed Rows --------
    if not failed_rows.empty:
        st.subheader("❌ Failed Records")
        st.dataframe(failed_rows)

        csv = failed_rows.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Failed Rows",
            csv,
            "failed_records.csv",
            "text/csv",
        )
    else:
        st.success("🎉 No failed records found!")
