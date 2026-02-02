import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
from groq import Groq

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="AI Data Quality Checker",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Data Quality Engine (Enterprise)")
st.caption("Completeness • Validity • Uniqueness • Accuracy (AI + Manual)")

# -----------------------------
# Groq API Key
# -----------------------------
with st.sidebar:
    groq_api_key = st.text_input("🔑 Enter Groq API Key", type="password")

if not groq_api_key:
    st.warning("Please enter Groq API key")
    st.stop()

client = Groq(api_key=groq_api_key)

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV file to start")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Dataset Preview")
st.dataframe(df.head())

all_columns = df.columns.tolist()

# -----------------------------
# Column selection
# -----------------------------
st.subheader("Select Columns")
completeness_cols = st.multiselect("Completeness Columns", all_columns, default=all_columns)
validity_cols = st.multiselect("Validity Columns", all_columns)
uniqueness_cols = st.multiselect("Uniqueness Columns", all_columns)
accuracy_cols = st.multiselect("Accuracy Columns", all_columns)

# -----------------------------
# Safe JSON loader
# -----------------------------
def safe_json_loads(text: str):
    if not text or not text.strip():
        raise ValueError("Empty AI response")
    text = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No valid JSON found in AI response")

# -----------------------------
# Completeness
# -----------------------------
st.subheader("1️⃣ Completeness")
completeness = {col: round((1 - df[col].isna().mean()) * 100, 2) for col in completeness_cols}
st.dataframe(pd.DataFrame.from_dict(completeness, orient="index", columns=["Completeness %"]))

# -----------------------------
# Validity
# -----------------------------
st.subheader("2️⃣ Validity")

def email_valid(series):
    return series.astype(str).str.match(r"^[\w\.-]+@[\w\.-]+\.\w+$").mean() * 100

def numeric_valid(series):
    return pd.to_numeric(series, errors="coerce").notna().mean() * 100

validity = {}
for col in validity_cols:
    if df[col].dtype == "object" and df[col].astype(str).str.contains("@").any():
        validity[col] = round(email_valid(df[col]), 2)
    elif df[col].dtype != "object":
        validity[col] = round(numeric_valid(df[col]), 2)
    else:
        validity[col] = None

st.dataframe(pd.DataFrame.from_dict(validity, orient="index", columns=["Validity %"]))

# -----------------------------
# Uniqueness
# -----------------------------
st.subheader("3️⃣ Uniqueness")
uniqueness = {}
for col in uniqueness_cols:
    non_null = df[col].dropna()
    uniqueness[col] = round(non_null.nunique() / len(non_null) * 100, 2) if len(non_null) > 0 else None
st.dataframe(pd.DataFrame.from_dict(uniqueness, orient="index", columns=["Uniqueness %"]))

# -----------------------------
# Accuracy - Manual rules
# -----------------------------
st.subheader("4️⃣ Accuracy Rules (Manual)")
manual_rules_input = {col: st.text_input(f"Rule for `{col}` (plain English)") for col in accuracy_cols}

# -----------------------------
# Accuracy rule engine
# -----------------------------
def apply_accuracy_rule(series, rule):
    t = rule.get("type")
    p = rule.get("parameters", {})

    if t == "not_null":
        return series.notna()
    if t == "unique":
        return ~series.duplicated(keep=False)
    if t == "regex":
        return series.astype(str).str.match(p.get("pattern", ""))
    series_num = pd.to_numeric(series, errors="coerce")
    if t == "range":
        return series_num.between(p.get("min"), p.get("max"), inclusive="both")
    if t == "min_value":
        return series_num >= p.get("value")
    if t == "starts_with":
        return series.astype(str).str.startswith(str(p.get("value")))
    return pd.Series([True] * len(series))

# -----------------------------
# Normalize rules
# -----------------------------
def normalize_rules(ai_rules):
    normalized = []
    # New schema: {"rules": [...]}
    if isinstance(ai_rules, dict) and "rules" in ai_rules:
        for r in ai_rules["rules"]:
            normalized.append({
                "column": r["column"],
                "type": r["type"],
                "parameters": {
                    "pattern": r.get("pattern"),
                    "min": r.get("min"),
                    "max": r.get("max"),
                    "value": r.get("value")
                }
            })
    # Old schema
    elif isinstance(ai_rules, dict):
        for col, rule in ai_rules.items():
            for t in rule.get("type", "").split("|"):
                normalized.append({
                    "column": col,
                    "type": t.strip(),
                    "parameters": rule.get("parameters", {})
                })
    return normalized

# -----------------------------
# AI Auto-Suggest Rules
# -----------------------------
st.subheader("5️⃣ AI Auto-Suggest Accuracy Rules")

if st.button("🤖 Generate AI Rules"):
    if not accuracy_cols:
        st.warning("Select columns for accuracy first")
    else:
        sample = df[accuracy_cols].head(3).to_dict(orient="list")
        prompt = f"""
Generate STRICT JSON accuracy rules for the following columns and sample data.

Allowed types: not_null | unique | range | min_value | starts_with | regex

Columns:
{accuracy_cols}

Sample:
{json.dumps(sample, indent=2)}

Output schema:
{{
  "rules": [
    {{
      "column": "<column>",
      "type": "<rule_type>",
      "pattern": "<regex_if_any>",
      "min": <min_if_any>,
      "max": <max_if_any>,
      "value": "<value_if_any>"
    }}
  ]
}}

Return only JSON, no markdown or explanations.
"""
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "Output STRICT JSON only"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            ai_rules = safe_json_loads(response.choices[0].message.content)
            st.session_state["ai_rules"] = ai_rules
            st.success("✅ AI rules generated")
            st.json(ai_rules)
        except Exception as e:
            st.error("❌ Failed to generate AI rules")
            st.exception(e)

# -----------------------------
# Run Full Data Quality
# -----------------------------
st.subheader("6️⃣ Run Full Data Quality Checks")

if st.button("🚀 Run Full DQ"):
    failed_rows = pd.DataFrame()
    accuracy_scores = {}

    # Completeness failures
    for col in completeness_cols:
        failed = df[df[col].isna()].copy()
        if not failed.empty:
            failed["Metric"] = "Completeness"
            failed["Failed_Column"] = col
            failed_rows = pd.concat([failed_rows, failed])

    # Validity failures
    for col in validity_cols:
        if validity.get(col) is None:
            continue
        mask = df[col].astype(str).str.match(r"^[\w\.-]+@[\w\.-]+\.\w+$") if df[col].dtype=="object" else pd.to_numeric(df[col], errors="coerce").notna()
        failed = df[~mask].copy()
        if not failed.empty:
            failed["Metric"] = "Validity"
            failed["Failed_Column"] = col
            failed_rows = pd.concat([failed_rows, failed])

    # Uniqueness failures
    for col in uniqueness_cols:
        mask = ~df[col].duplicated(keep=False)
        failed = df[~mask].copy()
        if not failed.empty:
            failed["Metric"] = "Uniqueness"
            failed["Failed_Column"] = col
            failed_rows = pd.concat([failed_rows, failed])

    # Accuracy rules
    all_rules = []

    manual_rules = {k:v for k,v in manual_rules_input.items() if v.strip()}
    if manual_rules:
        all_rules.extend(normalize_rules({k: {"type": v, "parameters": {}} for k,v in manual_rules.items()}))

    ai_rules = st.session_state.get("ai_rules", {})
    if ai_rules:
        all_rules.extend(normalize_rules(ai_rules))

    for rule in all_rules:
        col = rule["column"]
        if col not in df.columns:
            continue
        mask = apply_accuracy_rule(df[col], rule)
        accuracy_scores.setdefault(col, []).append(mask.mean()*100)
        failed = df[~mask].copy()
        if not failed.empty:
            failed["Metric"] = f"Accuracy ({rule['type']})"
            failed["Failed_Column"] = col
            failed_rows = pd.concat([failed_rows, failed])

    accuracy_scores = {k: round(sum(v)/len(v),2) for k,v in accuracy_scores.items()}
    st.subheader("Accuracy %")
    st.dataframe(pd.Series(accuracy_scores, name="Accuracy %"))

    # Overall DQ Score
    scores = []
    for col in set(completeness_cols + validity_cols + uniqueness_cols + accuracy_cols):
        vals = [v for v in [completeness.get(col), validity.get(col), uniqueness.get(col), accuracy_scores.get(col)] if v is not None]
        if vals:
            scores.append(sum(vals)/len(vals))
    dq_score = round(sum(scores)/len(scores),2) if scores else 0
    st.metric("Overall Data Quality Score", f"{dq_score} / 100")

    if not failed_rows.empty:
        st.subheader("❌ Failed Records")
        st.dataframe(failed_rows)
        st.download_button("⬇️ Download Failed Records", failed_rows.to_csv(index=False).encode(), "failed_records.csv", "text/csv")

    results_df = pd.DataFrame([{
        "Column": col,
        "Completeness %": completeness.get(col),
        "Validity %": validity.get(col),
        "Uniqueness %": uniqueness.get(col),
        "Accuracy %": accuracy_scores.get(col),
        "Checked At": datetime.now()
    } for col in set(completeness_cols + validity_cols + uniqueness_cols + accuracy_cols)])

    st.download_button("⬇️ Download DQ Results", results_df.to_csv(index=False).encode(), "dq_results.csv", "text/csv")
