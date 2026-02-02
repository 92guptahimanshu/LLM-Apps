import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
from groq import Groq

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="AI Data Quality Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Data Quality Engine")
st.caption(
    "Completeness • Validity • Accuracy (AI) • Rule Suggestions • Scorecard"
)

# -----------------------------------
# Sidebar – Groq Key
# -----------------------------------
with st.sidebar:
    groq_api_key = st.text_input(
        "🔑 Enter Groq API Key",
        type="password"
    )

if not groq_api_key:
    st.warning("Please enter Groq API key to continue.")
    st.stop()

client = Groq(api_key=groq_api_key)

# -----------------------------------
# Upload CSV
# -----------------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if not uploaded_file:
    st.info("Upload a CSV file to start.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -----------------------------------
# Utility – Safe JSON Loader
# -----------------------------------
def safe_json_loads(text: str):
    def safe_json_loads(text: str):
        """
    Extracts the FIRST valid JSON object from an LLM response.
    Handles extra text, explanations, markdown, etc.
        """
    try:
        # Remove markdown fences
        text = text.replace("```json", "").replace("```", "").strip()

        # Find first JSON object
        start = text.find("{")
        if start == -1:
            raise ValueError("No JSON object found")

        bracket_count = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                bracket_count += 1
            elif text[i] == "}":
                bracket_count -= 1
                if bracket_count == 0:
                    json_str = text[start:i+1]
                    return json.loads(json_str)

        raise ValueError("Unbalanced JSON braces")

    except Exception as e:
        raise ValueError(f"Failed to parse JSON safely: {e}")


# -----------------------------------
# 1️⃣ COMPLETENESS
# -----------------------------------
st.subheader("1️⃣ Completeness")

completeness = {
    col: round((1 - df[col].isnull().mean()) * 100, 2)
    for col in df.columns
}

st.dataframe(
    pd.DataFrame.from_dict(
        completeness, orient="index", columns=["Completeness %"]
    )
)

# -----------------------------------
# 2️⃣ VALIDITY (Rule-based)
# -----------------------------------
st.subheader("2️⃣ Validity")

def email_valid(series):
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return series.astype(str).str.match(pattern).mean() * 100

def numeric_valid(series):
    return pd.to_numeric(series, errors="coerce").notna().mean() * 100

validity = {}

for col in df.columns:
    if df[col].dtype == "object" and df[col].astype(str).str.contains("@").any():
        validity[col] = round(email_valid(df[col]), 2)
    elif df[col].dtype != "object":
        validity[col] = round(numeric_valid(df[col]), 2)
    else:
        validity[col] = None

st.dataframe(
    pd.DataFrame.from_dict(
        validity, orient="index", columns=["Validity %"]
    )
)

# -----------------------------------
# 3️⃣ ACCURACY – English Rules
# -----------------------------------
st.subheader("3️⃣ Accuracy (Plain-English Rules)")

accuracy_rules = {
    col: st.text_input(f"Accuracy rule for `{col}` (optional)")
    for col in df.columns
}

def parse_accuracy_rules(rules):
    prompt = f"""
You are a JSON generator.

CRITICAL RULES:
- Output ONLY valid JSON
- No explanations
- No text before or after JSON
- No markdown
- No comments

Supported rule types:
- range → parameters: min, max
- min_value → parameters: value
- not_null → parameters: none
- regex → parameters: pattern
- starts_with → parameters: value

Input rules:
{json.dumps(rules, indent=2)}

Return format EXACTLY:
{{
  "column_name": {{
    "type": "range|min_value|not_null|regex|starts_with",
    "parameters": {{}}
  }}
}}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return safe_json_loads(response.choices[0].message.content)

def apply_accuracy_rule(series, rule):
    t = rule["type"]
    p = rule.get("parameters", {})

    if t == "range":
        return series.between(p["min"], p["max"], inclusive="both")
    if t == "min_value":
        return series >= p["value"]
    if t == "not_null":
        return series.notna()
    if t == "starts_with":
        return series.astype(str).str.startswith(p["value"])
    if t == "regex":
        return series.astype(str).str.match(p["pattern"])

    return pd.Series([True] * len(series))

accuracy_scores = {}
failed_rows = pd.DataFrame()

# -----------------------------------
# 4️⃣ AI Rule Suggestion
# -----------------------------------
st.subheader("4️⃣ Auto-Suggest DQ Rules (AI)")

if st.button("🤖 Generate Suggested Rules"):
    sample = df.head(3).to_dict()

    prompt = f"""
You are a data quality expert.
Return ONLY valid JSON.

Format:
{{
  "column": {{
    "suggested_rule": "plain English rule",
    "confidence": 0-100
  }}
}}

Columns: {list(df.columns)}
Sample: {sample}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    try:
        rules = safe_json_loads(response.choices[0].message.content)
        st.dataframe(pd.DataFrame.from_dict(rules, orient="index"))
    except Exception:
        st.error("Failed to parse AI output")
        st.code(response.choices[0].message.content)

# -----------------------------------
# RUN ACCURACY CHECK
# -----------------------------------
if st.button("🚀 Run Full Data Quality Checks"):

    with st.spinner("Running accuracy checks..."):
       try:
         parsed_rules = parse_accuracy_rules(
        {k: v for k, v in accuracy_rules.items() if v}
    )
       except Exception as e:
            st.error("❌ AI failed to generate valid rules.")
            st.exception(e)
            st.stop()

       for col, rule in parsed_rules.items():
            mask = apply_accuracy_rule(df[col], rule)
            accuracy_scores[col] = round(mask.mean() * 100, 2)

            failed = df.loc[~mask].copy()
            if not failed.empty:
                failed["Failed_Column"] = col
                failed_rows = pd.concat([failed_rows, failed])

    st.subheader("Accuracy %")
    st.dataframe(
        pd.Series(accuracy_scores, name="Accuracy %")
    )

# -----------------------------------
# 5️⃣ DQ SCORECARD
# -----------------------------------
st.subheader("5️⃣ Data Quality Scorecard")

scores = []

for col in df.columns:
    c = completeness.get(col, 0)
    v = validity.get(col)
    a = accuracy_scores.get(col)

    vals = [x for x in [c, v, a] if x is not None]
    scores.append(sum(vals) / len(vals))

dq_score = round(sum(scores) / len(scores), 2)

st.metric("Overall DQ Score", f"{dq_score} / 100")

# -----------------------------------
# 6️⃣ Failed Rows Download
# -----------------------------------
if not failed_rows.empty:
    st.subheader("❌ Failed Records")
    st.dataframe(failed_rows)

    st.download_button(
        "⬇️ Download Failed Rows",
        failed_rows.to_csv(index=False).encode("utf-8"),
        "failed_records.csv",
        "text/csv"
    )

# -----------------------------------
# 7️⃣ Download DQ Results
# -----------------------------------
results = pd.DataFrame({
    "Column": df.columns,
    "Completeness %": [completeness[c] for c in df.columns],
    "Validity %": [validity.get(c) for c in df.columns],
    "Accuracy %": [accuracy_scores.get(c) for c in df.columns],
    "Checked At": datetime.now()
})

st.download_button(
    "⬇️ Download DQ Results CSV",
    results.to_csv(index=False).encode("utf-8"),
    "dq_results.csv",
    "text/csv"
)
