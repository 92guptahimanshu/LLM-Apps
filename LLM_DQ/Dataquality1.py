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
    page_title="AI Data Quality Checker",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Data Quality Checker")
st.caption("Completeness • Validity • AI Rule Suggestions • Scorecard")

# -----------------------------------
# Groq API Key
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
# Utility: Safe JSON Loader
# -----------------------------------
def safe_json_loads(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON found in LLM response")

    return json.loads(text[start:end + 1])

# -----------------------------------
# COMPLETENESS CHECK
# -----------------------------------
st.subheader("1️⃣ Completeness Check")

completeness = {}
for col in df.columns:
    completeness[col] = round(
        (1 - df[col].isnull().mean()) * 100, 2
    )

st.dataframe(
    pd.DataFrame.from_dict(
        completeness,
        orient="index",
        columns=["Completeness %"]
    )
)

# -----------------------------------
# VALIDITY CHECK (Rule-based)
# -----------------------------------
st.subheader("2️⃣ Validity Check")

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
        validity,
        orient="index",
        columns=["Validity %"]
    )
)

# -----------------------------------
# AI RULE SUGGESTION
# -----------------------------------
st.subheader("3️⃣ Auto-Suggest Data Quality Rules (AI)")

if st.button("🤖 Generate Rules with AI"):

    sample = df.head(3).to_dict()

    prompt = f"""
You are a data quality expert.

Return ONLY valid JSON.
No explanations.
No markdown.

Format EXACTLY like this:

{{
  "column_name": {{
    "suggested_rule": "plain English rule",
    "confidence": 0-100
  }}
}}

Columns:
{list(df.columns)}

Sample rows:
{sample}
"""

    with st.spinner("AI is generating rules..."):
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw_output = response.choices[0].message.content

        try:
            rules = safe_json_loads(raw_output)
            st.success("✅ AI rules generated successfully")

            rules_df = pd.DataFrame.from_dict(
                rules, orient="index"
            )

            st.dataframe(rules_df)

        except Exception:
            st.error("❌ Failed to parse AI output")
            st.code(raw_output)
            st.stop()

# -----------------------------------
# DQ SCORECARD
# -----------------------------------
st.subheader("4️⃣ Data Quality Scorecard")

scores = []

for col in df.columns:
    c = completeness.get(col, 0)
    v = validity.get(col)

    if v is None:
        score = c
    else:
        score = round((c + v) / 2, 2)

    scores.append(score)

dq_score = round(sum(scores) / len(scores), 2)

st.metric(
    label="Overall Data Quality Score",
    value=f"{dq_score} / 100"
)

# -----------------------------------
# SAVE RESULTS
# -----------------------------------
st.subheader("5️⃣ Download Results")

results = pd.DataFrame({
    "Column": df.columns,
    "Completeness %": [completeness[c] for c in df.columns],
    "Validity %": [validity.get(c) for c in df.columns],
    "Checked At": datetime.now()
})

csv = results.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download DQ Results CSV",
    data=csv,
    file_name="dq_results.csv",
    mime="text/csv"
)
