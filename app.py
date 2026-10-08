import time, joblib, numpy as np, pandas as pd, streamlit as st
from google import genai

MODEL_NAME = "gemini-3.8-flash"   # change here if Google renames the model
MODELS = [MODEL_NAME, "gemini-flash-latest", "gemini-flash-lite-latest"]   # backups if the first is overloaded
st.set_page_config(page_title="Loan Pre-Screener", page_icon="🏦", layout="wide")

@st.cache_resource
def gem_cache():
    return {}

@st.cache_resource
def load():
    b = joblib.load("model.joblib")
    return b["model"], b["info"]

@st.cache_data
def load_scored():
    return pd.read_csv("test_scored.csv")

model, info = load()
FEATS = info["features"]
LABELS = {"loan_amnt": "Loan amount", "annual_inc": "Annual income", "dti": "Debt-to-income ratio",
          "fico_range_low": "Credit score", "delinq_2yrs": "Recent delinquencies",
          "inq_last_6mths": "Recent credit inquiries", "open_acc": "Open credit lines",
          "pub_rec": "Public records", "mort_acc": "Mortgage accounts", "emp_length": "Years employed",
          "credit_history_years": "Credit history length", "revol_util": "Card utilization",
          "term": "Loan term", "purpose": "Loan purpose", "home_ownership": "Home ownership"}

REF = dict(info["medians"])
REF["term"] = "36 months" if "36 months" in info["levels"]["term"] else info["levels"]["term"][0]
REF["purpose"] = "debt_consolidation" if "debt_consolidation" in info["levels"]["purpose"] else info["levels"]["purpose"][0]
REF["home_ownership"] = "MORTGAGE" if "MORTGAGE" in info["levels"]["home_ownership"] else info["levels"]["home_ownership"][0]

def predict(row):
    df = pd.DataFrame([row])[FEATS]
    for c in info["cat"]:
        df[c] = pd.Categorical(df[c], categories=info["levels"][c])
    return float(model.predict_proba(df)[0, 1])

def drivers(row):
    base = predict(row)
    out = []
    for f in FEATS:
        r = dict(row); r[f] = REF[f]
        out.append((f, base - predict(r)))
    return base, sorted(out, key=lambda x: -abs(x[1]))

def decide(p, lo, hi):
    return "APPROVE" if p < lo else ("REJECT" if p > hi else "REFER TO HUMAN")

def explain(decision, p, top):
    facts = "; ".join(f"{LABELS[f]} {'raises' if d > 0 else 'lowers'} risk by about {abs(d)*100:.1f} percentage points" for f, d in top)
    template = (f"The model estimates a {p*100:.0f}% chance of default, so the suggestion is: {decision}. "
                f"Main factors: {facts}.")
    prompt = ("You are a credit analyst assistant. Explain this loan pre-screening result in 3 short sentences, "
              "plain language, for the applicant. Use ONLY the facts given, do not invent numbers or advice, "
              "and say it is an automated estimate, not a final lending decision.\n"
              f"Suggestion: {decision}. Estimated default probability: {p*100:.0f}%. Factors: {facts}.")
    cache = gem_cache()
    if prompt in cache:                      # same case asked before: reuse the earlier Gemini answer
        return cache[prompt], "Gemini"
    last = None
    for attempt in range(2):                 # Gemini sometimes returns 503 "high demand": try other models, then retry
        for m in MODELS:
            try:
                client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                text = client.models.generate_content(model=m, contents=prompt).text
                cache[prompt] = text
                return text, "Gemini"
            except Exception as e:
                last = e
        time.sleep(2)
    return template, f"built-in template (AI unavailable: {str(last)[:100]})"

st.title("🏦 Loan Pre-Screener")
st.caption("Educational prototype trained on real LendingClub loans (US dollars). Not a real lending decision. "
           "Your inputs are not stored; the summary of factors (no personal data) may be sent to Google's Gemini API to write the explanation.")

with st.sidebar:
    st.subheader("Decision thresholds")
    lo = st.slider("Approve if default risk below", 0.05, 0.40, 0.15, 0.01)
    hi = st.slider("Reject if default risk above", 0.10, 0.60, 0.30, 0.01)
    st.caption("Between the two = refer to a human underwriter. (In a real product only underwriters could change these.)")
    if hi <= lo + 0.01:
        st.error("Reject threshold must be higher than the approve threshold.")
        st.stop()
    if st.button("Reset session"):
        st.session_state.clear()
        st.rerun()

tab1, tab2, tab3 = st.tabs(["Assess an applicant", "Model validation & fairness", "Limits"])

with tab1:
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        loan_amnt = c1.number_input("Loan amount ($)", value=10000, step=500)
        term = c1.selectbox("Term", info["levels"]["term"])
        PUR = info["levels"]["purpose"]
        purpose = c1.selectbox("Purpose", PUR, index=PUR.index("debt_consolidation") if "debt_consolidation" in PUR else 0)
        annual_inc = c2.number_input("Annual income ($)", value=60000, step=1000)
        emp_length = c2.number_input("Years employed (0-10)", value=3, step=1)
        HOME = [h for h in info["levels"]["home_ownership"] if h not in ("ANY", "NONE", "OTHER")]
        home = c2.selectbox("Home ownership", HOME, index=HOME.index("RENT") if "RENT" in HOME else 0)
        fico = c3.number_input("Credit score (FICO)", value=700, step=5)
        monthly_debt = c3.number_input("Monthly debt payments ($)", value=800, step=50)
        hist = c3.number_input("Credit history (years)", value=10.0, step=1.0)
        with st.expander("Advanced credit details (defaults = typical applicant)"):
            a1, a2, a3 = st.columns(3)
            delinq = a1.number_input("Delinquencies, last 2 yrs", value=0, step=1)
            inq = a1.number_input("Credit inquiries, last 6 months", value=0, step=1)
            open_acc = a2.number_input("Open credit lines", value=int(info["medians"]["open_acc"]), step=1)
            pub_rec = a2.number_input("Public records (bankruptcies etc.)", value=0, step=1)
            revol = a3.number_input("Card utilization (%)", value=float(round(info["medians"]["revol_util"], 0)), step=5.0)
            mort = a3.number_input("Mortgage accounts", value=int(info["medians"]["mort_acc"]), step=1)
        submitted = st.form_submit_button("Assess")

    if submitted:
        errs = []
        if annual_inc <= 0: errs.append("Income must be greater than zero.")
        if loan_amnt < 500: errs.append("Loan amount must be at least $500.")
        if not 0 <= emp_length <= 10: errs.append("Years employed must be between 0 and 10.")
        if not 300 <= fico <= 850: errs.append("Credit score must be between 300 and 850.")
        if monthly_debt < 0: errs.append("Monthly debt cannot be negative.")
        if not 0 <= hist <= 70: errs.append("Credit history must be between 0 and 70 years.")
        if not 0 <= revol <= 150: errs.append("Card utilization must be between 0 and 150%.")
        if min(delinq, inq, open_acc, pub_rec, mort) < 0: errs.append("Counts cannot be negative.")
        if errs:
            for e in errs: st.error(e)
            st.session_state.pop("result", None)
        else:
            row = dict(loan_amnt=loan_amnt, annual_inc=annual_inc, dti=monthly_debt * 12 / annual_inc * 100,
                       fico_range_low=fico, delinq_2yrs=delinq, inq_last_6mths=inq, open_acc=open_acc,
                       pub_rec=pub_rec, mort_acc=mort, emp_length=emp_length, credit_history_years=hist,
                       revol_util=revol, term=term, purpose=purpose, home_ownership=home)
            st.session_state.result = row

    row = st.session_state.get("result")
    if row:
        warns = [f"{LABELS[f]} ({row[f]:.0f}) is outside the range the model was trained on, so the result is less reliable."
                 for f in info["num"] if not info["ranges"][f][0] <= row[f] <= info["ranges"][f][1]]
        for w in warns: st.warning(w)
        p, drv = drivers(row)
        dec = decide(p, lo, hi)
        if warns and dec == "APPROVE":
            dec = "REFER TO HUMAN"
            st.info("Automatic approval blocked: some inputs are outside the range the model was trained on, so a human should review.")
        color = {"APPROVE": "green", "REJECT": "red"}.get(dec, "orange")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated default risk", f"{p*100:.1f}%")
        m2.metric("Avg default rate (2016 test loans)", f"{info['base_rate']*100:.1f}%")
        m3.markdown(f"### :{color}[{dec}]")
        top = drv[:3]
        text, src = explain(dec, p, top)
        st.write(text)
        st.caption(f"Explanation written by: {src}")
        st.subheader("What drives this score")
        st.dataframe(pd.DataFrame({"Factor": [LABELS[f] for f, _ in drv[:6]],
                                   "Effect on risk (points)": [round(d * 100, 1) for _, d in drv[:6]],
                                   "Direction": ["raises risk" if d > 0 else "lowers risk" for _, d in drv[:6]]}),
                     hide_index=True)
        st.caption("Effect = change in risk if this factor were set to a typical applicant's value.")
        with st.expander("Stability check: nearly identical applicant"):
            r2 = dict(row); r2["annual_inc"] = row["annual_inc"] + 1000; r2["fico_range_low"] = row["fico_range_low"] + 5
            p2 = predict(r2)
            d2 = decide(p2, lo, hi)
            if warns and d2 == "APPROVE":
                d2 = "REFER TO HUMAN"
            st.write(f"Original: {p*100:.1f}% → with +$1,000 income and +5 credit score: {p2*100:.1f}% "
                     f"({d2}). Small input changes should cause small output changes.")
        h = st.session_state.setdefault("history", [])
        if not h or h[-1][0] != row:
            h.append((row, p, dec))
        st.subheader("This session's assessments")
        st.dataframe(pd.DataFrame([{"Loan": r["loan_amnt"], "Income": r["annual_inc"], "FICO": r["fico_range_low"],
                                    "Risk %": round(pp * 100, 1), "Decision": d} for r, pp, d in h]), hide_index=True)

with tab2:
    st.write(f"**Test AUC: {info['auc']:.3f}** on 2016 loans the model never saw. "
             f"(0.5 = random guessing, 1.0 = perfect. Credit-risk models on application data typically land around 0.65-0.75.)")
    try:
        s = load_scored()
        s["decision"] = [decide(x, lo, hi) for x in s["pred"]]
        st.subheader("Do the decisions match reality?")
        g = s.groupby("decision").agg(loans=("target", "size"), predicted_risk=("pred", "mean"),
                                       actual_default_rate=("target", "mean")).round(3)
        st.dataframe(g)
        st.caption(f"Overall the model predicted {s['pred'].mean():.1%} default but {s['target'].mean():.1%} actually defaulted: "
                   "default rates rose after the training years, so absolute risk levels are optimistic while the ranking (low vs high risk) still works.")
        st.subheader("Fairness check: approval rate by group")
        s["approved"] = (s["decision"] == "APPROVE").astype(int)
        s["income_band"] = pd.qcut(s["annual_inc"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
        for col in ["home_ownership", "income_band"]:
            t = s.groupby(col, observed=True).agg(loans=("target", "size"), approval_rate=("approved", "mean"),
                                                  actual_default_rate=("target", "mean")).round(3)
            t = t[t["loans"] >= 200]
            st.write(f"**By {col.replace('_', ' ')}** (groups under 200 loans hidden)")
            st.dataframe(t)
        st.caption("Large gaps in approval rates between groups need investigation: they may reflect real risk differences, or bias in the data.")
    except FileNotFoundError:
        st.info("test_scored.csv not found. Run train.py first.")

with tab3:
    st.markdown("""
- Trained on US LendingClub loans (2012-2016), so it does not reflect other countries, lenders or time periods.
- Only applicants who were already approved by LendingClub appear in the data (selection bias).
- Cannot see income verification, employer quality, or events after application.
- Accuracy is moderate (see AUC): it supports a human underwriter and must not decide alone.
- Do not enter real personal data.
""")
