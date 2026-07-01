# NA-03 — Bias Detection in a Credit Scoring Model
### Interactive Streamlit Dashboard

## What's inside
- `app.py` — the Streamlit dashboard
- `requirements.txt` — Python dependencies
- `tz_credit_applicants_300000.csv` — dataset (place it in the same folder as `app.py`, or upload it via the sidebar in-app)

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## What the dashboard does
1. **Overview** — KPIs and applicant composition (occupation tier, region, gender, default balance), filterable via the sidebar.
2. **Exploratory Analysis** — distributions (age, income, loan amount), default patterns by gender/occupation/region, income-vs-loan scatter, correlation heatmap.
3. **Model Performance** — trains a Logistic Regression classifier (same features/pipeline as the source notebook) and reports accuracy, ROC AUC, confusion matrix, ROC curve, classification report, and standardized coefficients.
4. **Fairness Audit** — Fairlearn `MetricFrame` broken out by gender, region, or occupation tier (your choice in the sidebar): selection rate, TPR/FPR by group, Demographic Parity Difference, Disparate Impact Ratio (four-fifths rule), Equalized Odds Difference.
5. **Bias Mitigation** — implements **Reweighing** (Kamiran & Calders): re-trains the model with sample weights that balance each (sensitive-group × outcome) combination, then compares fairness metrics and accuracy before vs. after, plus a written discussion of the risk of excluding informal-sector workers from digital credit.
6. **Score a New Applicant** — a form for live user input (age, gender, region, occupation tier, income, loan amount) that returns a predicted default-probability gauge and compares the applicant's score against historical default rates for their gender/region/occupation peer groups.

## Notes
- The model is always trained on the **full** dataset for stability; the sidebar filters only affect the Overview and Exploratory Analysis tabs.
- Model training and the mitigation routine are cached (`st.cache_resource`), so they only run once per session/data change — subsequent tab switches and filter changes are fast.
