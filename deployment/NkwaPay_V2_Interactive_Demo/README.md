# NkwaPay V2 Interactive Demo — original-field input + Ghana cedi display

A local **Streamlit** demo that loads the exact two trained V2-03 F1 pipelines and computes 26 features **from original dataset fields**, rather than asking users to enter engineered columns.

## 1. Install

Unzip this package **into the project root containing your own V2 models**. The expected layout is:

```
ProSIT2/
  deployment/
    app_v2.py
    v2_feature_adapter.py
    requirements.txt
    model_artifacts_v2_final/
      default_risk_pipeline_v2.joblib
      historical_amount_pipeline_v2.joblib
      model_metadata_v2.json
```

The ZIP intentionally **does not include trained weights**. Do **not** use old V1 models: their feature contracts are incompatible. Only load .joblib files you created and trust.

Create/activate the **same Python environment** as the final V2 training notebook and install Streamlit (and pandas/numpy/joblib). Check the exact `sklearn_version` in `model_metadata_v2.json` and use the same version. For example:

```bash
python -m pip install streamlit pandas numpy joblib
streamlit run deployment/app_v2.py
```

If your sklearn version differs, activate the environment from V2-03 instead of installing a random version. `requirements.txt` is a reference, not a lock file.

## 2. New application

The form collects `customer_id`, `loan_type`, `disbursement_date` (application-date proxy), `duration`, and optional **prior** loans with original fields `disbursement_date`, `Total_Amount`, `duration`. Previous loans can come from a raw CSV with `customer_id`, or be entered manually. `customer_id` is only a history lookup key; it is not passed to the model. Histories must reflect actual earlier loans; no information about current repayment outcomes is used.

**Important:** Loan-type categories must match values used in the original train.csv. This model does not learn a different Ghanaian loan-product taxonomy.

## 3. CSV / batch demo

- Original-format CSV mode needs `customer_id`, `disbursement_date`, `loan_type`, `duration`. If the file contains `Total_Amount` and you enable *Use strictly earlier loans as history*, past rows can be used as prior recorded events for later rows. **Current and same-day loan amounts are never predictors.** You can also upload a historical-rows reference CSV. If you need truly unseen applications, use a separate historical ledger instead of treating application-file amounts as known.
- Engineered F1 mode accepts the exact 26 predictor columns exported from the V2 notebook, for a direct contract / demonstration smoke test.
- The raw batch demo limits the batch to 2,000 rows so manual scoring doesn't freeze a laptop. With partial history, the computed features will differ from the original trained feature-engineering outputs.
- The app exports predictions with available identifiers, review flags and any input errors. An optional sample template is included.

## 4. FX rule

In the sidebar, enter **GHS per 1 KES** (not KES per 1 GHS). Regression model estimates historical disbursement in KES; the display calculation is:

`GHS = max(0, predicted_KES) × (GHS_per_1_KES)`

This is a user-supplied conversion. No FX rate is fetched or assumed, and classification probabilities are not altered by exchange rates. The converted value is **not** a recommended or approved credit limit.

## 5. Caveats

- The models were fitted using Kenyan training records and showed poor chronological holdout generalization. **Do not make actual credit decisions based on these outputs.**
- The threshold is a development-selected **human review flag**, not a policy-calibrated approve/decline threshold.
- Loan application date is the same date proxy as historical `disbursement_date` in the training pipeline. A future date may lie outside training distribution.
- For histories with multiple prior loans on the most recent identical day, `previous_most_recent_loan_amount` deliberately stays NaN, matching the V2 pipeline. Unknown past borrowing records cannot be manufactured.
- If original customer IDs were corrupted by the source pipeline, the model training integrity issue remains; this app cannot repair it.
- The model pipeline preprocessors handle missing engineered values. No global preprocessing or model retraining occurs in the app.
- Uploaded customer information stays in your local Streamlit process; be careful if hosting publicly.

## 6. Development tests

Run `python -m unittest discover -s tests -v` from the extracted package root. These tests compare engineered F1 features against a separate reference implementation for chronology, amount history, and same-day tie behavior, and test FX conversion. These do not substitute for a parity test against **your actual** processed_v2 dataset.
