"""NkwaPay V2 interactive research demonstration.
Run from the project root: streamlit run deployment/app_v2.py
V2 trained artifacts are user-supplied; do not load untrusted joblib objects.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st
import sklearn
import joblib

from v2_feature_adapter import (
    EXPECTED_F1, RAW_HISTORY_FIELDS, RAW_APPLICATION_FIELDS,
    make_f1_features, raw_batch_to_f1, engineered_batch_to_f1, convert_kes_to_ghs,
)

APP_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = APP_DIR / 'model_artifacts_v2_final'
st.set_page_config(page_title='NkwaPay | Research Credit Assessment', page_icon='◆', layout='wide', initial_sidebar_state='expanded')
st.markdown('''<style>
:root{--navy:#10263d;--teal:#0b7774;--light:#f4f8f9}
.block-container{padding-top:2.0rem;max-width:1250px}
[data-testid="stSidebar"]{background:#10263d}
[data-testid="stSidebar"] *{color:#eef7f9}
[data-testid="stSidebar"] input{color:#182d42!important}
[data-testid="stMetric"]{background:#f4f8f9;border:1px solid #dfebe9;padding:14px;border-radius:12px}
.hero{background:linear-gradient(118deg,#10263d,#0f6067);padding:28px 30px;border-radius:16px;color:#f6ffff;margin-bottom:16px}
.hero h2{color:#ffffff;margin:0;font-size:1.6rem}
.hero p{color:#cbe3e4;margin:9px 0 0}
.smallnote{color:#617585;font-size:.87rem}
</style>''',unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading trained V2 pipelines…')
def load_engines():
    metadata_file=ARTIFACT_DIR/'model_metadata_v2.json'
    if not metadata_file.is_file():
        raise FileNotFoundError(f'Missing {metadata_file}')
    meta=json.loads(metadata_file.read_text(encoding='utf-8'))
    features=meta.get('features',[])
    if features!=EXPECTED_F1:
        raise ValueError('This interface expects the exact 26-feature F1 contract in engineering V2; metadata does not match.')
    if meta.get('feature_set')!='F1_borrowing_primary':
        raise ValueError('Expected feature_set=F1_borrowing_primary.')
    if str(meta.get('sklearn_version'))!=sklearn.__version__:
        raise RuntimeError(f"scikit-learn mismatch: model uses {meta.get('sklearn_version')}; app uses {sklearn.__version__}. Activate the training environment.")
    clf_file=ARTIFACT_DIR/'default_risk_pipeline_v2.joblib'
    reg_file=ARTIFACT_DIR/'historical_amount_pipeline_v2.joblib'
    if not clf_file.is_file() or not reg_file.is_file():
        raise FileNotFoundError(f'Missing .joblib files in {ARTIFACT_DIR}. Copy the files exported by Notebook V2-03.')
    classifier=joblib.load(clf_file)
    regressor=joblib.load(reg_file)
    if not hasattr(classifier,'predict_proba') or not hasattr(regressor,'predict'):
        raise ValueError('Artifacts do not expose expected inference methods.')
    return meta,classifier,regressor


def predict_features(X,meta,clf,reg,rate):
    features=meta['features']
    X=engineered_batch_to_f1(X,required_features=features)
    prob=clf.predict_proba(X)
    classes=list(clf.classes_)
    if 1 not in classes:raise ValueError('Classifier positive class 1 was not found.')
    p=np.asarray(prob[:,classes.index(1)],dtype=float)
    kes=np.asarray(reg.predict(X),dtype=float)
    if not (np.isfinite(p).all() and np.isfinite(kes).all() and ((0<=p)&(p<=1)).all()):
        raise ValueError('Model returned invalid predictions.')
    kes=np.maximum(0,kes)
    threshold=float(meta.get('classification_threshold',0.06))
    df=pd.DataFrame({
        'default_probability':p,
        'default_risk_percent':100*p,
        'flag_for_review':p>=threshold,
        'estimated_historical_amount_KES':kes,
    })
    if rate>0:
        df['estimated_historical_amount_GHS']=convert_kes_to_ghs(kes,rate)
    return df


def _show_result(results,rate,meta):
    r=results.iloc[0]
    left,middle,right=st.columns(3)
    left.metric('Estimated default probability',f"{float(r.default_probability):.2%}")
    middle.metric('Estimated historical disbursement',f"KES {float(r.estimated_historical_amount_KES):,.2f}")
    if rate>0:
        right.metric('Converted amount (display only)',f"GH₵ {float(r.estimated_historical_amount_GHS):,.2f}")
    else:
        right.metric('Review flag','Yes' if r.flag_for_review else 'No')
    if bool(r.flag_for_review):
        st.warning(f"Risk score crosses the provisional research threshold ({float(meta['classification_threshold']):.2f}). Flag for human review, not an automatic rejection.")
    else:
        st.success(f"Score below the provisional research threshold ({float(meta['classification_threshold']):.2f}). This is NOT a loan approval.")
    st.caption('The regression estimates disbursement amounts observed in historical Kenyan data. It is NOT an optimal loan recommendation or an affordability/credit-limit calculation.')


st.markdown('<div class="hero"><h2>◆ NkwaPay · Credit Assessment Studio</h2><p>Convert original loan application fields and borrowing history into V2 model predictions. Educational demo · No automated lending decisions.</p></div>',unsafe_allow_html=True)

with st.sidebar:
    st.markdown('### ◆ NkwaPay')
    st.caption('Research demonstration · Kenya-trained models')
    st.divider()
    st.markdown('#### Currency conversion')
    rate=st.number_input('GHS per 1 KES',min_value=0.0,value=0.0,step=0.001,format='%.6f',help='Example: if 1 KES equals 0.08 GHS, enter 0.08. Enter a rate you have verified; there is no live FX feed.')
    fx_source=st.text_input('Rate source / date (optional)',placeholder='Bank quote, as-of date')
    if rate==0:
        st.info('Enter an FX rate to enable GHS display. The model still predicts in KES.')
    else:
        st.caption(f'Conversion: 1 KES × {rate:.6f} = {rate:.6f} GHS. Applied after regression only.')
    st.divider()
    st.caption('Run locally with your saved V2-03 models. The default-risk model is not calibrated for lending in Ghana.')

try:
    meta,clf,reg=load_engines()
    live=True
except Exception as e:
    live=False
    meta,clf,reg=None,None,None
    st.error('The V2 models are not available in this environment. You can inspect the interface, but predictions are disabled.')
    st.code(str(e))
    st.info('Place your trained files in deployment/model_artifacts_v2_final/ and use the same Python/scikit-learn environment as training. See README.md.')

if live:
    st.caption(f"Model: {meta['classification_model']} (default risk) · {meta['regression_model']} (amount) · 26 V2 F1 features · review threshold {float(meta['classification_threshold']):.2f}")

tab_manual,tab_batch,tab_contract=st.tabs(['✍️ New application','📂 Demo CSV / batch scoring','🛡️ Model contract & limitations'])

with tab_manual:
    st.subheader('New application')
    st.write('Enter the fields recorded in the original loan dataset. We construct borrower-history features automatically; no need to enter 26 engineered variables.')
    st.info('For repeat borrowers, upload original-format historical loans or enter their earlier loans below. If complete history is unavailable, scores may be unreliable.')
    st.markdown('**Optional: original loan-history file**')
    history_upload=st.file_uploader('Past loan ledger CSV (customer_id, disbursement_date, Total_Amount, duration)',type='csv',key='manual_history')
    history_csv=pd.DataFrame(columns=RAW_HISTORY_FIELDS)
    if history_upload is not None:
        try:
            history_csv=pd.read_csv(history_upload,dtype={'customer_id':'string'},low_memory=False)
            missing=[c for c in RAW_HISTORY_FIELDS if c not in history_csv]
            if missing:raise ValueError(f'Missing columns: {missing}')
            st.caption(f'Loaded {len(history_csv):,} candidate historical rows; only the applicant’s earlier-dated loans are used.')
        except Exception as e:
            st.error(f'History file problem: {e}')
            history_csv=pd.DataFrame(columns=RAW_HISTORY_FIELDS)

    with st.form('application',clear_on_submit=False):
        col1,col2=st.columns(2)
        with col1:
            cid=st.text_input('Customer ID',value='100001',help='Used only to match previous loans. Never passed directly to the model.')
            loan_type=st.text_input('Loan type',value='',placeholder='Enter a loan_type value from train.csv',help='Use exactly the product category found in your training dataset.')
        with col2:
            app_date=st.date_input('Application / expected disbursement date',value=date.today(),help='The V2 model uses disbursement_date as its date proxy. Historical date ranges are most comparable to training.')
            duration=st.number_input('Loan duration (days)',min_value=1,value=30,step=1)
        st.markdown('**Optional: manually enter previous loans**')
        st.caption('Original dataset columns only. These rows will be combined with the history file; do not enter the same loan twice. Current or same-day loan amounts are not used.')
        blank=pd.DataFrame([{'disbursement_date':None,'Total_Amount':None,'duration':None}])
        prior_edit=st.data_editor(blank,num_rows='dynamic',hide_index=True,use_container_width=True,key='prior_edit',column_config={
            'disbursement_date':st.column_config.TextColumn('Previous disbursement_date (YYYY-MM-DD)'),
            'Total_Amount':st.column_config.NumberColumn('Previous Total_Amount (KES)',min_value=0.01,format='%.2f'),
            'duration':st.column_config.NumberColumn('Previous duration (days)',min_value=1),
        })
        submit=st.form_submit_button('Assess application',type='primary',disabled=not live)

    if submit and live:
        try:
            ledger=prior_edit.copy()
            ledger=ledger.dropna(how='all')
            if len(ledger):
                if ledger['disbursement_date'].isna().any():
                    raise ValueError('Every manually entered previous loan needs a disbursement_date.')
                ledger['customer_id']=str(cid).strip()
                ledger=ledger[RAW_HISTORY_FIELDS]
            else:
                ledger=pd.DataFrame(columns=RAW_HISTORY_FIELDS)
            hist=pd.concat([history_csv[RAW_HISTORY_FIELDS],ledger],ignore_index=True)
            X=make_f1_features(customer_id=cid,disbursement_date=app_date,loan_type=loan_type,duration=duration,
                               history=hist,required_features=meta['features'])
            results=predict_features(X,meta,clf,reg,rate)
            _show_result(results,rate,meta)
            with st.expander('Show generated V2 input features'):
                st.dataframe(X.T.rename(columns={0:'Value'}),use_container_width=True)
            download=results.copy()
            download.insert(0,'customer_id',str(cid).strip())
            download.insert(1,'disbursement_date',str(app_date))
            st.download_button('Download assessment CSV',download.to_csv(index=False).encode(),file_name='nkwapay_manual_assessment.csv',mime='text/csv')
            if not len(hist):
                st.warning('Cold-start example: no prior borrowing records supplied. This is appropriate only if this borrower has no recorded prior loans.')
        except Exception as e:
            st.error(f'Assessment could not be completed: {e}')

with tab_batch:
    st.subheader('Demo dataset scoring')
    st.write('Load loan rows from the original dataset, such as train.csv or a test CSV, or upload already engineered F1 inputs.')
    mode=st.radio('Input format',('Original-format raw loan records','Engineered V2 F1 feature records'),horizontal=True)
    if mode=='Original-format raw loan records':
        st.caption('Raw files need customer_id, disbursement_date, loan_type and duration. If Total_Amount exists, it is used ONLY as history for loans on later calendar dates, never as that row’s current model input.')
        include_batch=st.checkbox('Use strictly earlier loans in this uploaded dataset as history',value=True)
        ref_upload=st.file_uploader('Optional additional historical loan ledger CSV',type='csv',key='batch_ref')
    else:
        ref_upload=None;include_batch=False
        st.caption('This mode accepts the 26 already engineered predictors exported by Feature Engineering V2.')
    uploaded=st.file_uploader('Upload CSV for demo scoring',type='csv',key='batch_data')
    st.download_button('Download original-format input template',pd.DataFrame([
        {'customer_id':'100001','disbursement_date':'2024-01-15','loan_type':'REPLACE_WITH_TRAINING_LOAN_TYPE','duration':30,'Total_Amount':12000},
        {'customer_id':'100001','disbursement_date':'2024-04-15','loan_type':'REPLACE_WITH_TRAINING_LOAN_TYPE','duration':45,'Total_Amount':18000},
    ]).to_csv(index=False).encode(),'nkwapay_original_field_template.csv','text/csv')
    if uploaded is not None:
        try:
            raw=pd.read_csv(uploaded,dtype={'customer_id':'string','ID':'string','tbl_loan_id':'string','loan_type':'string'},low_memory=False)
            if len(raw)==0:raise ValueError('The CSV is empty.')
            if len(raw)>2000:raise ValueError('Upload at most 2,000 applications per run for this local demo.')
            st.write(f'Loaded **{len(raw):,} records**')
            st.dataframe(raw.head(8),hide_index=True,use_container_width=True)
            if st.button('Score uploaded applications',type='primary',disabled=not live):
                if mode=='Original-format raw loan records':
                    historical=None
                    if ref_upload is not None:
                        historical=pd.read_csv(ref_upload,dtype={'customer_id':'string'},low_memory=False)
                        missing=[c for c in RAW_HISTORY_FIELDS if c not in historical]
                        if missing:raise ValueError(f'History CSV missing: {missing}')
                    X,errs=raw_batch_to_f1(raw,history=historical,required_features=meta['features'],use_batch_as_history=include_batch)
                    valid=errs.eq('')
                    if (~valid).any():st.warning(f'{int((~valid).sum())} rows could not be engineered and will be excluded from prediction. Their input_error appears in the results.')
                    result=pd.DataFrame(index=raw.index,columns=['default_probability','default_risk_percent','flag_for_review','estimated_historical_amount_KES']+(['estimated_historical_amount_GHS'] if rate>0 else []))
                    if valid.any():
                        scored=predict_features(X.loc[valid].reset_index(drop=True),meta,clf,reg,rate)
                        result.loc[valid.values,scored.columns]=scored.to_numpy()
                    result['input_error']=errs.to_numpy()
                else:
                    X=engineered_batch_to_f1(raw,required_features=meta['features'])
                    result=predict_features(X,meta,clf,reg,rate)
                for idcol in ('ID','customer_id','tbl_loan_id','disbursement_date'):
                    if idcol in raw:result.insert(len([c for c in result if c in ('ID','customer_id','tbl_loan_id','disbursement_date')]),idcol,raw[idcol].to_numpy())
                st.session_state['batch_result']=result
                st.session_state['batch_original']=uploaded.name
        except Exception as e:
            st.error(f'CSV problem: {e}')
    if 'batch_result' in st.session_state:
        r=st.session_state['batch_result']
        st.markdown('#### Scoring results')
        valid_r=pd.to_numeric(r.default_probability,errors='coerce').notna()
        a,b,c=st.columns(3)
        a.metric('Successfully scored',f'{int(valid_r.sum()):,}')
        b.metric('Flagged for review',f'{int(r.loc[valid_r,"flag_for_review"].astype(bool).sum()):,}')
        c.metric('Mean estimated default risk',f'{pd.to_numeric(r.loc[valid_r,"default_probability"]).mean():.2%}' if valid_r.any() else 'N/A')
        st.dataframe(r,hide_index=True,use_container_width=True)
        st.download_button('Download all demo predictions',r.to_csv(index=False).encode(),file_name='nkwapay_v2_demo_scored.csv',mime='text/csv')
        st.caption('Historical amount in GHS is simply the predicted KES amount multiplied by your entered FX rate. No currency conversion enters model features.')

with tab_contract:
    st.subheader('What the model actually receives')
    st.write('The form collects only columns present in the original loan dataset. The adapter calculates the 26 F1 predictors with the same prior-loan chronological rules used in the V2 engineering notebook.')
    with st.expander('Show the 26 features',expanded=True):
        st.dataframe(pd.DataFrame({'feature':EXPECTED_F1,'source':['Original dataset' if x in ('loan_type','duration') else 'Derived from disbursement_date' if x.startswith('disbursement_') else 'Derived from earlier loans' for x in EXPECTED_F1]}),hide_index=True,use_container_width=True)
    st.markdown('**Data fields excluded from prediction:** ID, customer_id, tbl_loan_id, lender_id, country_id, target, current Total_Amount, repayment outcomes, and post-decision values.')
    st.warning('The models were trained on Kenyan loan data and had weak chronological holdout performance. Converted GHS figures are display-only, not evidence that the regression model generalises to Ghana. This is research support only; do not use for real approvals.')
    st.write('**History coverage:** If only a partial CSV is available, missing earlier loans cannot be recovered. Scores may be affected because training features were built using a much more complete original loan history.')
    if live:
        st.json({'classifier':meta['classification_model'],'regressor':meta['regression_model'],
                 'threshold':meta['classification_threshold'],'feature_set':meta['feature_set'],
                 'currency_of_model':'KES','sklearn_version':meta['sklearn_version']})
    st.caption('Load only joblib files from your own training run; pickle-based model files are not safe from untrusted sources.')
