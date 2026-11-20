import os,sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deployment'))
from v2_feature_adapter import EXPECTED_F1,make_f1_features,raw_batch_to_f1,engineered_batch_to_f1,convert_kes_to_ghs

class F1Tests(unittest.TestCase):
    def setUp(self):
        self.raw=pd.DataFrame([
            {'customer_id':'147','disbursement_date':'2024-01-01','Total_Amount':100,'duration':30,'loan_type':'A'},
            {'customer_id':'147','disbursement_date':'2024-02-01','Total_Amount':200,'duration':40,'loan_type':'A'},
            {'customer_id':'147','disbursement_date':'2024-02-01','Total_Amount':300,'duration':50,'loan_type':'A'},
            {'customer_id':'147','disbursement_date':'2024-03-01','Total_Amount':400,'duration':60,'loan_type':'A'},
            {'customer_id':'999','disbursement_date':'2024-02-01','Total_Amount':999999,'duration':900,'loan_type':'B'},
        ])
    def test_contract(self):
        self.assertEqual(len(EXPECTED_F1),26)
        self.assertEqual(len(set(EXPECTED_F1)),26)
    def test_cold_start_nan(self):
        f=make_f1_features(customer_id='147',disbursement_date='2024-01-01',loan_type='A',duration=30,history=self.raw).iloc[0]
        self.assertEqual(f.previous_loan_count,0)
        self.assertEqual(f.previous_valid_amount_count,0)
        self.assertTrue(np.isnan(f.previous_mean_loan_amount))
        self.assertEqual(f.previous_loans_last_90d,0)
    def test_ties_and_future_excluded(self):
        f=make_f1_features(customer_id='147',disbursement_date='2024-03-01',loan_type='A',duration=60,history=self.raw).iloc[0]
        self.assertEqual(f.previous_loan_count,3)
        self.assertEqual(f.previous_total_amount_borrowed,600)
        self.assertEqual(f.previous_mean_loan_amount,200)
        self.assertEqual(f.previous_min_loan_amount,100)
        self.assertEqual(f.previous_max_loan_amount,300)
        self.assertEqual(f.previous_mean_duration,40)
        self.assertEqual(f.previous_valid_amount_count,3)
        self.assertEqual(f.days_since_previous_loan,29)
        self.assertEqual(f.previous_loans_last_90d,3)
        self.assertEqual(f.previous_loans_last_180d,3)
        self.assertEqual(f.previous_mean_days_between_loans,31)
        self.assertEqual(f.previous_mean_amount_last_3,200)
        self.assertTrue(np.isnan(f.previous_most_recent_loan_amount))
        self.assertTrue(np.isnan(f.previous_amount_trend_last_3))
    def test_same_day_excluded(self):
        f=make_f1_features(customer_id='147',disbursement_date='2024-02-01',loan_type='A',duration=40,history=self.raw).iloc[0]
        self.assertEqual(f.previous_loan_count,1)
        self.assertEqual(f.previous_total_amount_borrowed,100)
    def test_batch_current_amount_is_not_feature(self):
        x,err=raw_batch_to_f1(self.raw,required_features=EXPECTED_F1)
        self.assertFalse(err.astype(bool).any())
        self.assertEqual(x.loc[0,'previous_loan_count'],0)
        self.assertEqual(x.loc[1,'previous_loan_count'],1)
        self.assertEqual(x.loc[3,'previous_loan_count'],3)
        self.assertEqual(x.loc[4,'previous_total_amount_borrowed']!=x.loc[4,'previous_total_amount_borrowed'],True)
    def test_missing_amounts(self):
        h=self.raw.copy();h.loc[0,'Total_Amount']=None
        f=make_f1_features(customer_id='147',disbursement_date='2024-02-01',loan_type='A',duration=40,history=h).iloc[0]
        self.assertEqual(f.previous_valid_amount_count,0)
        self.assertTrue(np.isnan(f.previous_total_amount_borrowed))
    def test_fx(self):
        np.testing.assert_array_almost_equal(convert_kes_to_ghs([100,200],0.08),[8.,16.])
        with self.assertRaises(ValueError):convert_kes_to_ghs([100],0)
    def test_wrong_contract(self):
        with self.assertRaises(ValueError):make_f1_features(customer_id='147',disbursement_date='2024-02-01',loan_type='A',duration=30,history=self.raw,required_features=['previous_default_rate'])
    def test_no_target_in_features(self):
        f=make_f1_features(customer_id='147',disbursement_date='2024-03-01',loan_type='A',duration=30,history=self.raw)
        self.assertNotIn('target',f.columns)
        self.assertNotIn('Total_Amount',f.columns)

if __name__=='__main__':unittest.main()
