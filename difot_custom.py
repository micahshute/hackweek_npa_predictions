import pandas as pd
import numpy as np
from preprocessors.preprocessor import Preprocessor
from preprocessors.cycle_time_from_er_confirm_processor import CycleTimeFromErConfirmProcessor
from preprocessors.drop_columns_processor import DropColumnsProcessor
from preprocessors.has_run_payroll_processor import HasRunPayrollProcessor
from preprocessors.month_creator import MonthCreator
from preprocessors.taxable_as_scorp_processor import TaxableAsScorpProcessor
from preprocessors.drop_na_processor import DropNaProcessor
from preprocessors.replace_nan_with_mean_processor import ReplaceNanWithMeanProcessor
from preprocessors.medical_tiers_processor import MedicalTiersProcessor
from preprocessors.rename_columns_processor import RenameColumnsProcessor
from preprocessors.replace_inf_processor import ReplaceInfProcessor
from preprocessors.remove_outliers_processor import RemoveOutliersProcessor
from preprocessors.group_by_and_redefine_by_group_mean_processor import GroupByAndRedefineByGroupMeanProcessor
from preprocessors.replace_null_with_mean_processor import ReplaceNullWithMeanProcessor
from preprocessors.add_nonlinear_feature_processor import AddNonlinearFeatureProcessor
from preprocessors.select_features_processor import SelectFeaturesProcessor
from preprocessors.normalize_processor import NormalizeProcessor
from preprocessors.add_column_processor import AddColumnProcessor
from preprocessors.preprocessor import Preprocessor
from preprocessors.transform_boolean_to_numeric_processor import TransformBooleanToNumericProcessor

from preprocessors.train_test_split import TrainTestSplit

from models.logistic_regression import LogisticRegression

import code
# Import dataset

data = pd.read_csv("./data/dataset9.csv")

target = 'difot'
target_data = data[target]
features = data.drop(
  [
    target,
    'npa_id',
    "create_date",
    'user_provided_industry',
    'industry_classification',
    'product_plan',
    'last_month_subscription_revenue',
    'number_of_eins',
    'number_of_employee_logins_last_month',
    'estimated_annual_income_per_employee_last_month',
    'time_to_onboard_in_days',
    'time_to_first_payroll_in_days',
    'cycle_time_from_benefit_order',
    'cycle_time_from_er_confirm',
    'has_run_payroll',
    'taxable_as_scorp',
    'user_provided_industry',
    'industry_classification',
    'created_year'
  ],
  axis=1
).columns.to_list()
print(features)
preprocessors =  [
      # MonthCreator,
      # CycleTimeFromErConfirmProcessor,
      # TransformBooleanToNumericProcessor('has_run_payroll'),
      # TransformBooleanToNumericProcessor('taxable_as_scorp'),
      DropColumnsProcessor(
        [
          'npa_id',
          "create_date",
          'user_provided_industry',
          'industry_classification',
          'product_plan',
          'last_month_subscription_revenue',
          'number_of_eins',
          'number_of_employee_logins_last_month',
          'estimated_annual_income_per_employee_last_month',
          'time_to_onboard_in_days',
          'time_to_first_payroll_in_days',
          'cycle_time_from_benefit_order',
          'cycle_time_from_er_confirm',
          'has_run_payroll',
          'taxable_as_scorp',
          'user_provided_industry',
          'industry_classification',
          'created_year'
        ]
      ),
      ReplaceInfProcessor,
      ReplaceNanWithMeanProcessor('ee_average_age'),
      ReplaceNanWithMeanProcessor('primary_payroll_admin_age'),
      MedicalTiersProcessor,
      RenameColumnsProcessor(
        {
          "state": "state_raw",
          "created_month": "created_month_raw",
          "sic_code": "sic_code_raw",
          "wc_status": "wc_status_raw",
          "tax_payer_type": "tax_payer_type_raw",
          "medical_carrier": "medical_carrier_raw",
          "dental_carrier": "dental_carrier_raw"
         }
      ),
      GroupByAndRedefineByGroupMeanProcessor(
        'medical_carrier_raw',
        target,
        'medical_carrier'
      ),
      GroupByAndRedefineByGroupMeanProcessor(
        'dental_carrier_raw',
        target,
        'dental_carrier'
      ),
      GroupByAndRedefineByGroupMeanProcessor(
        'state_raw',
        target,
        'state'
      ),
      GroupByAndRedefineByGroupMeanProcessor(
        'tax_payer_type_raw',
        target,
        'tax_payer_type'
      ),
      GroupByAndRedefineByGroupMeanProcessor(
        'created_month_raw',
        target,
        'created_month'
      ),
      GroupByAndRedefineByGroupMeanProcessor(
        'wc_status_raw',
        target,
        'wc_status'
      ),
      GroupByAndRedefineByGroupMeanProcessor(
        'sic_code_raw',
        target,
        'sic_code'
      ),
      SelectFeaturesProcessor(features),
      NormalizeProcessor,
      AddColumnProcessor(target, target_data)
    ]

preprocessor = Preprocessor(data, preprocessors)
cleaned_data = preprocessor.process()

train_df, test_df = TrainTestSplit(0.8).process(cleaned_data)

model = LogisticRegression(target, features)

training_matrix = train_df[features].values.T
target_matrix = train_df[target].values.T

beta = model.train_until(training_matrix, target_matrix, 100)
beta = model.beta
print(beta)


output = model.predict(test_df[features].values.T)
actual = list(test_df[target])
actual_train = list(train_df[target])
output_train = model.predict(train_df[features].values.T)
incorrect_count_test = 0
incorrect_count_train = 0
total_count = len(actual)
for i, aval in enumerate(actual):
  if output[i] < 0.5 and actual[i] == 1 or output[i] >= 0.5 and actual[i] == 0:
    # print(f"We thought the probability was {output[i]} but in reality the answer was {actual[i]}")
    incorrect_count_test += 1
  if output_train[i] < 0.5 and actual_train[i] == 1 or output_train[i] >= 0.5 and actual_train[i] == 0:
    incorrect_count_train += 1

print("TRAIN")
print(f"Incorrect count: {incorrect_count_train}")
print(f"Total count: {total_count}")
print(f"Fraction incorrect: {incorrect_count_train / total_count}")
print("TEST")
print(f"Incorrect count: {incorrect_count_test}")
print(f"Total count: {total_count}")
print(f"Fraction incorrect: {incorrect_count_test / total_count}")

res = pd.DataFrame()
res[target] = actual
res['predicted_difoc'] = output
res.to_csv('./results_custom_log.csv')

code.interact(local=dict(globals(), **locals()))

