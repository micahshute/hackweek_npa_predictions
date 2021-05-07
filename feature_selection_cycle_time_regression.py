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

from xgboost import XGBRegressor
import random
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error as MAE,
    mean_squared_error as MSE,
    roc_curve,
    auc,
    classification_report,
)

import code
# Import dataset

data = pd.read_csv("./data/dataset9.csv")


target = 'cycle_time_from_er_confirm'
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
    'created_year',
    'difot'
  ],
  axis=1
).columns.to_list()
print(features)
preprocessors =  [
      # MonthCreator,
      # CycleTimeFromErConfirmProcessor,
      # TransformBooleanToNumericProcessor('has_run_payroll'),
      # TransformBooleanToNumericProcessor('taxable_as_scorp'),
      RemoveOutliersProcessor(target, 3, 90),
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
          'has_run_payroll',
          'taxable_as_scorp',
          'user_provided_industry',
          'industry_classification',
          'created_year',
          'difot'
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


possible_features = list(features)
random.shuffle(possible_features)
possible_features.insert(0, 'days_before_effective_date')
current_features = []
good_features = []
current_score = float('inf')
min_score = float('inf')

for i, feature in enumerate(possible_features):
  if feature == 'days_before_effective_date' and i != 0:
    continue
  current_features = list(good_features)
  current_features.append(feature)
  mean_abs_error_sum = 0
  print(f"Checking feature: {feature}...")
  for i in range(50):
    train_df, test_df = TrainTestSplit(0.8).process(cleaned_data)
    x_train = train_df[current_features]
    y_train = train_df[target]
    x_test = test_df[current_features]
    y_test = test_df[target]
    model = XGBRegressor(objective="reg:squarederror", max_depth=6, verbosity=0)
    model.fit(
        x_train,
        y_train,
        eval_metric="rmse",
        verbose=False
    )

    training_pred = model.predict(x_train[current_features])
    test_pred = model.predict(x_test[current_features])

    mae_train = MAE(y_train, training_pred)
    mae_test = MAE(y_test, test_pred)
    mean_abs_error_sum += mae_test
  
  mean_abs_error_avg = mean_abs_error_sum / 50
  print("mean test error: ", mean_abs_error_avg)
  print('for feature: ', feature)
  print('------')
  if mean_abs_error_avg < current_score:
    current_score = mean_abs_error_avg
    good_features = list(current_features)
    print('appending feature ', feature)



print(good_features)


model = XGBRegressor(objective="reg:squarederror", max_depth=6)
model.fit(
    train_df[good_features],
    train_df[target],
    eval_metric="rmse",
)

output_train = model.predict(train_df[good_features])
output = model.predict(test_df[good_features])

actual = list(test_df[target])
actual_train = list(train_df[target])


mae_train = MAE(train_df[target], output_train)
mae_test = MAE(test_df[target], output)
print("train mean absolute error: ", mae_train)
print("test mean absolute error: ", mae_test)

rmse_train = np.sqrt(MSE(train_df[target], output_train))
rmse_test = np.sqrt(MSE(test_df[target], output))

print("train root mean squared error: ", rmse_train)
print("test root mean squared error: ", rmse_test)


res = pd.DataFrame()
res[target] = actual
res['predicted_cycle_time'] = output
res.to_csv('./results_xgboost_lin_with_feature_selection.csv')

code.interact(local=dict(globals(), **locals()))


