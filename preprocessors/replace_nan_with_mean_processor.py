import math

class ReplaceNanWithMeanProcessor:

  def __init__(self, feature):
    self.feature = feature

  def process(self, data):
    mean_feature = self.feature_avg(data)
    val_is_nan = lambda val: math.isnan(val)
    data[self.feature] = [self.replace_if(val_is_nan, val, mean_feature, val) for val in data[self.feature]]
    return data

  def get_mean(self, data):
    try:
      return sum(data) / len(data)
    except:
      return 0

  def replace_if(self, pred, val, if_pred_val, if_not_pred_val):
    if pred(val):
        return if_pred_val
    else: 
        return if_not_pred_val

  def feature_avg(self, data):
    return self.get_mean([val for val in data[self.feature] if not math.isnan(val)])
