import pandas as pd
import numpy as np

class GroupByAndRedefineByGroupMeanProcessor:

  def __init__(self, primary_feature, averaged_feature, recieving_feature):
    self.primary_feature = primary_feature
    self.averaged_feature = averaged_feature
    self.recieving_feature = recieving_feature

  def process(self, data):
    grouped_df = data[[self.primary_feature, self.averaged_feature]].groupby(self.primary_feature).mean().reset_index()
    grouped_df.columns = [self.primary_feature, self.recieving_feature]
    data = pd.merge(data, grouped_df, on=self.primary_feature, how="left")
    data[self.recieving_feature] = np.where(data[self.recieving_feature].isnull(), data[self.recieving_feature].mean(), data[self.recieving_feature])
    return data