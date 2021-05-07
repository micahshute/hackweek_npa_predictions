from sklearn import preprocessing
import pandas as pd

class NormalizeProcessor:

  @classmethod
  def process(cls, data):
    features = data.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    current_data_scaled = min_max_scaler.fit_transform(data)
    new_data = pd.DataFrame(current_data_scaled)
    new_data.columns = features
    return new_data