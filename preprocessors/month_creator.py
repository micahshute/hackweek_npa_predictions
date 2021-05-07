import pandas as pd

class MonthCreator:

  @classmethod
  def process(cls, data):
    data["month"] = pd.to_datetime(data["create_date"]).dt.to_period("M")
    return data