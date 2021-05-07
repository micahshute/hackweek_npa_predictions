class CycleTimeFromErConfirmProcessor:

  @classmethod
  def process(cls, data):
    data[["cycle_time_from_er_confirm", "month"]].groupby("month").count()
    return data