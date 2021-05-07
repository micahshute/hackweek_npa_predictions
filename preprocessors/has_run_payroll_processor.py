import numpy as np

class HasRunPayrollProcessor:

  @classmethod
  def process(cls, data):
    data["has_run_payroll"] = np.where(data["has_run_payroll"] == "True", 1, 0)
    return data
  