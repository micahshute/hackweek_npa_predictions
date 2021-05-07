class RenameColumnsProcessor:


  def __init__(self, name_dict):
    self.name_dict = name_dict

  def process(self, data):
    data = data.rename(
      columns=self.name_dict
    )
    return data