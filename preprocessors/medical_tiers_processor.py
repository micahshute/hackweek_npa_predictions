
class MedicalTiersProcessor:

  MEDICAL_TIERS = {
    'platinum': 0,
    'gold': 1,
    'silver': 2,
    'bronze': 3
  }

  @classmethod
  def process(cls, data):
    data['selected_medical_base_tier'] = [cls.get_medical_tier_numeric(tier) for tier in data['selected_medical_base_tier']]
    return data

  @classmethod
  def get_medical_tier_numeric(cls, tier):
    try:
      return medical_tiers[tier]
    except:
      return 4