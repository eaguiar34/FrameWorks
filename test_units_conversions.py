import numpy as np
from core.units import UnitSystem, convert_stress, convert_force, convert_length, KSI_TO_MPA, KIP_TO_N, IN_TO_MM

def test_ksi_to_mpa_factor():
    assert np.isclose(convert_stress(1.0, UnitSystem.KIP_IN_KSI, UnitSystem.N_MM_MPA), KSI_TO_MPA, atol=1e-12)

def test_kip_to_n():
    assert np.isclose(convert_force(1.0, UnitSystem.KIP_IN_KSI, UnitSystem.N_MM_MPA), KIP_TO_N, atol=1e-9)

def test_in_to_mm():
    assert np.isclose(convert_length(1.0, UnitSystem.KIP_IN_KSI, UnitSystem.N_MM_MPA), IN_TO_MM, atol=1e-12)
