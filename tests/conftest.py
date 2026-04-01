import os
import pytest
import pandas as pd

ASPIRIN_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
IBUPROFEN_SMILES = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


@pytest.fixture(scope="session")
def api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        pytest.skip("OPENROUTER_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def model_id():
    return "google/gemini-3.1-flash-lite-preview"


@pytest.fixture(scope="session")
def sample_datasets():
    """Small realistic dataset for agent testing."""
    df = pd.DataFrame({
        "Compound": [
            "Aspirin", "Caffeine", "Ibuprofen", "Naproxen", "Acetaminophen",
            "Diclofenac", "Celecoxib", "Indomethacin", "Piroxicam", "Meloxicam",
        ],
        "Structure": [
            ASPIRIN_SMILES,
            CAFFEINE_SMILES,
            IBUPROFEN_SMILES,
            "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
            "CC(=O)Nc1ccc(O)cc1",
            "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
            "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1",
            "COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
            "OC(=O)C1=C(O)N2C(=O)c3ccccc3Nc3ccccc3S2(=O)=O",
            "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
        ],
        "KD_M": [3.2e-6, 1.5e-5, 8.7e-7, 2.1e-6, 5.0e-5,
                 4.3e-7, 1.2e-7, 6.5e-7, 3.8e-6, 9.1e-7],
        "AI_Binding_Score": [0.72, 0.45, 0.88, 0.65, 0.31,
                             0.91, 0.95, 0.85, 0.58, 0.82],
    })
    return {"Test_Assay": df}


@pytest.fixture(scope="session")
def system_prompt(sample_datasets):
    from app.services.chat_service import build_system_prompt
    return build_system_prompt(sample_datasets)
