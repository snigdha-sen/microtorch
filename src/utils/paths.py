from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIMULATION_PATH = PROJECT_ROOT.parent / "simulation_data" 
GRAD_PATH = PROJECT_ROOT.parent / "simulation_data" / "grad"
SIMULATION_DATA_PATH = SIMULATION_PATH / "data"
MODELS_PATH = PROJECT_ROOT / "models"