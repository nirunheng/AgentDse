from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # Beacon/agentdse
BEACON = ROOT.parent                               # Beacon
ATLAS = ROOT / "ATLAS-MICRO-2026-main"
ATLAS_LIB = ATLAS / "simulator" / "build" / "lib"

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
DRAM_DIR = DATA_DIR / "dram_cfgs"
ARTIFACT_DIR = DATA_DIR / "artifacts"
DB_PATH = DATA_DIR / "trials.db"
LOG_DIR = ROOT / "logs"

KEY_JSON_PATH = ROOT / "key.json"                 # 项目目录自带的配置
KEY_JSON_PATH_SHARED = BEACON / "key.json"        # Beacon 根目录的共享配置
