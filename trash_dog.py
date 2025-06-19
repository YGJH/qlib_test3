import subprocess
import pandas as pd
import shutil
from pathlib import Path

# Clean up old source data to avoid processing empty files
source_dir = Path("scripts/data_collector/yahoo/source")
normalize_dir = Path("scripts/data_collector/yahoo/normalize")

print("Cleaning up old source and normalize directories...")
if source_dir.exists():
    shutil.rmtree(source_dir)
    print(f"Removed {source_dir}")

if normalize_dir.exists():
    shutil.rmtree(normalize_dir)
    print(f"Removed {normalize_dir}")

# Create directories
source_dir.mkdir(parents=True, exist_ok=True)
normalize_dir.mkdir(parents=True, exist_ok=True)

cmd = [
    "uv",
    "run",
    "scripts/data_collector/yahoo/collector.py",
    "--region",
    "US",
    "update_data_to_bin",
    "--qlib_data_1d_dir",
    ".qlib/qlib_data/us_data",
    "--end_date",
    (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    "--delay",
    "1",
    "--exists_skip",
    "False",
]
print("Running command:", " ".join(cmd))

while True:
    try:
        subprocess.run(cmd, check=True)
        print("nice good dog.")
        break
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print("Cleaning up and retrying...")
        # Clean again before retry
        if source_dir.exists():
            shutil.rmtree(source_dir)
        if normalize_dir.exists():
            shutil.rmtree(normalize_dir)
        source_dir.mkdir(parents=True, exist_ok=True)
        normalize_dir.mkdir(parents=True, exist_ok=True)