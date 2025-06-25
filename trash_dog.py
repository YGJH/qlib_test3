import subprocess
import pandas as pd
import shutil
from pathlib import Path
from colors import print_green, print_yellow, print_red, warn_with_color, Colors

# Clean up old source data to avoid processing empty files
source_dir = Path("scripts/data_collector/yahoo/source")
normalize_dir = Path("scripts/data_collector/yahoo/normalize")
QLIB_DIR = Path(".qlib/qlib_data/us_data")  
def clean_dir():
    return
    print_green("Cleaning up old source and normalize directories...")
    if source_dir.exists():
        shutil.rmtree(source_dir)
        print_green(f"Removed {source_dir}")

    if normalize_dir.exists():
        shutil.rmtree(normalize_dir)
        print_green(f"Removed {normalize_dir}")
        
    if QLIB_DIR.exists():
        shutil.rmtree(QLIB_DIR, ignore_errors=True)
        print_yellow(f"Force removed {QLIB_DIR}")

    # Create directories
    QLIB_DIR.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    normalize_dir.mkdir(parents=True, exist_ok=True)


# update_data_to_bin --qlib_data_1d_dir
# Fix the command - remove start_date and end_date parameters
cmd = [
    "uv",
    "run",
    "scripts/data_collector/yahoo/collector.py",
    "--region",
    "US",
    "update_data_to_bin",
    "--qlib_data_1d_dir",
    str(QLIB_DIR),
    "--end_date",
    (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
]

filter_cat = [
    "uv",
    "run",
    "filter_cat.py",
]
# uv run scripts/data_collector/yahoo/collector.py normalize_data_1d_extend --old_qlib_dir <QLIB_DIR> --source_dir <dir2> --normalize_dir <dir3> --region US --interval 1d
running_main = [
    "uv",
    "run",
    "main.py",
]

print("Running command:", " ".join(cmd))

while True:
    try:
        clean_dir()
        subprocess.run(cmd, check=True)
        print_green("Data collection completed successfully.")
        subprocess.run(filter_cat, check=True)
        print_green("Category filtering completed successfully.")
        # subprocess.run(running_main, check=True)
        # print_green("Main script executed successfully.")
        print_green("nice good dog.")
        break
    except subprocess.CalledProcessError as e:
        print_red(f"Error occurred: {e}")
        break