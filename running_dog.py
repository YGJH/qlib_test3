import subprocess
cmd = [
    "uv",
    "run",
    "main.py"
]
while True:
    try:
        subprocess.run(cmd)
    except Exception as e:
        print(f"Error occurred: {e}")
# This script runs the main.py file using uv in a loop.
