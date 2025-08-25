import os
import time
import random
import schedule
import subprocess

def generate_directory_structure(folders):
    """
    Given a dictionary with the directory structure this function generates it. 
    """
    print("Generating directory structure...\n")

    for label, path in folders.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")

    print("\nDirectory structure generation completed.\n")

def run_script(script):
    """
    Given a script, runs it.
    """
    print(f"Executing script: {script}")
    try:
        subprocess.run(["python", script], capture_output=True, text=True)
        return True
    
    except subprocess.CalledProcessError:
        return False

def run_script_until_success(script, base_delay=30, cap_delay=600):
    """
    Given a script tries running it multiple times untill completed,
    this prevents possible internet connection issues.
    """
    attempt = 0

    while True:
        if run_script(script):
            return True

        attempt += 1
        delay = min(cap_delay, base_delay * (2 ** (attempt - 1)))
        time.sleep(random.uniform(delay / 2, delay))

def run_pipeline(pipeline):
    """
    Given a pipeline, runs it.
    """
    print("Executing pipeline...\n")

    for description, script in pipeline:
        print(f"\n{description}")
        run_script_until_success(script)

    print("Pipeline executed successfully.\n")

if __name__ == "__main__":
    # Directory structure declaration
    folders = {
        "market_data": "market_data",
        "derived_data": "derived_data",
        "architectures": "architectures",
        "models": "models",
        "logs": "logs",
    }
    
    generate_directory_structure(folders)

    # Pipeline declaration
    pipeline = [
        ("Step 1: Market data collection", "collect_market_data.py"),
        ("Step 2: Features derivation", "features.py"),
        ("Step 3: Targets derivation", "targets.py"),
        ("Step 4: Feature engineering", "feature_engineering.py"),
        ("Step 5: Features and targets split", "split.py"),
        ("Step 6: Prediction", "prediction.py"),
        ("Step 7: Trading", "trade.py")
    ]

    # Executes the pipeline daily (at 03:00 AM)
    schedule.every().day.at("03:00").do(run_pipeline, pipeline)

    print("Pipeline execution scheduled. Waiting for 3:00 AM...")
    while True:
        schedule.run_pending()
        time.sleep(30)