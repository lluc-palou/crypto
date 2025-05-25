import os
import subprocess

# Directory structure
folders = {
    "data": "data",
    "market_data": os.path.join("data", "market_data"),
    "data_mi": os.path.join("data", "mutual_information"),
    "data_split": os.path.join("data", "split"),
    "models": "models",
    "architectures": "architectures",
    "logs": "logs",
    "logs_perf": os.path.join("logs", "performance"),
    "logs_val": os.path.join("logs", "performance", "validation"),
    "logs_test": os.path.join("logs", "performance", "test"),
}

# Directory structure initialization
print("Initializing directory structure...\n")
for label, path in folders.items():
    os.makedirs(path, exist_ok=True)
    print(f"    Created directory: {path}")
print("\nDirectory initialization complete.\n")

def run_script(script_name):
    """
    Automatically runs a script and manages its output and runtime errors.
    """
    print(f"Running script: {script_name}")
    result = subprocess.run(["python", script_name], capture_output=True, text=True)

    if result.stdout.strip():
        print(result.stdout.strip())

    if result.returncode != 0:
        print(f"\nError during script: {script_name}")
        print(result.stderr.strip())
        exit(1)
    
    print(f"\n{script_name} completed successfully.\n")

# Pipeline steps
pipeline_steps = [
    ("Step 1: Market data collection", "collect_market_data.py"),
    ("Step 2: Features derivation", "features.py"),
    ("Step 3: Targets derivation", "targets.py"),
    ("Step 4: Feature engineering", "feature_engineering.py"),
    ("Step 5: Model training and performance evaluation", "modeling.py"),
    ("Step 6: Model selecion", "model_selection.py"),
]

# Pipeline execution
for description, script in pipeline_steps:
    print(f"\n{description}")
    run_script(script)

# Final summary
print("All pipeline steps completed successfully.")
print("Trained models saved in: models/")
print("Performance logs saved in: logs/validation_performance/ and logs/test_performance/")