import os
import subprocess

# Directory structure
folders = {
    "derived_data": "derived_data",
    "market_data": "market_data",
    "models": "models",
    "architectures": "architectures",
    "logs": "logs",
}

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

# Cold start up pipeline steps
cold_start_pipeline = [
    ("Step 1: Market data collection", "collect_market_data.py"),
    ("Step 2: Features derivation", "features.py"),
    ("Step 3: Targets derivation", "targets.py"),
    ("Step 4: Feature engineering", "feature_engineering.py"),
    ("Step 5: Model training and performance evaluation", "modeling.py"),
    ("Step 6: Model selecion", "model_selection.py"),
]

# Cold start up pipeline steps
warm_start_pipeline = [
    ("Step 1: Market data collection", "collect_market_data.py"),
    ("Step 2: Features derivation", "features.py"),
    ("Step 3: Targets derivation", "targets.py"),
    ("Step 4: Feature engineering", "feature_engineering.py"),
    ("Step 5: Features and targets split", "split.py"),
    ("Step 6: Prediction", "prediction.py"),
]

# Asks user which pipeline to run
print("Select pipeline to run:")
print("  1. Cold start-up pipeline (data collection to model selection)")
print("  2. Warm start-up pipeline (data collection + prediction)")

choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    selected_pipeline = cold_start_pipeline
    
    # Directory structure initialization
    print("Initializing directory structure...\n")

    for label, path in folders.items():
        os.makedirs(path, exist_ok=True)
        print(f"    Created directory: {path}")

    print("\nDirectory initialization complete.\n")

elif choice == "2":
    selected_pipeline = warm_start_pipeline

else:
    print("Invalid input. Exiting.")
    exit(1)

# Runs selected pipeline
for description, script in selected_pipeline:
    print(f"\n{description}")
    run_script(script)

# Final message
print("Pipeline execution completed successfully.")