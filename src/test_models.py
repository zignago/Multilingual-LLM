import subprocess
from config import SUPPORTED_LLM_MODELS

# Tests all models in SUPPORTED_LLM_MODELS
def run_test():
    # Parameters for the test run
    languages = ["italian", "german", "spanish"]
    limit = 4
    repeat = 1
    subset = 1

    # List to keep track of models that fail
    failed_models = []

    # Iterate through all models in SUPPORTED_LLM_MODELS
    for model_type, models in SUPPORTED_LLM_MODELS.items():
        for model in models:
            print(f"Testing model: {model}")
            try:
                # Construct the command to run main.py
                command = [
                    "python", "main.py",
                    "--languages", *languages,
                    "--limit", str(limit),
                    "--model", model,
                    "--repeat", str(repeat),
                    "--subset", str(subset)
                ]
                # Run the command and capture the output
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                # Check if the command executed successfully
                if result.returncode != 0:
                    print(f"Model {model} failed with error:\n{result.stderr}")
                    failed_models.append(model)
                else:
                    print(f"Model {model} completed successfully.")
            except Exception as e:
                # Catch any unexpected errors and add the model to the failed list
                print(f"An exception occurred while testing model {model}: {e}")
                failed_models.append(model)

    # Output the list of failed models
    if failed_models:
        print("\nThe following models encountered issues:")
        for model in failed_models:
            print(f"- {model}")
    else:
        print("\nAll models executed successfully.")

if __name__ == "__main__":
    run_test()
