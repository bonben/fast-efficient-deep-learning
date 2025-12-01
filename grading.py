import argparse
import numpy as np

def calculate_grade(params):
    # Constants
    TARGET_PARAMS = 80_000      # 20/20 Target
    BASE_PARAMS = 500_000     # 10/20 Baseline

    print(f"--- Grading Projection ---")
    print(f"Params: {params:,}")
    print(f"Note: This grade applies ONLY if Test Accuracy > 90%")

    # Calculate Logarithmic Score
    # We work in log space to reward orders of magnitude
    log_current = np.log(params)
    log_target = np.log(TARGET_PARAMS)
    log_base = np.log(BASE_PARAMS)

    # Ratio calculation:
    # 0.0 means we are at Target (100k) -> Grade 20
    # 1.0 means we are at Base (11M)   -> Grade 10
    ratio = (log_current - log_target) / (log_base - log_target)

    # Linear mapping on the log-ratio
    grade = 20 - (10 * ratio)

    # Clamp the grade between 0 and 20
    grade = max(0.0, min(20.0, grade))

    return grade

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the potential grade based on parameter count (assuming >90% accuracy).")

    parser.add_argument("params", type=int, help="Total number of parameters (e.g., 5000000)")

    args = parser.parse_args()

    final_grade = calculate_grade(args.params)

    print("-" * 30)
    print(f"🏆 Potential Grade: {final_grade:.2f} / 20")
    print("-" * 30)