#!/usr/bin/env python
"""
main.py

Runs the full machine learning pipeline for optimizing gasoline blend composition:
1. Synthetic Data Generation
2. Model Training
3. Single-Output Optimization
4. Multi-Output Optimization

The goal is to ensure that all steps (data creation, model building, and blend optimization)
run seamlessly in one place. Each step is handled in its own script, which we can run from here.
"""

import subprocess

def generate_synthetic_data():
    """
    Calls '01_generate_synthetic_blends.py' to create synthetic blend data.
    
    Why:
        - We want a controlled dataset reflecting tank volume constraints
          and blend specs (e.g., octane, RVP limits).
        - This step outputs 'blends_dataset.csv', which the subsequent
          scripts rely on for training and optimization.
    
    Expected Input:
        - A CSV file 'component_data_all.csv' with base component properties.
    Produces:
        - 'blends_dataset.csv', storing feasible and labeled synthetic blends.
    """
    subprocess.run(["python", "01_generate_synthetic_blends.py"], check=True)


def train_models():
    """
    Calls '02_model_training.py' to train:
        1) A classification model for in-spec/out-of-spec prediction.
        2) A single-output regression model for cost (on in-spec blends).
        3) A multi-output regression model for cost + margins.
    
    Why:
        - Classification helps quickly identify whether a blend meets specs.
        - Regression (single or multi-output) predicts cost and property margins
          for optimization.
    
    Expected Input:
        - 'blends_dataset.csv' from the synthetic data generation step.
    Produces:
        - 'model_inSpec_classifier.joblib'
        - 'model_cost_regressor.joblib'
        - 'model_costmargin_regressor.joblib'
    """
    subprocess.run(["python", "02_model_training.py"], check=True)


def optimize_blend_single_output():
    """
    Calls '03_optimize_blend.py' to run a single-objective optimization
    (typically cost minimization), subject to constraints.
    
    Why:
        - We aim to find a blend composition that meets spec constraints
          while minimizing cost, leveraging our trained cost regressor.
    
    Expected Input:
        - 'model_inSpec_classifier.joblib' (trained on in-spec data).
    Produces:
        - Printed or written results showing the optimal blend composition
          and associated metrics (e.g., total cost, volumes).
    """
    subprocess.run(["python", "03_optimize_blend.py"], check=True)


def optimize_blend_multi_output():
    """
    Calls '04_optimize_with_multioutput.py' to run a multi-objective optimization,
    typically balancing cost, octane margin, and RVP excess.
    
    Why:
        - Real-world blending requires trading off multiple factors (cost vs. quality margin).
          A multi-output regressor helps us explore that trade-off.
    
    Expected Input:
        - 'model_costmargin_regressor.joblib'
    Produces:
        - Printed or written results showing the best trade-offs
          between cost and margin properties.
    """
    subprocess.run(["python", "04_optimize_with_multioutput.py"], check=True)


def main():
    """
    Main entry point for running the entire pipeline.
    
    Why:
        - A single orchestrator function ensures each step is called in the
          correct order, guaranteeing all prerequisites are satisfied.
    """
    print("=== 1) Generating Synthetic Data ===")
    generate_synthetic_data()
    
    print("\n=== 2) Training Models ===")
    train_models()
    
    print("\n=== 3) Single-Output Blend Optimization ===")
    optimize_blend_single_output()
    
    print("\n=== 4) Multi-Output Blend Optimization ===")
    optimize_blend_multi_output()
    
    print("\nAll steps completed successfully.")


if __name__ == "__main__":
    main()
