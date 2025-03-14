# Gasoline Blend Optimization with Machine Learning
This repository demonstrates how to use machine learning pipelines (classification and regression) for gasoline blend optimization under realistic refinery constraints. 
By merging data-driven insights with domain-specific rules (e.g., tank volume limits, minimum octane requirements, and seasonal RVP caps), we aim to identify in-spec gasoline blends that minimize production costs.

## Repository Structure
1. **README.md** – Entry point to the project, containing an overview, setup instructions, and usage examples.
2. **data/** – Directory where you place input CSV files (e.g., `component_data_all.csv`) and where generated datasets (like `blends_dataset.csv`) may be stored.
3. **models/** – Directory to store the output `.joblib` files (trained models) after running the training scripts.
4. **scripts/** – Folder containing all Python scripts for data generation, model training, and blend optimization.
5. **main.py** – Executes the end-to-end pipeline.
6. **requirements.txt** – Lists the Python packages needed to run all scripts and code.

## Project Overview
Refineries blend multiple components to create gasoline meeting octane, RVP, and other environmental limits. This repository provides a Python-based approach for:
- Generating synthetic blend data that respects tank volumes and regulatory constraints.
- Training models to classify blends as in-spec or out-of-spec and to regress cost and near-limit margins.
- Optimizing blend composition by quickly filtering infeasible candidates and pinpointing the most cost-effective solutions.

## Data Preparation
1. Use the default or create a new input CSV file `component_data_all.csv` with daily tank volumes, component properties, and costs.
2. If it is a new file, make sure it follows the same format structure as the default file.
3. The data file `component_data_all.csv` should be placed in the `data/` folder:

## Installation & Dependencies
All Python packages required to run these scripts are listed in `requirements.txt`. To install them, use:
pip install -r requirements.txt
This ensures you have compatible versions of libraries such as:
- **pandas**
- **numpy**
- **scikit-learn**
- **joblib**

## Running Instructions
### Option 1: Single-Step Execution with main.py
python main.py
- Generates synthetic data (`blends_dataset.csv`) in `data/` using `scripts/01_generate_synthetic_blends.py`.
- Trains classification and regression models via `scripts/02_model_training.py`, saving `.joblib` files to `models/`.
- Optimizes blends with `scripts/03_optimize_blend.py` for single-output cost.
- Optionally demonstrates multi-output regression with `scripts/04_optimize_with_multioutput.py`.
You can edit `main.py` to skip certain steps.

### Option 2: Step-by-Step Execution
Run each script in sequence:

1. **Generate Synthetic Blends**
python scripts/01_generate_synthetic_blends.py
- Reads from `data/component_data_all.csv`
- Outputs `blends_dataset.csv` in `data/`

2. **Train Models**
python scripts/02_model_training.py
- Loads the synthetic dataset (`blends_dataset.csv`)
- Trains classification (Logistic Regression) and regression (Ridge) pipelines
- Saves trained models (e.g., `model_inSpec_classifier.joblib`, `model_cost_regressor.joblib`)

3. **Optimize Blends (Single-Output)**  
python scripts/03_optimize_blend.py
- Loads `model_inSpec_classifier.joblib` to quickly discard out-of-spec candidates
- Randomly searches candidate blends, confirms in-spec compliance via actual property calculations
- Identifies best-cost and near-limit blends, printing results to the console

4. **Optimize Blends (Multi-Output)**  
python scripts/04_optimize_with_multioutput.py
- Uses `model_costmargin_regressor.joblib` to predict cost plus octane/RVP margins
- Filters a large random pool of blends based on predicted feasibility
- Confirms final specs for top candidates, reports best-cost and near-limit options

## Configuration and Customization
- **Leftover Feasibility**: Adjust how much of each component’s daily tank volume can be drawn in `01_generate_synthetic_blends.py` (default is 75%).
- **Octane/RVP Limits**: Update the values of `OCTANE_REGULAR`, `OCTANE_PREMIUM`, `RVP_SUMMER`, `RVP_WINTER` in `01_generate_synthetic_blends.py` if you want to test alternative specifications.
- **Months for Summer/Winter**: In `scripts/01_generate_synthetic_blends.py`, edit the date ranges for “summer” vs. “winter” if your operational region uses different seasonal cutoffs.
- **Target Date & Gasoline Type**: In `03_optimize_blend.py`, modify `TARGET_DATE` and `TARGET_GASOLINE_TYPE` (e.g., “regular” vs. “premium”) to test different operating scenarios and operating conditions.
  
## Usage Examples
1. **Default run via main.py**
python main.py
- Generates data, trains models, and finds cost-optimal blends in one step.

2. **Individual Script Execution**
python scripts/01_generate_synthetic_blends.py 
python scripts/02_model_training.py 
python scripts/03_optimize_blend.py 
python scripts/04_optimize_with_multioutput.py

### Example: Changing Leftover Feasibility and Running a Different Scenario
Suppose you want to reduce leftover feasibility to 50% (meaning you can only draw half of each component’s daily tank volume at most) and check feasibility for premium gasoline on January 15, 2024. To do this:
- Open `scripts/01_generate_synthetic_blends.py` and change:
   ```python
   N_PER_DAY_PER_TYPE = 750
   ...
   leftover_ok = 1
   ...
   max_draw = 0.50 * tank_vol   # changed from 0.75
- In scripts/03_optimize_blend.py, update:
  ```python
  TARGET_DATE = `1/15/2024`
  TARGET_GASOLINE_TYPE = `premium`
- Run the pipeline again (via main.py or by individual script execution)
The model now searches blends using stricter draw constraints and focuses on premium gasoline for the operating conditions and regulations from January 15, 2024.
