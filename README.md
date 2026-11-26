# Water Potability Classification

ITEC 3040 Final Project · Group K

We use water-quality measurements to predict the recorded potability
label and compare classification models. Our planned Streamlit app
will let users explore the data and try sample predictions. This
project is for educational use and cannot replace laboratory testing.

## Dataset

Download `water_potability.csv` from the
[Kaggle Water Quality dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability).

## Project Report

Use the [shared Google Doc](https://docs.google.com/document/d/1gO0KoELg8l9ygaZ5QrnrgAlesLO77nwmD5nUdw_L7UU/edit?tab=t.0) to draft and review the final report. Use this repository for code, setup instructions, and testing.

## Contributing

- Create a branch for your part and open a pull request when it's ready
  to merge into `main`.
- In the pull request, include what you added, how to run it, and any
  dependencies or input files it needs.
- Use relative file paths so your code can run on other teammates'
  computers.

## Running the project

### Train the models

Run these commands from the repository root on a branch containing
`train_models.py`. Keep `water_potability.csv` next to that script. If the
CSV is missing, download it using the dataset link above.

The training workflow was verified on Windows (64-bit) with Python 3.14.6.
`requirements.txt` pins the packages from that environment.

Windows (PowerShell):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_models.py
```

macOS or Linux (with Python 3.14 installed):

```bash
python3.14 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python train_models.py
```

The script makes a stratified 80/20 split using seed 42. It prints test
accuracy and training cross-validation scores, then writes these files:

- `models/baseline.joblib`
- `models/logistic_regression.joblib`
- `models/random_forest.joblib`
- `results/predictions.csv`

With the assigned CSV, expect 2,620 training samples and 656 test
predictions. Rerunning the script replaces the files listed above in
the current checkout.

These instructions cover preprocessing and model training. The
Streamlit app has its own setup and run steps, which will be documented
after integration testing.
