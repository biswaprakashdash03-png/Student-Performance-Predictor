# Student Performance Predictor

A beginner-friendly machine learning project that estimates student exam scores from study hours, attendance, previous score, and assignments completed.

## Features

- Data analysis with Pandas
- Linear Regression model with Scikit-learn
- Missing-value handling and feature scaling in a pipeline
- MAE, RMSE, and R² evaluation on a held-out test set
- Interactive Streamlit interface
- Sample data visualization
- Reproducible sample dataset

> **Dataset notice:** `dataset/student_data.csv` is synthetic demo data generated for learning. It is not real student data, and model metrics do not establish real-world accuracy. Do not use this project for high-stakes educational decisions.

## Tech stack

Python, Pandas, NumPy, Matplotlib, Scikit-learn, Streamlit, Joblib.

## Project structure

```text
Student-Performance-Predictor/
├── dataset/
│   └── student_data.csv
├── docs/
│   └── PROJECT_DOCUMENTATION.md
├── models/
│   ├── student_model.pkl
│   └── metrics.json
├── src/
│   └── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

## Windows 11 setup (PowerShell)

Open PowerShell in the project directory.

### 1. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this in the current terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the model

```powershell
python src/train_model.py
```

This creates `models/student_model.pkl` and `models/metrics.json`.

### 4. Launch the app

```powershell
python -m streamlit run app.py
```

Streamlit will show a local address in the terminal, usually `http://localhost:8501`.

## Model workflow

1. Load the CSV dataset with Pandas.
2. Separate the input features from the target exam score.
3. Split the data into training and test sets.
4. Impute missing feature values with the median.
5. Standardize input features.
6. Train Linear Regression on the training set.
7. Evaluate predictions with MAE, RMSE, and R².
8. Save the fitted pipeline using Joblib.
9. Load the model in Streamlit and predict from user input.

## Input features

| Column | Description |
|---|---|
| `study_hours` | Study hours per day |
| `attendance` | Attendance percentage |
| `previous_score` | Previous exam score |
| `assignments_completed` | Assignments completed, from 0 to 10 |
| `exam_score` | Target score, from 0 to 100 |

## GitHub upload

Create an empty repository on GitHub, then run these commands from the project folder:

```powershell
git init
git add .
git commit -m "Build Student Performance Predictor ML project"
git branch -M main
git remote add origin https://github.com/biswaprakashdash03-png/Student-Performance-Predictor.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the repository URL GitHub gives you.

## Possible improvements

- Replace the synthetic dataset with a properly licensed real dataset.
- Compare Linear Regression, Random Forest, and Gradient Boosting.
- Add cross-validation and residual plots.
- Add unit tests and input validation.
- Deploy the Streamlit app after checking data privacy and hosting settings.

## License

Choose a license before publishing if you want others to reuse your code.
