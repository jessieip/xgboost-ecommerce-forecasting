# XGBoost Ecommerce Spend PredictorProject Installation & Setup Guide
## This guide covers the necessary steps to set up the development environment, from version management to automation.
### 1. Python Version Management (pyenv-win)We use pyenv to ensure the project runs on the exact Python version required.
#### Installation StepsInstall pyenv: Open PowerShell as Administrator and run:PowerShellInvoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
#### Install Python 3.12.9:
PowerShell
pyenv install 3.12.9
pyenv global 3.12.9
Verify Installation: Restart your terminal and run: python -V
Note: It should return 3.12.9.

### 2. Dependency Management (Poetry)This project uses Poetry to handle virtual environments and packages.
#### Setup CommandsInstall Poetry: pip install poetry
Initialize Environment: Navigate to the project root and run: poetry install
This will read the pyproject.toml file and automatically install all dependencies like numpy, pandas, and xgboost.

### 3. IDE Configuration (PyCharm)To ensure PyCharm uses the correct virtual environment, follow these steps:
Go to File > Settings > Project > Python Interpreter
Click Add Interpreter > Add Local Interpreter.Select Poetry Environment.Choose Existing environment and point it to the path created by Poetry (usually C:\Users\<User>\AppData\Local\pypoetry).

### 4. Development Workflow & Best PracticesManage your project lifecycle using Poetry rather than standard pip where possible.
#### System Configuration (Windows Only)Run this once in PowerShell as Admin to avoid installation errors due to path length limits:PowerShellNew-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
#### Adding PackagesCore Libraries:poetry add xgboost==2.0.0 shap==0.45.1 optuna google-cloud-bigquery db-dtypes
Dev Tools:poetry add --group dev black[jupyter] flake8 nbqa
##### Package MaintenanceView installed packages: poetry showGenerate requirements.txt: pip freeze > requirements.txt### 5. Automation with MakefileWe use make to standardize code quality across the team.
#### InstallationDownload via MinGW: In PowerShell, run mingw-get install mingw32-make.Usage: In your terminal, use make (or mingw32-make on some Windows setups).
#### Available CommandsCommandDescriptionmake installInstalls all dependencies via Poetry.make formatCleans up code style in .py and .ipynb files using Black.make lintChecks for PEP8 violations using Flake8.
#### Makefile Content
###### Makefile ConfigurationWarning: Indented lines below must use a Tab character.Makefileinstall:
	poetry install

format:
	poetry run black . --include '\.ipynb$$|\.py$$'

lint:
	poetry run nbqa flake8 . --ignore=E501
### 6. Git Hygiene (.gitignore)Ensure you have a .gitignore file in the root to prevent leaking large data or credentials.
Excluded Patterns:.venv/__pycache__/*.csv / *.json (Data files).env (Environment secrets)
