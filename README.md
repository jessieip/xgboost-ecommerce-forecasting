# XGBoost Ecommerce Spend Prediction Setup Guide
## Follow these steps to set up the development environment for the XGBoost Ecommerce Spend Prediction.
### 1. Python Version Management (pyenv-win)

We use `pyenv` to ensure the project runs on the exact Python version required.

#### Install pyenv
Open **PowerShell as Administrator** and run:

```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; & "./install-pyenv-win.ps1"
```

#### Install Python 3.12.9

```bash
pyenv install 3.12.9
pyenv global 3.12.9
```

#### Verify Installation
Restart your terminal and run:

```bash
python -V
```

> **Note:** It should return `3.12.9`.

---

### 2. Dependency Management (Poetry)

This project uses Poetry to handle virtual environments and packages.

#### Install Poetry
Run the following command:

```bash
pip install poetry
```

#### Initialize Environment
Navigate to the project root and run:

```bash
poetry install
```

This will read the `pyproject.toml` file and automatically install all dependencies like `numpy`, `pandas`, and `xgboost`.

---

### 3. IDE Configuration (PyCharm)

To ensure PyCharm uses the correct virtual environment, follow these steps:

1. Go to **File > Settings > Interpreter**
2. Click **Add Interpreter > Add Local Interpreter**
3. Select **Poetry Environment**
4. Choose **Existing environment**
5. Point it to the path created by Poetry  
   (usually: `C:\Users\<User>\AppData\Local\pypoetry`)
![5](https://github.com/user-attachments/assets/9df4e3c1-0b34-43c3-9500-c69c89829562)

---

### 4. Development Workflow

#### Add a New Package
Add the required package (e.g., `numpy`, `pandas`) to the project using Poetry:

```bash
poetry add <package-name>
```

#### Run the Project

```bash
poetry install
```

#### Check Installed Packages

```bash
poetry show
```

---

### 5. Dependency Management

We use **Poetry** to manage dependencies.  
❗ Do **not** use `pip install` directly unless absolutely necessary.

#### Configure Long Paths (Windows Only)

Run this once in **PowerShell (Admin)** to avoid installation errors:

```powershell
New-ItemProperty -Path "HKLM:\System\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

#### Install Core Libraries

```bash
poetry add xgboost==2.0.0 shap==0.45.1 optuna
poetry add google-cloud-bigquery db-dtypes
```

#### Install Dev Tools

```bash
poetry add --group dev black[jupyter] flake8 nbqa
```

> ⚠️ Avoid mixing `pip` with Poetry. Only use the following if required:
```bash
pip install flake8 nbqa "black[jupyter]"
pip freeze > requirements.txt
```

#### Install All Dependencies

```bash
make install
```

> 💡 Dev tools are added to a `"dev"` group because they are not required in production.

---

### 6. Automation with Makefile

We use `make` to standardise formatting and linting.

#### Installation

1. Install via MinGW (PowerShell):
   ```powershell
   mingw-get install mingw32-make
   ```

2. Use in terminal:
   - `make`  
   - or `mingw32-make` (on some Windows setups)

#### Available Commands

| Command        | Description |
|----------------|------------|
| `make install` | Installs all dependencies from `pyproject.toml` |
| `make format`  | Formats `.py` and `.ipynb` files using Black |
| `make lint`    | Checks for PEP8 violations (ignores `E501`) |

#### Makefile Content

> ⚠️ **Important:** All indented lines must use a **Tab** (not spaces)

```makefile
install:
	poetry install

format:
	poetry run black . --include '\.ipynb$$|\.py$$'

lint:
	poetry run nbqa flake8 . --ignore=E501
```

---

### 7. Git Hygiene (.gitignore)

Ensure you have a `.gitignore` file in the project root to avoid committing unnecessary or sensitive files.

#### Recommended Exclusions

```
.venv/
__pycache__/
.ipynb_checkpoints/
*.csv
*.json
.env
```

> 💡 Exclude data files (`*.csv`, `*.json`) only if they contain sensitive or large data.
