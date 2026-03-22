import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

project_name = "churn_intelligence"

files_to_remove = [
    "README.md       # API deployment    main.py",  
    "src/visualization.py",                          
]
for f in files_to_remove:
    p = Path(f)
    if p.exists():
        p.unlink()
        logging.info(f"Removed unwanted file: {f}")



list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    "src/__init__.py",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "notebooks/01_data_preparation.ipynb",
    "notebooks/02_exploratory_analysis.ipynb",
    "notebooks/03_feature_engineering.ipynb",
    "notebooks/04_model_training.ipynb",
    "notebooks/05_business_insights.ipynb",
    "src/data_processing.py",
    "src/feature_engineering.py",
    "src/model.py",                          
    "models/.gitkeep",
    "reports/figures/.gitkeep",
    "reports/executive_summary.md",
    "config/config.yaml",
    "requirements.txt",
    "setup.py",
    "README.md",                             
    # API deployment
    "main.py",
    "render.yaml",
    "Dockerfile",
    "DEPLOYMENT.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

logging.info("Project structure created successfully!")