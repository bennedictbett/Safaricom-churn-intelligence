from setuptools import setup, find_packages

setup(
    name="churn_intelligence",
    version="0.1.0",
    author="Benedict Bett",
    author_email="benedictbett08@gmail.com", 
    description="ML-powered customer churn prediction for Safaricom with Kenyan context",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "plotly",
        "pyyaml",
    ],
    python_requires=">=3.8",
)