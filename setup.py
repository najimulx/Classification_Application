from setuptools import setup, find_packages

setup(
    name="aeroreach",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "scikit-learn",
        "kmodes",
        "gower",
        "scipy",
        "streamlit",
        "matplotlib",
        "seaborn", 
        "plotly",
        "numpy"
    ]
)