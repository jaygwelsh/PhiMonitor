# setup.py

from setuptools import setup, find_packages
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# Read the README file for the long description
README = (HERE / "README.md").read_text()

# Read the LICENSE file
LICENSE = (HERE / "LICENSE").read_text()

setup(
    name="phi_monitor",
    version="0.2.0",
    author="Your Name",
    author_email="your-email@example.com",
    description="A Python library for monitoring machine learning models to detect data drift and overfitting.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/phi_monitor",
    packages=find_packages(exclude=["tests*", "benchmarks*", "examples*"]),
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "scikit-learn>=0.22.0",
        "matplotlib>=3.1.0",
        "pandas>=1.0.0",
        "seaborn>=0.11.0",
        "streamlit>=1.0.0",
        "shap>=0.46.0",
    ],
    extras_require={
        "optional": [
            "boto3>=1.35.0",
            "google-cloud-storage>=2.18.0",
            "azure-storage-blob>=12.24.0",
            "requests>=2.32.0",
        ],
        "dev": [
            "pytest>=8.3.0",
            "flake8>=4.0.0",
            "black>=23.0.0",
            "mypy>=0.991",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="MIT",
    license_files=["LICENSE"],
    python_requires='>=3.7',
    include_package_data=True,
    keywords="machine-learning data-drift overfitting monitoring",
)
