"""
Setup script for fourier_pkg.

This file exists for backwards compatibility with older pip versions.
Modern pip (>= 21.1) prefers pyproject.toml, but setup.py is still
widely used and understood.

To install the package in development mode:
    pip install -e .

To install with all dependencies:
    pip install -e ".[dev]"

To build a distribution package:
    python -m build
"""

from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fourier-pkg",
    version="0.1.0",
    author="ProgrammingForScientist",
    author_email="author@example.com",
    description="A teaching package for understanding Fourier Transform with visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HCOG/ProgrammingForScientist",
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    package_data={
        "fourier_pkg": ["py.typed"],
    },
    include_package_data=True,
)
