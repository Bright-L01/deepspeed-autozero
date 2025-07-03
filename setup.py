#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="deepspeed-autozero",
    version="0.1.0",
    author="DeepSpeed AutoZero Team",
    author_email="team@deepspeed-autozero.io",
    description="Automated DeepSpeed ZeRO configuration generator based on model and hardware analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deepspeed-autozero",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "typer[all]>=0.9.0",
        "deepspeed>=0.12.0",
        "torch>=2.0.0",
        "jsonschema>=4.0.0",
        "rich>=13.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "autozero=autozero.cli:app",
        ],
    },
)