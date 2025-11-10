"""
Setup script for the Occupation Data Reports application.
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read the README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read version from __init__.py
version_file = Path(__file__).parent / "src" / "__init__.py"
version = "1.0.0"  # Default version
if version_file.exists():
    version_content = version_file.read_text(encoding="utf-8")
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_content, re.M)
    if version_match:
        version = version_match.group(1)

# Core requirements (always installed)
core_requirements = [
    "pandas>=1.5.0,<3.0.0",
    "numpy>=1.21.0,<2.0.0",
    "scipy>=1.9.0,<2.0.0",
    "plotly>=5.10.0,<6.0.0",
    "matplotlib>=3.5.0,<4.0.0",
    "seaborn>=0.11.0,<1.0.0",
    "jinja2>=3.1.0,<4.0.0",
    "weasyprint>=56.0,<61.0",
    "reportlab>=3.6.0,<5.0.0",
    "pillow>=9.0.0,<11.0.0",
    "pyyaml>=6.0,<7.0.0",
    "pydantic>=1.10.0,<3.0.0",
    "psutil>=5.9.0,<6.0.0",
    "scikit-learn>=1.1.0,<2.0.0",
    "statsmodels>=0.13.0,<1.0.0",
    "openpyxl>=3.0.0,<4.0.0",
    "click>=8.0.0,<9.0.0",
    "rich>=12.0.0,<14.0.0",
    "tqdm>=4.64.0,<5.0.0",
    "structlog>=22.0.0,<24.0.0",
]

setup(
    name="occupation-data-reports",
    version=version,
    author="Occupation Data Reports Team",
    author_email="team@occupation-reports.com",
    description="A comprehensive Python application for analyzing occupational requirements survey data from the Bureau of Labor Statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/occupation-data-reports",
    project_urls={
        "Bug Reports": "https://github.com/your-org/occupation-data-reports/issues",
        "Source": "https://github.com/your-org/occupation-data-reports",
        "Documentation": "https://occupation-data-reports.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Office/Business :: Financial :: Spreadsheet",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: Console",
        "Natural Language :: English",
    ],
    keywords="occupational-data, workforce-analysis, bls-data, labor-statistics, data-science, reporting, visualization",
    python_requires=">=3.9",
    install_requires=core_requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "pytest-cov>=4.0.0,<5.0.0",
            "pytest-mock>=3.10.0,<4.0.0",
            "black>=22.0.0,<24.0.0",
            "flake8>=5.0.0,<7.0.0",
            "mypy>=0.991,<2.0.0",
            "isort>=5.10.0,<6.0.0",
            "bandit>=1.7.0,<2.0.0",
            "safety>=2.0.0,<4.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0,<8.0.0",
            "sphinx-rtd-theme>=1.0.0,<3.0.0",
            "myst-parser>=0.18.0,<3.0.0",
        ],
        "excel": [
            "xlrd>=2.0.0,<3.0.0",  # For legacy Excel files
        ],
        "all": [
            # Include all optional dependencies
            "pytest>=7.0.0,<8.0.0",
            "pytest-cov>=4.0.0,<5.0.0",
            "pytest-mock>=3.10.0,<4.0.0",
            "black>=22.0.0,<24.0.0",
            "flake8>=5.0.0,<7.0.0",
            "mypy>=0.991,<2.0.0",
            "isort>=5.10.0,<6.0.0",
            "bandit>=1.7.0,<2.0.0",
            "safety>=2.0.0,<4.0.0",
            "sphinx>=5.0.0,<8.0.0",
            "sphinx-rtd-theme>=1.0.0,<3.0.0",
            "myst-parser>=0.18.0,<3.0.0",
            "xlrd>=2.0.0,<3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "occupation-reports=src.main:main",
            "ors-reports=src.main:main",  # Alternative command name
        ],
    },
    include_package_data=True,
    package_data={
        "src": [
            "config/*.yaml",
            "config/templates/**/*.html",
            "config/templates/**/*.css",
        ],
        "": [
            "README.md",
            "LICENSE",
            "CHANGELOG.md",
        ],
    },
    data_files=[
        ("docs", [
            "docs/user_guide.md",
            "docs/troubleshooting.md",
            "docs/faq.md",
            "docs/report_types.md",
        ]),
    ],
    zip_safe=False,
    platforms=["any"],
    test_suite="tests",
)