# Data Files

This directory contains the data files used by the Occupation Data Reports application.

## Required Data Files

To run the application, you need to download the following data files from the Bureau of Labor Statistics:

### Main Dataset (Not included in repository due to size)

- `2023-complete-dataset.csv` (42MB) - Main occupational requirements survey data
- `2023-complete-dataset.xlsx` (11MB) - Excel version of the main dataset

**Download from:** [BLS Occupational Requirements Survey](https://www.bls.gov/ors/)

### Metadata Files (Included in repository)

- `2023-complete-dataset-field names and description.csv` - Field descriptions and metadata
- `2023-complete-dataset-footnote-codes.csv` - Footnote codes and their meanings
- `StudentsPerformance.csv` - Sample dataset for testing

## Data Directory Structure

```
.
├── 2023-complete-dataset.csv              # Main dataset (download required)
├── 2023-complete-dataset.xlsx             # Excel version (download required)
├── 2023-complete-dataset-field names and description.csv
├── 2023-complete-dataset-footnote-codes.csv
├── StudentsPerformance.csv                # Sample data
└── DATA_README.md                         # This file
```

## Setup Instructions

1. Download the required dataset files from the BLS website
2. Place them in the project root directory
3. Run the application using the CLI commands

## Data Processing

The application automatically processes the data files and generates reports based on the configuration in `config/reports_config.yaml`.

For more information, see the main [README.md](README.md) and [User Guide](docs/user_guide.md).
