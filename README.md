# Occupation Data Reports

A Python-based data science application that processes the 2023 Occupational Requirements Survey (ORS) dataset and generates comprehensive analytical reports for workforce planning, career guidance, and occupational research.

## Features

- **Comprehensive Data Processing**: Load and validate occupational survey data with footnote interpretation
- **Statistical Analysis**: Calculate confidence intervals, correlations, and reliability metrics
- **Multiple Report Types**: Generate 12+ different report types for various stakeholder needs
- **Interactive Visualizations**: Create charts, heatmaps, and dashboards using Plotly
- **Multiple Export Formats**: Output reports in HTML, PDF, and CSV formats
- **Configurable**: Flexible configuration system for customizing analysis parameters

## Project Structure

```
occupation-data-reports/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── main.py                   # Application entry point with CLI
│   ├── interfaces.py             # Base interfaces and data models
│   ├── config/                   # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── data_processing/          # Data loading and cleaning
│   │   └── __init__.py
│   ├── analysis/                 # Statistical analysis and calculations
│   │   └── __init__.py
│   ├── visualization/            # Chart and dashboard generation
│   │   └── __init__.py
│   └── export/                   # Report export and formatting
│       └── __init__.py
├── config/                       # Configuration files (auto-generated)
│   ├── data_sources.yaml
│   ├── output.yaml
│   ├── visualization.yaml
│   ├── analysis.yaml
│   └── reports.yaml
├── reports/                      # Generated reports (created at runtime)
├── requirements.txt              # Python dependencies
├── setup.py                      # Package installation script
└── README.md                     # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd occupation-data-reports
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Data Requirements

The application expects the following CSV files in the working directory:
- `2023-complete-dataset.csv` - Main occupational survey dataset
- `2023-complete-dataset-footnote-codes.csv` - Footnote reference codes
- `2023-complete-dataset-field names and description.csv` - Field descriptions

## Usage

### Command Line Interface

List available report types:
```bash
python -m src.main --list-reports
```

Generate a specific report:
```bash
python -m src.main --report-type occupation_distribution
```

Generate all enabled reports:
```bash
python -m src.main --generate-all
```

Validate configuration:
```bash
python -m src.main --validate-config
```

### Configuration

The application uses YAML configuration files in the `config/` directory:

- `data_sources.yaml` - Data file paths and loading settings
- `output.yaml` - Output directory and format settings
- `visualization.yaml` - Chart styling and visualization parameters
- `analysis.yaml` - Statistical analysis parameters
- `reports.yaml` - Individual report configurations

Configuration files are automatically created with default values on first run.

## Available Report Types

1. **occupation_distribution** - Occupation frequency and characteristics analysis
2. **environmental_risk** - Workplace environmental conditions and risk assessment
3. **skills_training** - Education, training, and skill requirements analysis
4. **work_autonomy** - Job flexibility and worker control patterns
5. **public_interaction** - Customer service and communication requirements
6. **physical_demands** - Physical requirements and ergonomic considerations
7. **data_quality** - Data confidence levels and reliability assessment
8. **cognitive_requirements** - Cognitive demands and mental requirements
9. **additive_analysis** - Combined occupational requirements analysis
10. **statistical_precision** - Confidence intervals and data precision visualization
11. **correlation_analysis** - Cross-requirement relationship analysis
12. **workforce_insights** - Policy-focused workforce representation analysis

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/
flake8 src/
```

### Adding New Report Types

1. Add report configuration to `config/reports.yaml`
2. Implement analysis logic in the `analysis/` module
3. Add visualization components in the `visualization/` module
4. Update export templates in the `export/` module

## Architecture

The application follows a modular architecture with clear separation of concerns:

- **Data Layer**: Handles CSV file ingestion, validation, and preprocessing
- **Processing Layer**: Performs statistical analysis and data transformations
- **Visualization Layer**: Creates charts, graphs, and interactive dashboards
- **Export Layer**: Generates reports in multiple formats

All components implement abstract interfaces defined in `interfaces.py` to ensure consistency and enable easy testing and extension.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For questions, issues, or contributions, please open an issue on the project repository.