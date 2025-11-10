# Occupation Data Reports - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Configuration](#configuration)
5. [Report Types](#report-types)
6. [Command Line Interface](#command-line-interface)
7. [Batch Processing](#batch-processing)
8. [Output Formats](#output-formats)
9. [Data Requirements](#data-requirements)
10. [Advanced Usage](#advanced-usage)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)

## Introduction

The Occupation Data Reports application is a comprehensive Python-based data science platform that processes the 2023 Occupational Requirements Survey (ORS) dataset from the Bureau of Labor Statistics. It generates detailed analytical reports for workforce planning, career guidance, and occupational research.

### Key Features

- **Comprehensive Data Processing**: Handles 148,600+ occupational observations from 56,300 establishments
- **14 Report Types**: Specialized reports for different stakeholder needs
- **Statistical Rigor**: Confidence intervals, precision metrics, and footnote interpretation
- **Multiple Export Formats**: HTML (interactive), PDF (static), and CSV (raw data)
- **Batch Processing**: Generate multiple reports efficiently with parallel processing
- **Configurable**: Flexible configuration system for customizing analysis parameters

### Target Users

- **Workforce Analysts**: Occupation distribution and employment characteristics
- **Safety Managers**: Environmental risk assessment and workplace safety
- **HR Professionals**: Skills, training, and recruitment strategies
- **Career Counselors**: Career guidance and educational pathway planning
- **Policy Makers**: Evidence-based labor policy development
- **Researchers**: Academic and industry occupational research

## Installation

### Prerequisites

- Python 3.9 or higher
- 4GB+ RAM (recommended for full dataset processing)
- 2GB+ free disk space for reports and temporary files

### Step-by-Step Installation

1. **Clone or download the application**:
   ```bash
   git clone <repository-url>
   cd occupation-data-reports
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package**:
   ```bash
   pip install -e .
   ```

5. **Verify installation**:
   ```bash
   python -m src.main --help
   ```

### Docker Installation (Alternative)

```bash
# Build the Docker image
docker build -t occupation-reports .

# Run the application
docker run -v $(pwd)/data:/app/data -v $(pwd)/reports:/app/reports occupation-reports --list-reports
```

## Quick Start

### 1. Prepare Your Data

Download the required CSV files from the Bureau of Labor Statistics and place them in your working directory:

- `2023-complete-dataset.csv` - Main occupational survey dataset
- `2023-complete-dataset-footnote-codes.csv` - Footnote reference codes
- `2023-complete-dataset-field names and description.csv` - Field descriptions

### 2. Generate Your First Report

```bash
# List available report types
python -m src.main --list-reports

# Generate an occupation distribution report
python -m src.main --report-type occupation_distribution

# Generate all enabled reports
python -m src.main --generate-all
```

### 3. View Your Results

Reports are generated in the `reports/` directory with timestamped folders:
```
reports/
└── 20231101_143022/
    ├── occupation_distribution/
    │   ├── occupation_distribution.html
    │   ├── occupation_distribution.pdf
    │   └── occupation_distribution.csv
    └── master_dashboard.html
```

## Configuration

The application uses YAML configuration files in the `config/` directory. These files are automatically created with default values on first run.

### Configuration Files

#### `data_sources.yaml`
```yaml
main_dataset_path: "2023-complete-dataset.csv"
footnote_dataset_path: "2023-complete-dataset-footnote-codes.csv"
field_descriptions_path: "2023-complete-dataset-field names and description.csv"
encoding: "utf-8"
validation:
  required_columns: 18
  check_soc_codes: true
```

#### `output.yaml`
```yaml
base_output_dir: "reports"
create_timestamp_dirs: true
export_formats:
  - html
  - pdf
  - csv
master_dashboard: true
```

#### `reports.yaml`
```yaml
occupation_distribution:
  enabled: true
  title: "Occupation Distribution Analysis"
  description: "Frequency distribution and characteristics of occupations"
  custom_parameters:
    top_n: 20
    include_confidence_intervals: true

environmental_risk:
  enabled: true
  title: "Environmental Risk Assessment"
  description: "Workplace environmental conditions and risk analysis"
  custom_parameters:
    risk_threshold: 0.7
    detailed_analysis: true
```

### Customizing Configuration

1. **Edit configuration files** directly in the `config/` directory
2. **Validate your changes**:
   ```bash
   python -m src.main --validate-config
   ```
3. **Create configuration templates**:
   ```bash
   python -m src.main create-config --config-dir ./my_config
   ```

## Report Types

### 1. Occupation Distribution (`occupation_distribution`)
**Purpose**: Analyze the frequency and characteristics of different occupations.

**Key Insights**:
- Top 20 most common occupations
- Occupation diversity metrics
- Employment distribution across SOC codes
- Statistical confidence intervals

**Stakeholders**: Workforce analysts, career counselors, policy makers

### 2. Environmental Risk Assessment (`environmental_risk`)
**Purpose**: Identify workplace environmental hazards and risk levels.

**Key Insights**:
- Risk scoring for extreme temperatures, hazardous materials, heights
- Occupation ranking by environmental risk exposure
- Safety intervention recommendations
- Risk correlation analysis

**Stakeholders**: Safety managers, occupational health specialists

### 3. Physical Demands Analysis (`physical_demands`)
**Purpose**: Analyze physical requirements across occupations.

**Key Insights**:
- Lifting, carrying, climbing requirements
- Postural demands and ergonomic considerations
- Physical intensity matrices
- Accommodation recommendations

**Stakeholders**: Ergonomics specialists, HR professionals, disability services

### 4. Skills and Training Requirements (`skills_training`)
**Purpose**: Compare education, training, and skill requirements.

**Key Insights**:
- Minimum education levels by occupation
- Vocational preparation time requirements
- Credential and certification needs
- Skill similarity clustering

**Stakeholders**: HR professionals, educational planners, career counselors

### 5. Work Pace and Autonomy (`work_pace_autonomy`)
**Purpose**: Analyze job flexibility and worker control patterns.

**Key Insights**:
- Self-paced work opportunities
- Supervision frequency requirements
- Work flexibility scores
- Autonomy benchmarking

**Stakeholders**: Business analysts, HR professionals, organizational psychologists

### 6. Public Interaction Requirements (`public_interaction`)
**Purpose**: Identify customer service and communication requirements.

**Key Insights**:
- Public-facing role identification
- Verbal interaction frequency
- Communication skill requirements
- Customer service training needs

**Stakeholders**: Customer service managers, training coordinators

### 7. Data Quality Assessment (`data_quality`)
**Purpose**: Assess reliability and confidence levels of occupational data.

**Key Insights**:
- Footnote interpretation and precision levels
- Reliability scoring by occupation
- Confidence interval analysis
- Data interpretation guidelines

**Stakeholders**: Research analysts, statisticians, data scientists

### 8. Cognitive Requirements (`cognitive_requirements`)
**Purpose**: Analyze mental and cognitive demands across occupations.

**Key Insights**:
- Problem-solving complexity levels
- Literacy and decision-making requirements
- Cognitive demand categorization
- Educational pathway recommendations

**Stakeholders**: Educational planners, cognitive assessment specialists

### 9. Additive Category Analysis (`additive_category`)
**Purpose**: Understand how different occupational requirements combine.

**Key Insights**:
- Comprehensive requirement profiles
- Job family classification
- Requirement interaction patterns
- Combined demand analysis

**Stakeholders**: Workforce researchers, job classification specialists

### 10. Statistical Precision Dashboard (`statistical_precision`)
**Purpose**: Visualize confidence intervals and data reliability.

**Key Insights**:
- Interactive precision visualizations
- Confidence interval comparisons
- Reliability rankings
- Statistical methodology explanations

**Stakeholders**: Statisticians, research analysts, data quality specialists

### 11. Correlation Analysis (`correlation_analysis`)
**Purpose**: Analyze relationships between different occupational requirements.

**Key Insights**:
- Cross-requirement correlation matrices
- Requirement interdependencies
- Pattern identification
- Unusual requirement combinations

**Stakeholders**: Labor economists, workforce researchers

### 12. Workforce Insights (`workforce_insights`)
**Purpose**: Provide policy-focused workforce representation analysis.

**Key Insights**:
- Establishment sample representativeness
- Population-weighted insights
- Policy implications
- Coverage assessments

**Stakeholders**: Policy makers, labor economists, government analysts

### 13. Comprehensive Summary (`comprehensive_summary`)
**Purpose**: Executive summary combining insights from multiple report types.

**Key Insights**:
- Cross-report synthesis
- Key findings summary
- Executive recommendations
- Integrated analysis

**Stakeholders**: Executives, decision makers, policy makers

### 14. Establishment Coverage (`establishment_coverage`)
**Purpose**: Analyze survey coverage and representativeness.

**Key Insights**:
- Sample coverage statistics
- Industry representation
- Geographic distribution
- Survey methodology assessment

**Stakeholders**: Survey methodologists, policy makers, researchers

## Command Line Interface

### Basic Commands

```bash
# List all available report types
python -m src.main --list-reports

# Generate a specific report
python -m src.main --report-type occupation_distribution

# Generate all enabled reports
python -m src.main --generate-all

# Validate configuration
python -m src.main --validate-config

# Show application status
python -m src.main status

# Test the data processing pipeline
python -m src.main test-pipeline
```

### Advanced Options

```bash
# Custom output directory
python -m src.main --report-type environmental_risk --output-dir ./custom_reports

# Specific output format
python -m src.main --report-type data_quality --output-format pdf

# Parallel processing for multiple reports
python -m src.main --generate-all --parallel --max-workers 8

# Custom configuration directory
python -m src.main --report-type skills_training --config-dir ./my_config

# Dry run (show what would be done)
python -m src.main --generate-all --dry-run

# Verbose logging
python -m src.main --report-type correlation_analysis --verbose

# Custom data file
python -m src.main --report-type workforce_insights --dataset-file ./data/custom_dataset.csv
```

### Configuration Management

```bash
# Create configuration templates
python -m src.main create-config --config-dir ./new_config

# Validate specific configuration directory
python -m src.main validate-config --config-dir ./my_config

# Export sample batch configuration
python -m src.main export-sample-config --output batch_config.json
```

## Batch Processing

Batch processing allows you to generate multiple reports efficiently with custom parameters and parallel execution.

### Using Batch Configuration Files

1. **Create a batch configuration**:
   ```bash
   python -m src.main export-sample-config --output my_batch.json
   ```

2. **Customize the configuration**:
   ```json
   {
     "description": "Custom batch for safety analysis",
     "jobs": [
       {
         "job_id": "env_risk_detailed",
         "report_type": "environmental_risk",
         "filters": {
           "risk_threshold": 0.8
         },
         "config": {
           "detailed_analysis": true
         },
         "output_path": "reports/safety_analysis",
         "export_formats": ["html", "pdf"]
       },
       {
         "job_id": "physical_demands",
         "report_type": "physical_demands",
         "filters": {},
         "config": {
           "include_accommodations": true
         },
         "output_path": "reports/safety_analysis",
         "export_formats": ["html", "csv"]
       }
     ],
     "global_settings": {
       "parallel_processing": true,
       "max_workers": 4
     }
   }
   ```

3. **Run the batch**:
   ```bash
   python -m src.main --batch-config my_batch.json
   ```

### Using Predefined Templates

```bash
# Comprehensive analysis (all reports)
python -m src.main --template comprehensive

# Safety-focused analysis
python -m src.main --template safety_focused

# Workforce analysis for policy makers
python -m src.main --template workforce_analysis

# Statistical analysis for researchers
python -m src.main --template statistical_analysis

# Basic analysis (core reports only)
python -m src.main --template basic
```

## Output Formats

### HTML Reports (Interactive)
- **Features**: Interactive charts, tooltips, filtering
- **Best for**: Presentations, detailed exploration
- **File size**: Larger (includes embedded JavaScript)
- **Example**: `occupation_distribution.html`

### PDF Reports (Static)
- **Features**: Professional formatting, print-ready
- **Best for**: Documentation, formal reports
- **File size**: Medium (optimized for printing)
- **Example**: `environmental_risk.pdf`

### CSV Data (Raw)
- **Features**: Raw data, statistical summaries
- **Best for**: Further analysis, data integration
- **File size**: Smallest (text-based)
- **Example**: `data_quality.csv`

### Master Dashboard
- **Features**: Combined overview of all reports
- **Location**: `master_dashboard.html` in output directory
- **Purpose**: Executive summary and navigation

## Data Requirements

### Required Files

1. **Main Dataset** (`2023-complete-dataset.csv`)
   - 148,600+ occupational observations
   - 18 required columns including SOC codes, estimates, standard errors
   - Size: ~50-100 MB

2. **Footnote Codes** (`2023-complete-dataset-footnote-codes.csv`)
   - 36 footnote code definitions
   - Precision indicators and interpretation guidance
   - Size: ~10 KB

3. **Field Descriptions** (`2023-complete-dataset-field names and description.csv`)
   - Column definitions and metadata
   - Data dictionary for all fields
   - Size: ~5 KB

### Data Quality Considerations

- **Missing Values**: Handled gracefully with footnote interpretation
- **Standard Errors**: Used for confidence interval calculations
- **SOC Code Validation**: Verified against 2018 classification system
- **Footnote Interpretation**: Converts range estimates to numeric values

### File Placement

Place data files in one of these locations:
1. **Working directory** (default): Same folder as the application
2. **Custom directory**: Use `--data-dir` parameter
3. **Specific files**: Use `--dataset-file` parameter

## Advanced Usage

### Custom Analysis Parameters

Modify report behavior through configuration files:

```yaml
# In reports.yaml
environmental_risk:
  enabled: true
  custom_parameters:
    risk_threshold: 0.8          # Higher threshold for risk classification
    include_correlations: true   # Add correlation analysis
    detailed_breakdown: true     # Include detailed risk components
    confidence_level: 0.99       # Use 99% confidence intervals
```

### Performance Optimization

1. **Parallel Processing**:
   ```bash
   python -m src.main --generate-all --parallel --max-workers 8
   ```

2. **Memory Management**:
   - Process reports individually for large datasets
   - Use CSV output format to reduce memory usage
   - Monitor system resources with `python -m src.main status`

3. **Caching**:
   - Processed data is cached between report generations
   - Clear cache by deleting temporary files in `temp/` directory

### Integration with Other Tools

1. **Export to Excel**:
   ```python
   import pandas as pd
   
   # Read CSV output
   df = pd.read_csv('reports/occupation_distribution/occupation_distribution.csv')
   
   # Export to Excel
   df.to_excel('occupation_analysis.xlsx', index=False)
   ```

2. **API Integration** (Future):
   ```python
   # Planned feature for programmatic access
   from occupation_reports import ReportAPI
   
   api = ReportAPI()
   result = api.generate_report('environmental_risk', filters={'risk_threshold': 0.8})
   ```

### Custom Report Development

To add new report types:

1. **Create report generator class**:
   ```python
   from src.reports.base_report_generator import BaseReportGenerator
   
   class CustomReportGenerator(BaseReportGenerator):
       def generate_report(self, data, config):
           # Your custom analysis logic
           pass
   ```

2. **Register with factory**:
   ```python
   from src.reports.report_factory import ReportFactory
   
   factory = ReportFactory()
   factory.register_generator('custom_report', CustomReportGenerator)
   ```

3. **Add configuration**:
   ```yaml
   # In reports.yaml
   custom_report:
     enabled: true
     title: "Custom Analysis Report"
     description: "Your custom analysis description"
   ```

## Troubleshooting

### Common Issues

#### 1. "File not found" errors
**Problem**: Data files are not in the expected location.

**Solutions**:
- Verify file names match exactly: `2023-complete-dataset.csv`
- Check file permissions (read access required)
- Use absolute paths: `--dataset-file /full/path/to/dataset.csv`
- Verify files are not corrupted (check file sizes)

#### 2. Memory errors during processing
**Problem**: Insufficient RAM for large dataset processing.

**Solutions**:
- Close other applications to free memory
- Process reports individually instead of using `--generate-all`
- Use a machine with more RAM (4GB+ recommended)
- Enable swap space on your system

#### 3. Configuration validation failures
**Problem**: Invalid configuration files.

**Solutions**:
- Run `python -m src.main --validate-config` for detailed errors
- Recreate configuration files: `python -m src.main create-config`
- Check YAML syntax (indentation, colons, quotes)
- Verify file paths exist and are accessible

#### 4. Empty or incomplete reports
**Problem**: Reports generate but contain no data or charts.

**Solutions**:
- Verify data files contain expected columns
- Check footnote interpretation (some estimates may be excluded)
- Review log files for processing warnings
- Ensure sufficient data meets report criteria

#### 5. Slow performance
**Problem**: Report generation takes too long.

**Solutions**:
- Enable parallel processing: `--parallel`
- Increase worker count: `--max-workers 8`
- Generate specific reports instead of all reports
- Check system resources: `python -m src.main status`

### Debugging Steps

1. **Check system status**:
   ```bash
   python -m src.main status --verbose
   ```

2. **Validate configuration**:
   ```bash
   python -m src.main --validate-config
   ```

3. **Test data pipeline**:
   ```bash
   python -m src.main test-pipeline
   ```

4. **Enable verbose logging**:
   ```bash
   python -m src.main --report-type occupation_distribution --verbose --log-file debug.log
   ```

5. **Run dry run**:
   ```bash
   python -m src.main --generate-all --dry-run
   ```

### Log Files

Log files are created in the `logs/` directory:
- `application.log` - General application logs
- `performance_metrics_YYYYMMDD_HHMMSS.json` - Performance data
- `error.log` - Error details and stack traces

### Getting Help

1. **Check this documentation** for common solutions
2. **Review log files** for specific error messages
3. **Validate your setup** using built-in diagnostic tools
4. **Check data file integrity** and format compliance
5. **Open an issue** on the project repository with:
   - Error messages and log files
   - System information (`python -m src.main status`)
   - Configuration files (remove sensitive data)
   - Steps to reproduce the issue

## FAQ

### General Questions

**Q: What is the Occupational Requirements Survey (ORS)?**
A: The ORS is a Bureau of Labor Statistics survey that collects data on the physical demands, environmental conditions, education and training requirements, and cognitive and mental requirements of occupations in the U.S. economy.

**Q: How often is the ORS data updated?**
A: The BLS typically updates ORS data every few years. This application is designed for the 2023 dataset but can be adapted for future releases.

**Q: Can I use this application with other datasets?**
A: The application is specifically designed for ORS data format. Using other datasets would require code modifications to handle different column structures and data formats.

### Technical Questions

**Q: What Python version is required?**
A: Python 3.9 or higher is required. The application uses modern Python features and type hints.

**Q: How much disk space do I need?**
A: Minimum 2GB free space is recommended: ~100MB for data files, ~500MB for generated reports, and additional space for temporary processing files.

**Q: Can I run this on a server without a GUI?**
A: Yes, the application is command-line based and works on headless servers. All visualizations are saved as files rather than displayed interactively.

**Q: How do I cite this application in research?**
A: Include the application name, version, and repository URL. Also cite the original BLS ORS dataset as the data source.

### Data Questions

**Q: What do the footnote codes mean?**
A: Footnote codes (1-36) provide important context about data quality, precision, and interpretation. The application automatically interprets these codes and includes explanations in reports.

**Q: Why are some occupations missing from reports?**
A: Occupations may be excluded due to insufficient data, high standard errors, or specific footnote codes indicating unreliable estimates.

**Q: How are confidence intervals calculated?**
A: Confidence intervals are calculated using standard errors provided in the dataset, typically at the 95% confidence level unless otherwise specified.

### Configuration Questions

**Q: Can I customize report parameters?**
A: Yes, modify the `reports.yaml` configuration file to adjust parameters like confidence levels, thresholds, and analysis options.

**Q: How do I disable certain reports?**
A: Set `enabled: false` for specific report types in the `reports.yaml` configuration file.

**Q: Can I change the output directory structure?**
A: Yes, modify the `output.yaml` configuration file to customize directory names, timestamp formats, and file organization.

### Performance Questions

**Q: How long does it take to generate all reports?**
A: Processing time varies by system specifications. Typically 5-15 minutes for all reports on a modern computer with parallel processing enabled.

**Q: Can I run multiple instances simultaneously?**
A: Yes, but ensure each instance uses different output directories to avoid file conflicts.

**Q: How do I optimize performance for large datasets?**
A: Enable parallel processing, increase worker count, ensure sufficient RAM, and consider processing reports individually rather than in batch.

### Output Questions

**Q: Which output format should I use?**
A: Use HTML for interactive exploration, PDF for formal documentation, and CSV for further data analysis.

**Q: Can I customize the appearance of reports?**
A: Yes, modify templates in the `config/templates/` directory and visualization settings in `visualization.yaml`.

**Q: How do I share reports with others?**
A: HTML reports are self-contained and can be shared directly. PDF reports are ideal for email distribution. CSV files can be imported into other analysis tools.

---

*This user guide covers the essential aspects of using the Occupation Data Reports application. For additional support, please refer to the troubleshooting section or contact the development team.*