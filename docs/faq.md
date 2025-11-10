# Frequently Asked Questions (FAQ)

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation and Setup](#installation-and-setup)
3. [Data and Dataset Questions](#data-and-dataset-questions)
4. [Report Generation](#report-generation)
5. [Configuration and Customization](#configuration-and-customization)
6. [Performance and System Requirements](#performance-and-system-requirements)
7. [Output and Export](#output-and-export)
8. [Technical Questions](#technical-questions)
9. [Research and Academic Use](#research-and-academic-use)
10. [Troubleshooting Quick Answers](#troubleshooting-quick-answers)

## General Questions

### What is the Occupation Data Reports application?

The Occupation Data Reports application is a Python-based data science platform that processes the 2023 Occupational Requirements Survey (ORS) dataset from the Bureau of Labor Statistics. It generates comprehensive analytical reports for workforce planning, career guidance, and occupational research.

### Who should use this application?

The application is designed for:
- **Workforce analysts** analyzing employment characteristics
- **Safety managers** assessing workplace environmental risks
- **HR professionals** developing recruitment and training strategies
- **Career counselors** providing career guidance
- **Policy makers** developing evidence-based labor policies
- **Researchers** conducting occupational studies
- **Educational planners** designing curriculum and training programs

### What is the Occupational Requirements Survey (ORS)?

The ORS is a Bureau of Labor Statistics survey that collects data on:
- Physical demands of occupations
- Environmental conditions in workplaces
- Education, training, and experience requirements
- Cognitive and mental requirements

The 2023 dataset contains 148,600+ occupational observations from 56,300 establishments representing 145,866,200 civilian workers.

### Is this application free to use?

Yes, the application is open source and free to use. However, you need to obtain the ORS dataset from the Bureau of Labor Statistics, which is also freely available.

### How often is the ORS data updated?

The Bureau of Labor Statistics typically updates ORS data every few years. This application is designed for the 2023 dataset, but can be adapted for future releases with minor modifications.

## Installation and Setup

### What are the system requirements?

**Minimum Requirements:**
- Python 3.9 or higher
- 4GB RAM
- 2GB free disk space
- Windows 10, macOS 10.14, or Linux (Ubuntu 18.04+)

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- 5GB+ free disk space
- SSD storage for better performance

### Can I run this on a server without a GUI?

Yes, the application is completely command-line based and works perfectly on headless servers. All visualizations are saved as files rather than displayed interactively.

### Do I need to install additional software?

The application handles most dependencies automatically through pip. However, for PDF generation, you may need to install system-level dependencies:

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install python3-cffi python3-brotli libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0
```

**Windows:**
- Visual C++ Build Tools (for some packages)

**macOS:**
- Xcode Command Line Tools: `xcode-select --install`

### Can I use conda instead of pip?

Yes, you can use conda for package management:
```bash
conda create -n occupation-reports python=3.9
conda activate occupation-reports
conda install pandas numpy matplotlib plotly pyyaml
pip install -r requirements.txt  # For packages not available in conda
```

### How do I update to a newer version?

1. **Pull latest changes** (if using git):
   ```bash
   git pull origin main
   ```

2. **Update dependencies**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Reinstall the package**:
   ```bash
   pip install -e . --force-reinstall
   ```

## Data and Dataset Questions

### Where do I get the ORS dataset?

Download the required files from the Bureau of Labor Statistics:
1. Visit the BLS ORS data page
2. Download the 2023 complete dataset files:
   - `2023-complete-dataset.csv`
   - `2023-complete-dataset-footnote-codes.csv`
   - `2023-complete-dataset-field names and description.csv`

### What if I have the data in Excel format?

Convert Excel files to CSV format:
1. Open the Excel file
2. Use "Save As" → "CSV (Comma delimited)"
3. Ensure the filename matches exactly: `2023-complete-dataset.csv`

### Can I use a subset of the data for testing?

Yes, you can create a smaller test dataset:
```bash
# Create a sample with first 1000 rows (plus header)
head -n 1001 2023-complete-dataset.csv > test-dataset.csv
```

Then specify the test file:
```bash
python -m src.main --dataset-file test-dataset.csv --report-type occupation_distribution
```

### What do the footnote codes mean?

Footnote codes (1-36) provide important context about data quality and interpretation:
- **Codes 1-10**: Precision and reliability indicators
- **Codes 11-20**: Range estimates (e.g., "less than 0.5%")
- **Codes 21-30**: Special conditions and limitations
- **Codes 31-36**: Survey methodology notes

The application automatically interprets these codes and includes explanations in reports.

### Why are some occupations missing from my reports?

Occupations may be excluded due to:
- **Insufficient sample size**: Too few observations for reliable analysis
- **High standard errors**: Estimates are too imprecise
- **Footnote restrictions**: Data marked as unreliable or suppressed
- **Filter criteria**: Your configuration excludes certain occupations

Check the data quality report to understand data availability by occupation.

### Can I use this application with other datasets?

The application is specifically designed for the ORS data format. Using other datasets would require significant code modifications to handle different:
- Column structures
- Data types
- Footnote systems
- Statistical methodologies

## Report Generation

### How many report types are available?

The application includes 14 specialized report types:

1. **Occupation Distribution** - Employment characteristics and frequency
2. **Environmental Risk** - Workplace safety and environmental hazards
3. **Physical Demands** - Physical requirements and ergonomic analysis
4. **Skills Training** - Education and training requirements
5. **Work Pace Autonomy** - Job flexibility and worker control
6. **Public Interaction** - Customer service and communication needs
7. **Data Quality** - Reliability and confidence assessment
8. **Cognitive Requirements** - Mental and cognitive demands
9. **Additive Category** - Combined requirement analysis
10. **Statistical Precision** - Confidence intervals and precision metrics
11. **Correlation Analysis** - Cross-requirement relationships
12. **Workforce Insights** - Policy-focused workforce analysis
13. **Comprehensive Summary** - Executive overview of all analyses
14. **Establishment Coverage** - Survey representativeness analysis

### Which reports should I generate first?

For new users, start with:
1. **Data Quality** - Understand data reliability
2. **Occupation Distribution** - Get overview of the dataset
3. **Environmental Risk** or **Physical Demands** - Specific analysis examples

### How long does it take to generate all reports?

Processing time varies by system:
- **Single report**: 30 seconds to 2 minutes
- **All reports (sequential)**: 10-30 minutes
- **All reports (parallel)**: 5-15 minutes

Factors affecting speed:
- System specifications (CPU, RAM, storage)
- Data size and complexity
- Number of visualizations
- Export formats selected

### Can I generate reports for specific occupations only?

Yes, you can filter by occupation through configuration:
```yaml
# In reports.yaml
occupation_distribution:
  custom_parameters:
    occupation_filter: ["Management occupations", "Healthcare practitioners"]
    soc_codes: ["11-0000", "29-0000"]  # Specific SOC codes
```

### What's the difference between the output formats?

- **HTML**: Interactive charts, tooltips, filtering capabilities. Best for exploration and presentations.
- **PDF**: Static, print-ready format. Best for formal documentation and sharing.
- **CSV**: Raw data and statistical summaries. Best for further analysis in other tools.

## Configuration and Customization

### How do I customize report parameters?

Edit the `reports.yaml` configuration file:
```yaml
environmental_risk:
  enabled: true
  custom_parameters:
    risk_threshold: 0.8        # Higher threshold for risk classification
    confidence_level: 0.99     # Use 99% confidence intervals
    detailed_analysis: true    # Include detailed breakdowns
    include_correlations: true # Add correlation analysis
```

### Can I change the appearance of reports?

Yes, modify visualization settings in `visualization.yaml`:
```yaml
chart_style:
  color_scheme: "viridis"     # Color palette
  font_family: "Arial"        # Font for charts
  figure_size: [12, 8]        # Chart dimensions

formatting:
  decimal_places: 2           # Number precision
  percentage_format: "0.1%"   # Percentage display
```

### How do I disable certain reports?

Set `enabled: false` in `reports.yaml`:
```yaml
correlation_analysis:
  enabled: false  # This report will be skipped
```

### Can I create custom output directories?

Yes, specify custom directories:
```bash
# Command line
python -m src.main --output-dir /path/to/custom/reports

# Configuration (output.yaml)
base_output_dir: "/path/to/custom/reports"
create_timestamp_dirs: true
```

### How do I add new report types?

Adding new reports requires Python development:
1. Create a new report generator class inheriting from `BaseReportGenerator`
2. Register it with the `ReportFactory`
3. Add configuration to `reports.yaml`
4. Implement the analysis logic and visualization

See the developer documentation for detailed instructions.

## Performance and System Requirements

### Why is report generation slow?

Common causes and solutions:

**Large dataset processing:**
- Enable parallel processing: `--parallel`
- Increase worker count: `--max-workers 8`

**Insufficient memory:**
- Generate reports individually instead of all at once
- Close other applications to free RAM

**Slow storage:**
- Use SSD instead of traditional hard drive
- Ensure sufficient free disk space

**Complex visualizations:**
- Reduce chart complexity in `visualization.yaml`
- Use static charts instead of interactive ones

### How much memory does the application use?

Memory usage varies by operation:
- **Data loading**: 500MB - 1GB
- **Single report**: 1GB - 2GB
- **All reports (parallel)**: 2GB - 4GB
- **Peak usage**: Up to 6GB for complex analyses

Monitor usage with:
```bash
python -m src.main status --verbose
```

### Can I run multiple instances simultaneously?

Yes, but ensure:
- Each instance uses different output directories
- Sufficient system resources (RAM, CPU)
- Different configuration directories if needed

Example:
```bash
# Terminal 1
python -m src.main --report-type occupation_distribution --output-dir reports1

# Terminal 2
python -m src.main --report-type environmental_risk --output-dir reports2
```

### How do I optimize performance?

**System optimization:**
- Use SSD storage
- Ensure adequate RAM (8GB+)
- Close unnecessary applications

**Application optimization:**
- Enable parallel processing for multiple reports
- Use data sampling for testing: `sample_size: 10000`
- Generate specific reports instead of all reports
- Use CSV output format (fastest)

**Configuration optimization:**
```yaml
# In analysis.yaml
performance:
  chunk_size: 1000          # Process data in smaller chunks
  lazy_loading: true        # Load data on demand
  cache_results: true       # Cache intermediate results
  max_iterations: 1000      # Limit complex calculations
```

## Output and Export

### Where are the generated reports saved?

By default, reports are saved in:
```
reports/
└── YYYYMMDD_HHMMSS/     # Timestamped directory
    ├── occupation_distribution/
    ├── environmental_risk/
    └── master_dashboard.html
```

### What is the master dashboard?

The master dashboard (`master_dashboard.html`) is a combined overview that:
- Links to all generated reports
- Provides executive summary
- Shows generation metadata
- Enables easy navigation between reports

### Can I customize the file naming?

Yes, modify `output.yaml`:
```yaml
file_naming:
  timestamp_format: "%Y%m%d_%H%M%S"
  include_report_type: true
  custom_prefix: "company_analysis"
```

### How do I share reports with others?

**HTML reports:**
- Self-contained files that can be shared directly
- Open in any modern web browser
- Include all interactive features

**PDF reports:**
- Ideal for email distribution
- Professional formatting
- Print-ready

**CSV data:**
- Can be imported into Excel, R, or other analysis tools
- Contains raw data and statistical summaries

### Can I automate report generation?

Yes, several options:

**Batch processing:**
```bash
python -m src.main --batch-config automated_reports.json
```

**Scheduled execution (cron on Linux/macOS):**
```bash
# Add to crontab for daily reports at 2 AM
0 2 * * * /path/to/python -m src.main --generate-all --output-dir /path/to/reports
```

**Windows Task Scheduler:**
- Create a task to run the Python command
- Set desired schedule and parameters

## Technical Questions

### What Python libraries does this use?

**Core dependencies:**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib/plotly**: Visualization
- **pyyaml**: Configuration file handling
- **jinja2**: Template processing

**Optional dependencies:**
- **weasyprint**: PDF generation
- **psutil**: System monitoring
- **scipy**: Advanced statistical functions

### Is the code open source?

Yes, the application is open source. You can:
- View and modify the source code
- Contribute improvements
- Report issues and bugs
- Create custom extensions

### Can I integrate this with other systems?

The application is designed for command-line use but can be integrated:

**API integration (planned):**
```python
from occupation_reports import ReportAPI
api = ReportAPI()
result = api.generate_report('environmental_risk')
```

**Database integration:**
- Export CSV data to databases
- Use pandas to connect to SQL databases
- Integrate with data warehouses

**Web integration:**
- Embed HTML reports in web applications
- Use generated visualizations in dashboards
- Create web services around the core functionality

### How do I extend the application?

**Adding new analysis methods:**
1. Create new analyzer classes in `src/analysis/`
2. Register with the analysis engine
3. Update configuration schemas

**Adding new visualization types:**
1. Extend visualization classes in `src/visualization/`
2. Add new chart types and styling options
3. Update template systems

**Adding new export formats:**
1. Create new exporter classes in `src/export/`
2. Implement format-specific logic
3. Register with export manager

### What about data privacy and security?

**Data handling:**
- All processing is done locally on your system
- No data is sent to external servers
- You control all data access and storage

**Security considerations:**
- Validate input data sources
- Use secure file permissions
- Review configuration files for sensitive information
- Consider data anonymization for shared reports

## Research and Academic Use

### How do I cite this application in research?

**Application citation:**
```
Occupation Data Reports [Computer software]. (2023). 
Retrieved from [repository URL]
```

**Data source citation:**
```
U.S. Bureau of Labor Statistics. (2023). 
Occupational Requirements Survey. 
Retrieved from https://www.bls.gov/ors/
```

### Can I use this for academic research?

Yes, the application is suitable for academic research:
- Rigorous statistical methodology
- Comprehensive documentation
- Reproducible analysis
- Open source for transparency

**Research applications:**
- Labor economics studies
- Occupational health research
- Workforce development analysis
- Policy impact assessment

### How do I ensure reproducible results?

**Version control:**
- Document application version used
- Save configuration files with research data
- Use specific dataset versions

**Configuration management:**
```bash
# Save exact configuration
cp -r config/ research_config_v1/

# Document system information
python -m src.main status --verbose > system_info.txt
```

**Data documentation:**
- Record data source and download date
- Document any data preprocessing steps
- Save original and processed datasets

### What statistical methods are used?

**Confidence intervals:**
- Based on standard errors provided in ORS data
- Typically 95% confidence level (configurable)
- Uses normal distribution approximation

**Correlation analysis:**
- Pearson correlation coefficients
- Significance testing with p-values
- Correction for multiple comparisons

**Risk assessment:**
- Weighted scoring based on exposure percentages
- Standard error propagation
- Confidence interval calculation for risk scores

**Quality assessment:**
- Reliability scoring based on standard errors
- Footnote interpretation for data quality
- Coverage analysis for representativeness

## Troubleshooting Quick Answers

### "File not found" error
**Quick fix:** Ensure data files are in the working directory with exact names:
- `2023-complete-dataset.csv`
- `2023-complete-dataset-footnote-codes.csv`
- `2023-complete-dataset-field names and description.csv`

### "Configuration invalid" error
**Quick fix:** Recreate configuration files:
```bash
python -m src.main create-config
```

### Reports are empty
**Quick fix:** Check data quality and filters:
```bash
python -m src.main --report-type data_quality
```

### Memory error
**Quick fix:** Generate reports individually:
```bash
python -m src.main --report-type occupation_distribution
```

### Slow performance
**Quick fix:** Enable parallel processing:
```bash
python -m src.main --generate-all --parallel
```

### PDF generation fails
**Quick fix:** Install PDF dependencies:
```bash
pip install weasyprint
```

### Charts not displaying
**Quick fix:** Update visualization libraries:
```bash
pip install --upgrade plotly matplotlib
```

### Permission denied
**Quick fix:** Use different output directory:
```bash
python -m src.main --output-dir ~/reports
```

---

*This FAQ covers the most common questions about the Occupation Data Reports application. If you have a question not covered here, please check the user guide and troubleshooting documentation, or contact support.*