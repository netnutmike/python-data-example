# Troubleshooting Guide

## Table of Contents

1. [Quick Diagnostic Steps](#quick-diagnostic-steps)
2. [Installation Issues](#installation-issues)
3. [Data Loading Problems](#data-loading-problems)
4. [Configuration Errors](#configuration-errors)
5. [Report Generation Issues](#report-generation-issues)
6. [Performance Problems](#performance-problems)
7. [Output and Export Issues](#output-and-export-issues)
8. [System-Specific Issues](#system-specific-issues)
9. [Error Messages Reference](#error-messages-reference)
10. [Getting Additional Help](#getting-additional-help)

## Quick Diagnostic Steps

Before diving into specific issues, run these diagnostic commands to identify the problem area:

### 1. System Status Check
```bash
python -m src.main status --verbose
```
This shows:
- Configuration validation status
- Data file availability
- System resources
- Performance metrics

### 2. Configuration Validation
```bash
python -m src.main --validate-config
```
This identifies:
- Invalid configuration files
- Missing required settings
- File path issues

### 3. Pipeline Test
```bash
python -m src.main test-pipeline
```
This tests:
- Data loading capabilities
- Basic processing functionality
- System readiness

### 4. Dry Run
```bash
python -m src.main --generate-all --dry-run
```
This shows what would be executed without actually running it.

## Installation Issues

### Python Version Compatibility

**Problem**: Application fails to start with Python version errors.

**Symptoms**:
```
SyntaxError: invalid syntax
TypeError: unsupported operand type(s)
```

**Solutions**:
1. **Check Python version**:
   ```bash
   python --version
   # Should be 3.9 or higher
   ```

2. **Use correct Python executable**:
   ```bash
   python3.9 -m src.main --help
   # or
   python3 -m src.main --help
   ```

3. **Update Python** if version is too old:
   - Windows: Download from python.org
   - macOS: `brew install python@3.9`
   - Linux: `sudo apt install python3.9`

### Dependency Installation Failures

**Problem**: `pip install -r requirements.txt` fails.

**Common Errors**:
```
ERROR: Could not find a version that satisfies the requirement
ERROR: Failed building wheel for [package]
Microsoft Visual C++ 14.0 is required
```

**Solutions**:

1. **Update pip and setuptools**:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Install system dependencies** (Linux):
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev build-essential
   ```

3. **Install Visual C++ Build Tools** (Windows):
   - Download from Microsoft Visual Studio website
   - Or install Visual Studio Community

4. **Use conda instead of pip** (if available):
   ```bash
   conda create -n occupation-reports python=3.9
   conda activate occupation-reports
   conda install --file requirements.txt
   ```

5. **Install packages individually**:
   ```bash
   pip install pandas numpy matplotlib plotly
   pip install pyyaml jinja2 weasyprint
   # Continue with other packages
   ```

### Virtual Environment Issues

**Problem**: Package conflicts or permission errors.

**Solutions**:

1. **Create fresh virtual environment**:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate  # Linux/macOS
   fresh_env\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

2. **Clear pip cache**:
   ```bash
   pip cache purge
   ```

3. **Use --user flag** (if virtual environment isn't working):
   ```bash
   pip install --user -r requirements.txt
   ```

### Package Installation

**Problem**: `pip install -e .` fails.

**Solutions**:

1. **Check setup.py exists** in the project root
2. **Install in development mode**:
   ```bash
   pip install -e . --verbose
   ```
3. **Install without development mode**:
   ```bash
   pip install .
   ```

## Data Loading Problems

### File Not Found Errors

**Problem**: Application cannot find data files.

**Error Messages**:
```
FileNotFoundError: [Errno 2] No such file or directory: '2023-complete-dataset.csv'
Data file not found: 2023-complete-dataset.csv
```

**Solutions**:

1. **Verify file names exactly match**:
   - `2023-complete-dataset.csv`
   - `2023-complete-dataset-footnote-codes.csv`
   - `2023-complete-dataset-field names and description.csv`

2. **Check file location**:
   ```bash
   ls -la *.csv
   # Files should be in working directory or specified path
   ```

3. **Use absolute paths**:
   ```bash
   python -m src.main --dataset-file /full/path/to/2023-complete-dataset.csv
   ```

4. **Specify data directory**:
   ```bash
   python -m src.main --data-dir /path/to/data/folder
   ```

5. **Check file permissions**:
   ```bash
   ls -la 2023-complete-dataset.csv
   # Should show read permissions for your user
   ```

### File Format Issues

**Problem**: Data files are corrupted or in wrong format.

**Symptoms**:
```
UnicodeDecodeError: 'utf-8' codec can't decode
pandas.errors.EmptyDataError: No columns to parse
Expected 18 columns, found 15
```

**Solutions**:

1. **Check file encoding**:
   ```bash
   file 2023-complete-dataset.csv
   # Should show text file with UTF-8 encoding
   ```

2. **Try different encoding** in configuration:
   ```yaml
   # In data_sources.yaml
   encoding: "latin-1"  # or "cp1252"
   ```

3. **Verify file integrity**:
   ```bash
   wc -l 2023-complete-dataset.csv
   # Should show ~148,600 lines
   
   head -n 5 2023-complete-dataset.csv
   # Should show proper CSV header and data
   ```

4. **Re-download files** from BLS if corrupted

5. **Check for Excel format**:
   - If you have `.xlsx` files, convert to CSV first
   - Use "Save As" → "CSV (Comma delimited)"

### Column Validation Failures

**Problem**: Dataset doesn't have expected columns.

**Error Messages**:
```
Missing required columns: ['SOC 2018 code', 'Estimate']
Column validation failed: Expected 18 columns, found 15
```

**Solutions**:

1. **Check column names** in your CSV file:
   ```bash
   head -n 1 2023-complete-dataset.csv
   ```

2. **Compare with expected columns**:
   - Series ID
   - Series title
   - SOC 2018 code
   - Occupation
   - Requirement
   - Estimate code
   - Estimate text
   - Category code
   - Category
   - Additive code
   - Additive
   - Datatype code
   - Datatype
   - Estimate
   - Standard error
   - Data footnote
   - Standard error footnote
   - Series footnote

3. **Update field mapping** if column names differ:
   ```yaml
   # In data_sources.yaml
   column_mapping:
     "SOC Code": "SOC 2018 code"
     "Est": "Estimate"
   ```

## Configuration Errors

### YAML Syntax Errors

**Problem**: Configuration files have invalid YAML syntax.

**Error Messages**:
```
yaml.scanner.ScannerError: mapping values are not allowed here
yaml.parser.ParserError: expected <block end>
```

**Solutions**:

1. **Check indentation** (use spaces, not tabs):
   ```yaml
   # Correct
   reports:
     occupation_distribution:
       enabled: true
   
   # Incorrect (mixed tabs/spaces)
   reports:
   	occupation_distribution:
       enabled: true
   ```

2. **Validate YAML syntax** online:
   - Copy configuration to yamllint.com
   - Fix syntax errors identified

3. **Recreate configuration files**:
   ```bash
   mv config config_backup
   python -m src.main create-config
   ```

4. **Check for special characters**:
   - Wrap strings with special characters in quotes
   - Escape backslashes: `"C:\\path\\to\\file"`

### Missing Configuration Files

**Problem**: Configuration files don't exist.

**Solutions**:

1. **Create default configuration**:
   ```bash
   python -m src.main create-config
   ```

2. **Specify custom config directory**:
   ```bash
   python -m src.main create-config --config-dir ./my_config
   python -m src.main --config-dir ./my_config --list-reports
   ```

### Invalid File Paths

**Problem**: Configuration contains invalid file paths.

**Error Messages**:
```
Path does not exist: /invalid/path/to/data.csv
Permission denied: /restricted/path/
```

**Solutions**:

1. **Use absolute paths**:
   ```yaml
   # In data_sources.yaml
   main_dataset_path: "/home/user/data/2023-complete-dataset.csv"
   ```

2. **Use relative paths from working directory**:
   ```yaml
   main_dataset_path: "./data/2023-complete-dataset.csv"
   ```

3. **Check path permissions**:
   ```bash
   ls -la /path/to/file
   # Ensure read permissions
   ```

4. **Use forward slashes** even on Windows:
   ```yaml
   main_dataset_path: "C:/Users/username/data/dataset.csv"
   ```

## Report Generation Issues

### Empty Reports

**Problem**: Reports generate but contain no data or visualizations.

**Possible Causes**:
- Data filtering excludes all records
- Footnote codes mark all estimates as unreliable
- Insufficient data meets report criteria

**Solutions**:

1. **Check data filtering**:
   ```yaml
   # In reports.yaml - reduce restrictive filters
   environmental_risk:
     custom_parameters:
       risk_threshold: 0.5  # Lower threshold
       min_sample_size: 10  # Reduce minimum
   ```

2. **Review footnote interpretation**:
   ```bash
   python -m src.main --report-type data_quality
   # This shows data reliability by occupation
   ```

3. **Enable verbose logging**:
   ```bash
   python -m src.main --report-type occupation_distribution --verbose
   ```

4. **Check raw data**:
   ```bash
   python -m src.main --report-type occupation_distribution --output-format csv
   # Review the CSV output for data availability
   ```

### Visualization Errors

**Problem**: Charts fail to generate or display incorrectly.

**Error Messages**:
```
PlotlyError: Invalid property specified
ValueError: x and y must have same first dimension
```

**Solutions**:

1. **Update visualization libraries**:
   ```bash
   pip install --upgrade plotly matplotlib seaborn
   ```

2. **Check data types**:
   - Ensure numeric columns are properly formatted
   - Verify date columns are parsed correctly

3. **Reduce chart complexity**:
   ```yaml
   # In visualization.yaml
   max_categories: 20  # Reduce from higher number
   simplify_charts: true
   ```

4. **Use alternative chart types**:
   ```yaml
   # In reports.yaml
   occupation_distribution:
     custom_parameters:
       chart_type: "bar"  # Instead of complex interactive charts
   ```

### Memory Errors During Report Generation

**Problem**: Application runs out of memory.

**Error Messages**:
```
MemoryError: Unable to allocate array
OverflowError: Python int too large to convert
```

**Solutions**:

1. **Generate reports individually**:
   ```bash
   python -m src.main --report-type occupation_distribution
   python -m src.main --report-type environmental_risk
   # Instead of --generate-all
   ```

2. **Reduce data processing**:
   ```yaml
   # In analysis.yaml
   max_records: 50000  # Process subset of data
   sample_data: true   # Use sampling for large datasets
   ```

3. **Increase system memory**:
   - Close other applications
   - Add swap space (Linux/macOS)
   - Use a machine with more RAM

4. **Use CSV output only**:
   ```bash
   python -m src.main --report-type correlation_analysis --output-format csv
   ```

### Export Format Failures

**Problem**: Specific export formats fail to generate.

**Common Issues**:

1. **PDF Export Fails**:
   ```bash
   # Install additional dependencies
   pip install weasyprint
   
   # On Linux, install system dependencies
   sudo apt-get install python3-cffi python3-brotli libpango-1.0-0 libharfbuzz0b libpangoft2-1.0-0
   ```

2. **HTML Export Issues**:
   ```bash
   # Update web dependencies
   pip install --upgrade jinja2 plotly
   ```

3. **CSV Export Problems**:
   ```bash
   # Check pandas version
   pip install --upgrade pandas
   ```

## Performance Problems

### Slow Report Generation

**Problem**: Reports take too long to generate.

**Solutions**:

1. **Enable parallel processing**:
   ```bash
   python -m src.main --generate-all --parallel --max-workers 4
   ```

2. **Optimize system resources**:
   ```bash
   # Check current resource usage
   python -m src.main status --verbose
   ```

3. **Reduce analysis complexity**:
   ```yaml
   # In analysis.yaml
   confidence_level: 0.95  # Instead of 0.99
   max_iterations: 1000    # Reduce from higher number
   detailed_analysis: false
   ```

4. **Use data sampling**:
   ```yaml
   # In data_sources.yaml
   sample_size: 10000  # Process subset for testing
   ```

### High Memory Usage

**Problem**: Application uses too much RAM.

**Solutions**:

1. **Monitor memory usage**:
   ```bash
   python -m src.main status --verbose
   # Shows current memory usage
   ```

2. **Process reports sequentially**:
   ```bash
   # Instead of parallel processing
   python -m src.main --generate-all --max-workers 1
   ```

3. **Clear cache between reports**:
   ```bash
   # Remove temporary files
   rm -rf temp/ cache/
   ```

4. **Optimize data loading**:
   ```yaml
   # In data_sources.yaml
   chunk_size: 1000      # Process data in smaller chunks
   lazy_loading: true    # Load data on demand
   ```

### Disk Space Issues

**Problem**: Insufficient disk space for reports.

**Solutions**:

1. **Check available space**:
   ```bash
   df -h .
   # Shows available disk space
   ```

2. **Clean up old reports**:
   ```bash
   # Remove old timestamped directories
   rm -rf reports/202310*
   ```

3. **Use compressed output**:
   ```yaml
   # In output.yaml
   compress_output: true
   cleanup_temp_files: true
   ```

4. **Generate specific reports only**:
   ```bash
   python -m src.main --report-type occupation_distribution
   # Instead of generating all reports
   ```

## Output and Export Issues

### File Permission Errors

**Problem**: Cannot write to output directory.

**Error Messages**:
```
PermissionError: [Errno 13] Permission denied: 'reports/'
OSError: [Errno 30] Read-only file system
```

**Solutions**:

1. **Check directory permissions**:
   ```bash
   ls -la reports/
   # Should show write permissions
   ```

2. **Create output directory**:
   ```bash
   mkdir -p reports
   chmod 755 reports
   ```

3. **Use different output directory**:
   ```bash
   python -m src.main --output-dir ~/my_reports
   ```

4. **Run with appropriate permissions**:
   ```bash
   # On shared systems, ensure you have write access
   python -m src.main --output-dir /tmp/reports
   ```

### Corrupted Output Files

**Problem**: Generated files are corrupted or incomplete.

**Solutions**:

1. **Check disk space** during generation:
   ```bash
   df -h .
   ```

2. **Verify file integrity**:
   ```bash
   file reports/*/occupation_distribution.html
   # Should show HTML document
   ```

3. **Regenerate with verbose logging**:
   ```bash
   python -m src.main --report-type occupation_distribution --verbose
   ```

4. **Use different export format**:
   ```bash
   python -m src.main --report-type environmental_risk --output-format csv
   ```

### Missing Visualizations

**Problem**: Reports generate but charts are missing.

**Solutions**:

1. **Check JavaScript console** (for HTML reports):
   - Open HTML file in browser
   - Press F12 → Console tab
   - Look for JavaScript errors

2. **Update visualization libraries**:
   ```bash
   pip install --upgrade plotly matplotlib seaborn
   ```

3. **Enable static charts**:
   ```yaml
   # In visualization.yaml
   use_static_charts: true  # Fallback to static images
   ```

4. **Check chart data**:
   ```bash
   python -m src.main --report-type correlation_analysis --output-format csv
   # Verify data exists for visualization
   ```

## System-Specific Issues

### Windows-Specific Problems

1. **Path Length Limitations**:
   ```bash
   # Use shorter paths
   python -m src.main --output-dir C:\reports
   ```

2. **PowerShell Execution Policy**:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Antivirus Interference**:
   - Add project directory to antivirus exclusions
   - Temporarily disable real-time scanning

### macOS-Specific Problems

1. **Homebrew Python Issues**:
   ```bash
   # Use system Python or pyenv
   /usr/bin/python3 -m src.main --help
   ```

2. **Permission Issues**:
   ```bash
   # Don't use sudo with pip
   pip install --user -r requirements.txt
   ```

3. **SSL Certificate Errors**:
   ```bash
   # Update certificates
   /Applications/Python\ 3.9/Install\ Certificates.command
   ```

### Linux-Specific Problems

1. **Missing System Dependencies**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev python3-pip build-essential
   
   # CentOS/RHEL
   sudo yum install python3-devel python3-pip gcc
   ```

2. **Display Issues** (headless servers):
   ```bash
   # Set display backend for matplotlib
   export MPLBACKEND=Agg
   python -m src.main --report-type occupation_distribution
   ```

## Error Messages Reference

### Common Error Patterns

#### Configuration Errors
```
ConfigurationError: Invalid configuration file
→ Run: python -m src.main --validate-config

ValidationError: Missing required field
→ Check configuration file syntax and required fields

FileNotFoundError: Configuration file not found
→ Run: python -m src.main create-config
```

#### Data Loading Errors
```
DataLoadingError: Failed to load dataset
→ Check file path and permissions

ColumnValidationError: Missing required columns
→ Verify CSV file format and column names

FootnoteProcessingError: Invalid footnote code
→ Check footnote reference file
```

#### Processing Errors
```
AnalysisError: Insufficient data for analysis
→ Check data filters and availability

StatisticalError: Cannot calculate confidence intervals
→ Verify standard error values in data

CorrelationError: Invalid correlation matrix
→ Check for missing or infinite values
```

#### Export Errors
```
ExportError: Failed to generate HTML report
→ Check template files and dependencies

PDFGenerationError: WeasyPrint failed
→ Install PDF dependencies

VisualizationError: Chart generation failed
→ Update plotting libraries
```

### Exit Codes

- **0**: Success
- **1**: General error
- **2**: Configuration error
- **3**: Data loading error
- **4**: Processing error
- **5**: Export error
- **130**: Interrupted by user (Ctrl+C)

## Getting Additional Help

### Before Seeking Help

1. **Run diagnostic commands**:
   ```bash
   python -m src.main status --verbose > system_info.txt
   python -m src.main --validate-config > config_validation.txt
   python -m src.main test-pipeline > pipeline_test.txt
   ```

2. **Collect log files**:
   ```bash
   # Copy recent log files
   cp logs/*.log ./
   ```

3. **Document your environment**:
   ```bash
   python --version > environment_info.txt
   pip list >> environment_info.txt
   uname -a >> environment_info.txt  # Linux/macOS
   ```

### Information to Include

When seeking help, provide:

1. **Error message** (complete text)
2. **Command that failed** (exact command line)
3. **System information** (OS, Python version)
4. **Log files** (application.log, error.log)
5. **Configuration files** (remove sensitive data)
6. **Data file information** (size, format, source)
7. **Steps to reproduce** the issue

### Support Channels

1. **Documentation**: Check this guide and user guide first
2. **Project Repository**: Open an issue with detailed information
3. **Community Forums**: Search for similar issues
4. **Direct Contact**: Use provided contact information

### Self-Help Resources

1. **Built-in Help**:
   ```bash
   python -m src.main --help
   python -m src.main status --help
   ```

2. **Configuration Examples**:
   ```bash
   python -m src.main export-sample-config
   ```

3. **Test Environment**:
   ```bash
   python -m src.main test-pipeline
   ```

4. **Verbose Output**:
   ```bash
   python -m src.main --report-type occupation_distribution --verbose
   ```

---

*This troubleshooting guide covers the most common issues encountered when using the Occupation Data Reports application. If you encounter an issue not covered here, please follow the "Getting Additional Help" section to report it.*