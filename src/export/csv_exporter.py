"""
CSV export functionality for occupation data reports.
Exports raw data and statistical summaries with proper headers, metadata, and filtering options.
"""

import os
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

from ..interfaces import ReportData, AnalysisResult, ExportError


class CSVExporter:
    """
    CSV exporter class for raw data and statistical summaries with proper headers,
    metadata, and data filtering capabilities.
    """
    
    def __init__(self, include_metadata: bool = True, decimal_places: int = 4):
        """
        Initialize the CSV exporter with configuration options.
        
        Args:
            include_metadata: Whether to include metadata in exported files
            decimal_places: Number of decimal places for numeric values
        """
        self.include_metadata = include_metadata
        self.decimal_places = decimal_places
        self.encoding = 'utf-8'
        
        # Standard column mappings for consistency
        self.standard_columns = {
            'occupation_category': 'Occupation Category',
            'metric_name': 'Metric Name',
            'value': 'Value',
            'confidence_interval_lower': 'CI Lower Bound',
            'confidence_interval_upper': 'CI Upper Bound',
            'reliability_score': 'Reliability Score',
            'footnote_context': 'Footnotes'
        }
    
    def export_csv_data(self, data: pd.DataFrame, output_path: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export DataFrame to CSV with proper formatting and metadata.
        
        Args:
            data: DataFrame to export
            output_path: Path where CSV file should be saved
            metadata: Optional metadata to include in the file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Format numeric columns
            formatted_data = self._format_numeric_columns(data.copy())
            
            # Write CSV file
            with open(output_path, 'w', newline='', encoding=self.encoding) as csvfile:
                # Write metadata header if requested
                if self.include_metadata and metadata:
                    self._write_metadata_header(csvfile, metadata)
                
                # Write data
                formatted_data.to_csv(csvfile, index=False, float_format=f'%.{self.decimal_places}f')
            
            return True
            
        except Exception as e:
            raise ExportError(f"Failed to export CSV data: {str(e)}")
    
    def export_analysis_results(self, analysis_results: List[AnalysisResult], 
                              output_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export analysis results to CSV format.
        
        Args:
            analysis_results: List of AnalysisResult objects
            output_path: Path where CSV file should be saved
            metadata: Optional metadata to include
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Convert analysis results to DataFrame
            data_rows = []
            for result in analysis_results:
                row = {
                    'Occupation Category': result.occupation_category,
                    'Metric Name': result.metric_name,
                    'Value': round(result.value, self.decimal_places),
                    'CI Lower Bound': round(result.confidence_interval[0], self.decimal_places),
                    'CI Upper Bound': round(result.confidence_interval[1], self.decimal_places),
                    'Reliability Score': round(result.reliability_score, self.decimal_places),
                    'Footnotes': '; '.join(result.footnote_context) if result.footnote_context else ''
                }
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            
            # Add metadata if not provided
            if metadata is None:
                metadata = {
                    'export_type': 'Analysis Results',
                    'total_results': len(analysis_results),
                    'export_timestamp': datetime.now().isoformat()
                }
            
            return self.export_csv_data(df, output_path, metadata)
            
        except Exception as e:
            raise ExportError(f"Failed to export analysis results to CSV: {str(e)}")
    
    def export_filtered_data(self, data: pd.DataFrame, output_path: str,
                           filters: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export filtered data based on specified criteria.
        
        Args:
            data: Source DataFrame
            output_path: Path where CSV file should be saved
            filters: Dictionary of column filters to apply
            metadata: Optional metadata to include
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Apply filters
            filtered_data = self._apply_filters(data.copy(), filters)
            
            # Add filter information to metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'filters_applied': filters,
                'original_rows': len(data),
                'filtered_rows': len(filtered_data),
                'filter_timestamp': datetime.now().isoformat()
            })
            
            return self.export_csv_data(filtered_data, output_path, metadata)
            
        except Exception as e:
            raise ExportError(f"Failed to export filtered data: {str(e)}")
    
    def export_summary_statistics(self, data: pd.DataFrame, output_path: str,
                                group_by: Optional[str] = None, 
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export summary statistics for numeric columns.
        
        Args:
            data: Source DataFrame
            output_path: Path where CSV file should be saved
            group_by: Optional column to group statistics by
            metadata: Optional metadata to include
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Select numeric columns
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                raise ExportError("No numeric columns found for summary statistics")
            
            # Calculate statistics
            if group_by and group_by in data.columns:
                # Grouped statistics
                summary_stats = data.groupby(group_by)[numeric_columns].agg([
                    'count', 'mean', 'std', 'min', 'max', 'median'
                ]).round(self.decimal_places)
                
                # Flatten column names
                summary_stats.columns = [f"{col[0]}_{col[1]}" for col in summary_stats.columns]
                summary_stats = summary_stats.reset_index()
                
            else:
                # Overall statistics
                stats_data = []
                for col in numeric_columns:
                    col_stats = {
                        'Column': col,
                        'Count': data[col].count(),
                        'Mean': round(data[col].mean(), self.decimal_places),
                        'Std Dev': round(data[col].std(), self.decimal_places),
                        'Min': round(data[col].min(), self.decimal_places),
                        'Max': round(data[col].max(), self.decimal_places),
                        'Median': round(data[col].median(), self.decimal_places)
                    }
                    stats_data.append(col_stats)
                
                summary_stats = pd.DataFrame(stats_data)
            
            # Add metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'export_type': 'Summary Statistics',
                'grouped_by': group_by if group_by else 'None',
                'numeric_columns': len(numeric_columns),
                'total_rows': len(data)
            })
            
            return self.export_csv_data(summary_stats, output_path, metadata)
            
        except Exception as e:
            raise ExportError(f"Failed to export summary statistics: {str(e)}")
    
    def export_correlation_matrix(self, data: pd.DataFrame, output_path: str,
                                columns: Optional[List[str]] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export correlation matrix for numeric columns.
        
        Args:
            data: Source DataFrame
            output_path: Path where CSV file should be saved
            columns: Optional list of columns to include in correlation
            metadata: Optional metadata to include
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            # Select columns for correlation
            if columns:
                correlation_data = data[columns]
            else:
                correlation_data = data.select_dtypes(include=[np.number])
            
            if correlation_data.empty:
                raise ExportError("No numeric data available for correlation analysis")
            
            # Calculate correlation matrix
            correlation_matrix = correlation_data.corr().round(self.decimal_places)
            
            # Add metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'export_type': 'Correlation Matrix',
                'variables_included': len(correlation_matrix.columns),
                'correlation_method': 'Pearson'
            })
            
            return self.export_csv_data(correlation_matrix, output_path, metadata)
            
        except Exception as e:
            raise ExportError(f"Failed to export correlation matrix: {str(e)}")
    
    def create_data_dictionary(self, data: pd.DataFrame, output_path: str,
                             column_descriptions: Optional[Dict[str, str]] = None) -> bool:
        """
        Create a data dictionary CSV describing the dataset structure.
        
        Args:
            data: Source DataFrame
            output_path: Path where data dictionary should be saved
            column_descriptions: Optional descriptions for each column
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            dictionary_data = []
            
            for column in data.columns:
                col_info = {
                    'Column Name': column,
                    'Data Type': str(data[column].dtype),
                    'Non-Null Count': data[column].count(),
                    'Null Count': data[column].isnull().sum(),
                    'Unique Values': data[column].nunique(),
                    'Description': column_descriptions.get(column, '') if column_descriptions else ''
                }
                
                # Add sample values for categorical columns
                if data[column].dtype == 'object':
                    unique_values = data[column].dropna().unique()[:5]  # First 5 unique values
                    col_info['Sample Values'] = '; '.join(str(v) for v in unique_values)
                else:
                    # Add range for numeric columns
                    if data[column].count() > 0:
                        col_info['Min Value'] = data[column].min()
                        col_info['Max Value'] = data[column].max()
                        col_info['Mean Value'] = round(data[column].mean(), self.decimal_places)
                
                dictionary_data.append(col_info)
            
            dictionary_df = pd.DataFrame(dictionary_data)
            
            metadata = {
                'export_type': 'Data Dictionary',
                'total_columns': len(data.columns),
                'total_rows': len(data),
                'creation_timestamp': datetime.now().isoformat()
            }
            
            return self.export_csv_data(dictionary_df, output_path, metadata)
            
        except Exception as e:
            raise ExportError(f"Failed to create data dictionary: {str(e)}")
    
    def _format_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Format numeric columns for consistent CSV output.
        
        Args:
            data: DataFrame to format
            
        Returns:
            Formatted DataFrame
        """
        for column in data.select_dtypes(include=[np.number]).columns:
            # Round to specified decimal places
            data[column] = data[column].round(self.decimal_places)
            
            # Replace NaN with empty string for cleaner CSV
            data[column] = data[column].fillna('')
        
        return data
    
    def _apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to DataFrame.
        
        Args:
            data: DataFrame to filter
            filters: Dictionary of filters to apply
            
        Returns:
            Filtered DataFrame
        """
        filtered_data = data.copy()
        
        for column, filter_value in filters.items():
            if column not in data.columns:
                continue
            
            if isinstance(filter_value, dict):
                # Range filter
                if 'min' in filter_value:
                    filtered_data = filtered_data[filtered_data[column] >= filter_value['min']]
                if 'max' in filter_value:
                    filtered_data = filtered_data[filtered_data[column] <= filter_value['max']]
            elif isinstance(filter_value, list):
                # Value list filter
                filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
            else:
                # Exact match filter
                filtered_data = filtered_data[filtered_data[column] == filter_value]
        
        return filtered_data
    
    def _write_metadata_header(self, csvfile, metadata: Dict[str, Any]) -> None:
        """
        Write metadata as comments at the top of CSV file.
        
        Args:
            csvfile: Open CSV file object
            metadata: Metadata dictionary to write
        """
        csvfile.write("# Occupation Data Report Export\n")
        csvfile.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        csvfile.write("#\n")
        
        for key, value in metadata.items():
            csvfile.write(f"# {key}: {value}\n")
        
        csvfile.write("#\n")
    
    def export_multiple_datasets(self, datasets: Dict[str, pd.DataFrame], 
                               output_dir: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export multiple datasets to separate CSV files in a directory.
        
        Args:
            datasets: Dictionary mapping dataset names to DataFrames
            output_dir: Directory where CSV files should be saved
            metadata: Optional metadata to include in all files
            
        Returns:
            True if all exports successful, False otherwise
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            success_count = 0
            
            for dataset_name, data in datasets.items():
                # Create safe filename
                safe_filename = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_filename = safe_filename.replace(' ', '_') + '.csv'
                
                file_path = output_path / safe_filename
                
                # Add dataset-specific metadata
                dataset_metadata = metadata.copy() if metadata else {}
                dataset_metadata.update({
                    'dataset_name': dataset_name,
                    'rows': len(data),
                    'columns': len(data.columns)
                })
                
                if self.export_csv_data(data, str(file_path), dataset_metadata):
                    success_count += 1
            
            return success_count == len(datasets)
            
        except Exception as e:
            raise ExportError(f"Failed to export multiple datasets: {str(e)}")
    
    def create_export_manifest(self, exported_files: List[str], output_path: str) -> bool:
        """
        Create a manifest file listing all exported CSV files.
        
        Args:
            exported_files: List of exported file paths
            output_path: Path where manifest should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            manifest_data = []
            
            for file_path in exported_files:
                file_info = {
                    'File Name': Path(file_path).name,
                    'File Path': file_path,
                    'File Size (bytes)': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    'Export Timestamp': datetime.now().isoformat()
                }
                manifest_data.append(file_info)
            
            manifest_df = pd.DataFrame(manifest_data)
            
            metadata = {
                'export_type': 'Export Manifest',
                'total_files': len(exported_files),
                'manifest_created': datetime.now().isoformat()
            }
            
            return self.export_csv_data(manifest_df, output_path, metadata)
            
        except Exception as e:
            raise ExportError(f"Failed to create export manifest: {str(e)}")