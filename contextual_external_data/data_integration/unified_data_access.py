#!/usr/bin/env python3
"""
Unified Data Access Layer
Provides a single interface to access all contextual and external data
for the Dynamic Digital Twin Framework
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import sqlite3
from pathlib import Path

class ContextualDataManager:
    """
    Unified interface for accessing all contextual and external data
    """
    
    def __init__(self, data_root_path: str = "contextual_external_data"):
        self.data_root = Path(data_root_path)
        self.db_path = self.data_root / "unified_database.db"
        self._initialize_database()
        
        # Data category mappings
        self.data_categories = {
            'weather_climate': {
                'tmy_data': 'weather_climate/tmy_data_*.csv',
                'climate_projections': 'weather_climate/projections_*/climate_projection_*.csv'
            },
            'economic_market': {
                'energy_prices': 'economic_market/energy_prices_*/historical_prices_*.csv',
                'material_costs': 'economic_market/material_technology_costs/detailed_material_costs_*.csv',
                'labor_costs': 'economic_market/labor_costs_*/trade_labor_rates_*.csv',
                'financial_parameters': 'economic_market/financial_parameters/discount_rate_scenarios_*.csv'
            },
            'geospatial_regulatory': {
                'location_data': 'geospatial_regulatory/location_data/city_location_data.csv',
                'carbon_intensity': 'geospatial_regulatory/carbon_intensity/*/hourly_carbon_intensity_*.csv',
                'building_codes': 'geospatial_regulatory/building_codes/iecc_2021_requirements.csv'
            }
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for fast data access"""
        
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.data_root}")
        
        # Create database connection
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                subcategory TEXT NOT NULL,
                file_path TEXT NOT NULL,
                location TEXT,
                year INTEGER,
                data_type TEXT,
                last_updated TIMESTAMP,
                file_size INTEGER,
                record_count INTEGER,
                metadata TEXT
            )
        """)
        
        self.conn.commit()
    
    def refresh_data_catalog(self):
        """Refresh the data catalog by scanning all data files"""
        
        print("Refreshing data catalog...")
        
        # Clear existing catalog
        self.conn.execute("DELETE FROM data_catalog")
        
        catalog_entries = []
        
        # Scan all data directories
        for category, subcategories in self.data_categories.items():
            category_path = self.data_root / category
            
            if not category_path.exists():
                continue
            
            for subcategory, pattern in subcategories.items():
                # Find all matching files
                for file_path in category_path.rglob("*.csv"):
                    if file_path.is_file():
                        # Extract metadata from file
                        metadata = self._extract_file_metadata(file_path, category, subcategory)
                        catalog_entries.append(metadata)
                
                # Also scan JSON files
                for file_path in category_path.rglob("*.json"):
                    if file_path.is_file():
                        metadata = self._extract_file_metadata(file_path, category, subcategory)
                        catalog_entries.append(metadata)
        
        # Insert catalog entries
        if catalog_entries:
            self.conn.executemany("""
                INSERT INTO data_catalog 
                (category, subcategory, file_path, location, year, data_type, 
                 last_updated, file_size, record_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, catalog_entries)
            
            self.conn.commit()
        
        print(f"Data catalog refreshed with {len(catalog_entries)} entries")
    
    def _extract_file_metadata(self, file_path: Path, category: str, subcategory: str) -> tuple:
        """Extract metadata from a data file"""
        
        try:
            # Basic file info
            stat = file_path.stat()
            file_size = stat.st_size
            last_updated = datetime.fromtimestamp(stat.st_mtime)
            
            # Extract location and year from filename
            filename = file_path.stem
            location = self._extract_location_from_filename(filename)
            year = self._extract_year_from_filename(filename)
            
            # Determine data type
            data_type = file_path.suffix[1:]  # Remove the dot
            
            # Count records for CSV files
            record_count = 0
            if data_type == 'csv':
                try:
                    df = pd.read_csv(file_path, nrows=0)  # Just read headers
                    with open(file_path, 'r') as f:
                        record_count = sum(1 for line in f) - 1  # Subtract header
                except:
                    record_count = 0
            
            # Create metadata JSON
            metadata = {
                'filename': filename,
                'relative_path': str(file_path.relative_to(self.data_root)),
                'columns': self._get_csv_columns(file_path) if data_type == 'csv' else None
            }
            
            return (
                category, subcategory, str(file_path), location, year,
                data_type, last_updated, file_size, record_count,
                json.dumps(metadata)
            )
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return (
                category, subcategory, str(file_path), None, None,
                'unknown', datetime.now(), 0, 0, '{}'
            )
    
    def _extract_location_from_filename(self, filename: str) -> Optional[str]:
        """Extract location from filename"""
        
        # Common location patterns
        locations = [
            'new_york', 'los_angeles', 'chicago', 'houston', 'phoenix',
            'philadelphia', 'san_antonio', 'san_diego', 'dallas', 'san_jose',
            'austin', 'jacksonville', 'fort_worth', 'columbus', 'charlotte',
            'san_francisco', 'indianapolis', 'seattle', 'denver', 'boston',
            'pjm', 'caiso', 'ercot', 'nyiso', 'iso_ne', 'miso'
        ]
        
        filename_lower = filename.lower()
        for location in locations:
            if location in filename_lower:
                return location.replace('_', ' ').title()
        
        return None
    
    def _extract_year_from_filename(self, filename: str) -> Optional[int]:
        """Extract year from filename"""
        
        import re
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            return int(year_match.group(1))
        
        return None
    
    def _get_csv_columns(self, file_path: Path) -> Optional[List[str]]:
        """Get column names from CSV file"""
        
        try:
            df = pd.read_csv(file_path, nrows=0)
            return df.columns.tolist()
        except:
            return None
    
    def get_weather_data(self, location: str, year: int = 2023, 
                        data_type: str = 'tmy') -> Optional[pd.DataFrame]:
        """Get weather data for a specific location and year"""
        
        # Query catalog for weather data
        query = """
            SELECT file_path FROM data_catalog 
            WHERE category = 'weather_climate' 
            AND (location LIKE ? OR file_path LIKE ?)
            AND (year = ? OR year IS NULL)
            AND data_type = 'csv'
        """
        
        location_pattern = f"%{location.lower().replace(' ', '_')}%"
        
        cursor = self.conn.execute(query, (location_pattern, location_pattern, year))
        results = cursor.fetchall()
        
        if not results:
            print(f"No weather data found for {location}, {year}")
            return None
        
        # Load the first matching file
        file_path = results[0][0]
        try:
            df = pd.read_csv(file_path)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except Exception as e:
            print(f"Error loading weather data from {file_path}: {e}")
            return None
    
    def get_energy_prices(self, location: str, price_type: str = 'historical') -> Optional[pd.DataFrame]:
        """Get energy pricing data for a location"""
        
        query = """
            SELECT file_path FROM data_catalog 
            WHERE category = 'economic_market' 
            AND subcategory = 'energy_prices'
            AND (location LIKE ? OR file_path LIKE ?)
            AND file_path LIKE ?
            AND data_type = 'csv'
        """
        
        location_pattern = f"%{location.lower().replace(' ', '_')}%"
        price_pattern = f"%{price_type}%"
        
        cursor = self.conn.execute(query, (location_pattern, location_pattern, price_pattern))
        results = cursor.fetchall()
        
        if not results:
            print(f"No energy price data found for {location}")
            return None
        
        file_path = results[0][0]
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading energy price data from {file_path}: {e}")
            return None
    
    def get_carbon_intensity(self, region: str, year: int = 2023) -> Optional[pd.DataFrame]:
        """Get carbon intensity data for a region"""
        
        query = """
            SELECT file_path FROM data_catalog 
            WHERE category = 'geospatial_regulatory' 
            AND subcategory = 'carbon_intensity'
            AND (location LIKE ? OR file_path LIKE ?)
            AND year = ?
            AND data_type = 'csv'
        """
        
        region_pattern = f"%{region.lower()}%"
        
        cursor = self.conn.execute(query, (region_pattern, region_pattern, year))
        results = cursor.fetchall()
        
        if not results:
            print(f"No carbon intensity data found for {region}, {year}")
            return None
        
        file_path = results[0][0]
        try:
            df = pd.read_csv(file_path)
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except Exception as e:
            print(f"Error loading carbon intensity data from {file_path}: {e}")
            return None
    
    def get_building_codes(self, climate_zone: str = None, 
                          state: str = None) -> Optional[pd.DataFrame]:
        """Get building code requirements"""
        
        query = """
            SELECT file_path FROM data_catalog 
            WHERE category = 'geospatial_regulatory' 
            AND subcategory = 'building_codes'
            AND data_type = 'csv'
        """
        
        cursor = self.conn.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("No building code data found")
            return None
        
        file_path = results[0][0]
        try:
            df = pd.read_csv(file_path)
            
            # Filter by climate zone or state if specified
            if climate_zone:
                df = df[df['climate_zone'] == climate_zone]
            if state:
                df = df[df['applicable_states'].str.contains(state, na=False)]
            
            return df
        except Exception as e:
            print(f"Error loading building code data from {file_path}: {e}")
            return None
    
    def get_material_costs(self, year: int = 2023) -> Optional[pd.DataFrame]:
        """Get material and technology costs"""
        
        query = """
            SELECT file_path FROM data_catalog 
            WHERE category = 'economic_market' 
            AND subcategory = 'material_costs'
            AND year = ?
            AND data_type = 'csv'
        """
        
        cursor = self.conn.execute(query, (year,))
        results = cursor.fetchall()
        
        if not results:
            print(f"No material cost data found for {year}")
            return None
        
        file_path = results[0][0]
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading material cost data from {file_path}: {e}")
            return None
    
    def get_location_data(self, city: str = None) -> Optional[pd.DataFrame]:
        """Get location and geospatial data"""
        
        query = """
            SELECT file_path FROM data_catalog 
            WHERE category = 'geospatial_regulatory' 
            AND subcategory = 'location_data'
            AND data_type = 'csv'
        """
        
        cursor = self.conn.execute(query)
        results = cursor.fetchall()
        
        if not results:
            print("No location data found")
            return None
        
        file_path = results[0][0]
        try:
            df = pd.read_csv(file_path)
            
            if city:
                df = df[df['city_name'].str.contains(city, case=False, na=False)]
            
            return df
        except Exception as e:
            print(f"Error loading location data from {file_path}: {e}")
            return None
    
    def search_data(self, category: str = None, subcategory: str = None,
                   location: str = None, year: int = None) -> pd.DataFrame:
        """Search for data files matching criteria"""
        
        query = "SELECT * FROM data_catalog WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if subcategory:
            query += " AND subcategory = ?"
            params.append(subcategory)
        
        if location:
            query += " AND (location LIKE ? OR file_path LIKE ?)"
            location_pattern = f"%{location.lower()}%"
            params.extend([location_pattern, location_pattern])
        
        if year:
            query += " AND year = ?"
            params.append(year)
        
        query += " ORDER BY category, subcategory, location, year"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of available data"""
        
        # Overall statistics
        total_files = self.conn.execute("SELECT COUNT(*) FROM data_catalog").fetchone()[0]
        total_size = self.conn.execute("SELECT SUM(file_size) FROM data_catalog").fetchone()[0] or 0
        total_records = self.conn.execute("SELECT SUM(record_count) FROM data_catalog").fetchone()[0] or 0
        
        # By category
        category_stats = pd.read_sql_query("""
            SELECT category, COUNT(*) as file_count, 
                   SUM(file_size) as total_size,
                   SUM(record_count) as total_records
            FROM data_catalog 
            GROUP BY category
        """, self.conn)
        
        # By location
        location_stats = pd.read_sql_query("""
            SELECT location, COUNT(*) as file_count
            FROM data_catalog 
            WHERE location IS NOT NULL
            GROUP BY location
            ORDER BY file_count DESC
        """, self.conn)
        
        # By year
        year_stats = pd.read_sql_query("""
            SELECT year, COUNT(*) as file_count
            FROM data_catalog 
            WHERE year IS NOT NULL
            GROUP BY year
            ORDER BY year
        """, self.conn)
        
        return {
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_records': total_records,
            'categories': category_stats.to_dict('records'),
            'locations': location_stats.to_dict('records'),
            'years': year_stats.to_dict('records'),
            'last_updated': datetime.now().isoformat()
        }
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity across all files"""
        
        validation_results = {
            'total_files_checked': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'errors': [],
            'warnings': []
        }
        
        # Get all files from catalog
        cursor = self.conn.execute("SELECT file_path, category, subcategory FROM data_catalog")
        files = cursor.fetchall()
        
        for file_path, category, subcategory in files:
            validation_results['total_files_checked'] += 1
            
            try:
                # Check if file exists
                if not os.path.exists(file_path):
                    validation_results['errors'].append(f"File not found: {file_path}")
                    validation_results['invalid_files'] += 1
                    continue
                
                # Validate CSV files
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    
                    # Check for empty files
                    if len(df) == 0:
                        validation_results['warnings'].append(f"Empty file: {file_path}")
                    
                    # Check for required columns based on category
                    required_columns = self._get_required_columns(category, subcategory)
                    missing_columns = set(required_columns) - set(df.columns)
                    
                    if missing_columns:
                        validation_results['errors'].append(
                            f"Missing columns in {file_path}: {missing_columns}"
                        )
                        validation_results['invalid_files'] += 1
                        continue
                
                validation_results['valid_files'] += 1
                
            except Exception as e:
                validation_results['errors'].append(f"Error validating {file_path}: {str(e)}")
                validation_results['invalid_files'] += 1
        
        return validation_results
    
    def _get_required_columns(self, category: str, subcategory: str) -> List[str]:
        """Get required columns for a data category"""
        
        required_columns = {
            'weather_climate': {
                'tmy_data': ['datetime', 'dry_bulb_temp_c', 'relative_humidity'],
                'climate_projections': ['datetime', 'scenario', 'dry_bulb_temp_c']
            },
            'economic_market': {
                'energy_prices': ['year', 'customer_type', 'electricity_price_kwh'],
                'material_costs': ['year', 'category', 'material_type', 'cost_per_unit'],
                'labor_costs': ['year', 'region', 'trade', 'base_rate_union']
            },
            'geospatial_regulatory': {
                'location_data': ['city_name', 'latitude', 'longitude', 'climate_zone'],
                'carbon_intensity': ['datetime', 'region', 'average_intensity'],
                'building_codes': ['climate_zone', 'building_type']
            }
        }
        
        return required_columns.get(category, {}).get(subcategory, [])
    
    def export_unified_dataset(self, output_path: str, format: str = 'sqlite'):
        """Export all data as a unified dataset"""
        
        if format == 'sqlite':
            # Copy current database
            import shutil
            shutil.copy2(self.db_path, output_path)
            
        elif format == 'excel':
            # Create Excel file with multiple sheets
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Data catalog
                catalog_df = pd.read_sql_query("SELECT * FROM data_catalog", self.conn)
                catalog_df.to_excel(writer, sheet_name='Data_Catalog', index=False)
                
                # Summary statistics
                summary = self.get_data_summary()
                pd.DataFrame([summary]).to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"Unified dataset exported to {output_path}")
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

def main():
    """Demonstrate the unified data access layer"""
    
    print("Initializing Contextual Data Manager...")
    
    # Initialize data manager
    data_manager = ContextualDataManager()
    
    # Refresh data catalog
    data_manager.refresh_data_catalog()
    
    # Get data summary
    summary = data_manager.get_data_summary()
    print(f"\nData Summary:")
    print(f"Total files: {summary['total_files']}")
    print(f"Total size: {summary['total_size_bytes'] / 1024 / 1024:.1f} MB")
    print(f"Total records: {summary['total_records']:,}")
    
    # Demonstrate data access
    print("\nTesting data access methods...")
    
    # Get weather data
    weather_data = data_manager.get_weather_data("New York", 2023)
    if weather_data is not None:
        print(f"Weather data shape: {weather_data.shape}")
    
    # Get energy prices
    energy_prices = data_manager.get_energy_prices("New York")
    if energy_prices is not None:
        print(f"Energy prices shape: {energy_prices.shape}")
    
    # Get building codes
    building_codes = data_manager.get_building_codes(climate_zone="4A")
    if building_codes is not None:
        print(f"Building codes shape: {building_codes.shape}")
    
    # Validate data integrity
    print("\nValidating data integrity...")
    validation_results = data_manager.validate_data_integrity()
    print(f"Valid files: {validation_results['valid_files']}")
    print(f"Invalid files: {validation_results['invalid_files']}")
    
    if validation_results['errors']:
        print("Errors found:")
        for error in validation_results['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    # Export unified dataset
    data_manager.export_unified_dataset("unified_contextual_data.db", "sqlite")
    
    # Close connection
    data_manager.close()
    
    print("\nContextual Data Manager demonstration completed!")

if __name__ == "__main__":
    main()