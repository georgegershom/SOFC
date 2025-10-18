"""
TGA/DTA (Thermogravimetric/Differential Thermal Analysis) Data Generator
PhD Research: Thermal Decomposition Kinetics of Rubberized Concrete
Critical for understanding mass loss mechanisms and endothermic/exothermic reactions
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal, integrate
from scipy.interpolate import UnivariateSpline
import warnings
warnings.filterwarnings('ignore')

class TGADTADataGenerator:
    """
    Generate comprehensive TGA/DTA data showing thermal decomposition
    of rubberized concrete components
    """
    
    def __init__(self):
        self.rubber_contents = [0, 5, 10, 15, 20]  # % by volume
        self.heating_rates = [5, 10, 20]  # °C/min
        self.temp_range = np.linspace(20, 1000, 981)  # 20°C to 1000°C
        
        # Key decomposition events
        self.decomposition_events = {
            'Free_Water': {'range': (20, 105), 'peak': 80},
            'C-S-H_Water': {'range': (105, 200), 'peak': 140},
            'Ettringite': {'range': (70, 110), 'peak': 90},
            'Gypsum': {'range': (110, 170), 'peak': 140},
            'Rubber_Volatiles': {'range': (200, 350), 'peak': 280},
            'Rubber_Pyrolysis': {'range': (350, 500), 'peak': 420},
            'Portlandite': {'range': (400, 500), 'peak': 450},
            'C-S-H_Decomposition': {'range': (600, 800), 'peak': 700},
            'Calcite': {'range': (600, 850), 'peak': 750},
            'Rubber_Char_Oxidation': {'range': (500, 700), 'peak': 600}
        }
        
    def generate_tga_curves(self):
        """
        Generate TGA curves showing mass loss vs temperature
        This directly explains strength loss mechanisms
        """
        tga_data = []
        
        for rubber in self.rubber_contents:
            for heating_rate in self.heating_rates:
                
                # Initialize mass (100%)
                mass = np.ones_like(self.temp_range) * 100
                
                # Calculate mass loss for each decomposition event
                for event, params in self.decomposition_events.items():
                    
                    # Determine mass loss magnitude based on component
                    if event == 'Free_Water':
                        max_loss = 2.5 - rubber * 0.05  # Less water with more rubber
                    
                    elif event == 'C-S-H_Water':
                        max_loss = 4.0 - rubber * 0.1
                    
                    elif event == 'Ettringite':
                        max_loss = 0.5 - rubber * 0.01
                    
                    elif event == 'Gypsum':
                        max_loss = 0.3
                    
                    elif event == 'Rubber_Volatiles':
                        max_loss = rubber * 0.15  # Proportional to rubber content
                    
                    elif event == 'Rubber_Pyrolysis':
                        max_loss = rubber * 0.35  # Major rubber decomposition
                    
                    elif event == 'Portlandite':
                        max_loss = 4.5 - rubber * 0.1  # Ca(OH)2 → CaO + H2O
                    
                    elif event == 'C-S-H_Decomposition':
                        max_loss = 3.0 - rubber * 0.08
                    
                    elif event == 'Calcite':
                        max_loss = 2.0 + rubber * 0.02  # Some carbonation with rubber
                    
                    elif event == 'Rubber_Char_Oxidation':
                        max_loss = rubber * 0.1
                    
                    else:
                        max_loss = 0
                    
                    if max_loss > 0:
                        # Create sigmoid-like mass loss curve
                        temp_start, temp_end = params['range']
                        peak_temp = params['peak']
                        
                        # Adjust for heating rate (higher rate shifts to higher temp)
                        peak_shift = heating_rate * 0.5
                        peak_temp += peak_shift
                        
                        # Generate mass loss curve
                        for i, T in enumerate(self.temp_range):
                            if temp_start <= T <= temp_end:
                                # Sigmoid function for smooth mass loss
                                x = (T - peak_temp) / 20
                                loss_fraction = 1 / (1 + np.exp(-x))
                                mass[i] -= max_loss * loss_fraction
                
                # Ensure mass doesn't go below final residue
                residual_mass = 65 - rubber * 2  # More rubber = less residue
                mass = np.maximum(mass, residual_mass)
                
                # Add realistic noise
                mass += np.random.normal(0, 0.05, len(mass))
                
                # Calculate derivative (DTG)
                dtg = -np.gradient(mass, self.temp_range)
                
                # Smooth the curves
                from scipy.ndimage import gaussian_filter1d
                mass = gaussian_filter1d(mass, sigma=2)
                dtg = gaussian_filter1d(dtg, sigma=2)
                
                # Store data
                for i, T in enumerate(self.temp_range[::10]):  # Sample every 10 points
                    tga_data.append({
                        'Rubber_Content_%': rubber,
                        'Heating_Rate_C_per_min': heating_rate,
                        'Temperature_C': T,
                        'Mass_%': mass[i*10],
                        'DTG_%_per_C': dtg[i*10],
                        'Time_min': (T - 20) / heating_rate,
                        'Sample_ID': f'TGA_{rubber}_{heating_rate}',
                        'Timestamp': datetime.now().isoformat()
                    })
        
        return pd.DataFrame(tga_data)
    
    def generate_dta_curves(self):
        """
        Generate DTA curves showing heat flow vs temperature
        Identifies endothermic and exothermic reactions
        """
        dta_data = []
        
        for rubber in self.rubber_contents:
            for heating_rate in self.heating_rates:
                
                # Initialize heat flow (baseline)
                heat_flow = np.zeros_like(self.temp_range)
                
                # Add thermal events
                for event, params in self.decomposition_events.items():
                    
                    temp_start, temp_end = params['range']
                    peak_temp = params['peak'] + heating_rate * 0.5  # Rate adjustment
                    
                    # Determine if endothermic or exothermic
                    if event in ['Free_Water', 'C-S-H_Water', 'Ettringite', 'Gypsum', 
                                'Portlandite', 'C-S-H_Decomposition', 'Calcite']:
                        # Endothermic (negative heat flow)
                        if event == 'Portlandite':
                            peak_height = -2.5 + rubber * 0.05
                        elif event == 'C-S-H_Decomposition':
                            peak_height = -1.8 + rubber * 0.03
                        elif event == 'Calcite':
                            peak_height = -3.0 + rubber * 0.02
                        else:
                            peak_height = -0.8 - rubber * 0.01
                    
                    elif event in ['Rubber_Pyrolysis', 'Rubber_Char_Oxidation']:
                        # Exothermic for rubber (positive heat flow)
                        if event == 'Rubber_Pyrolysis':
                            peak_height = rubber * 0.15
                        else:
                            peak_height = rubber * 0.08
                    
                    else:
                        peak_height = -0.5  # Default endothermic
                    
                    # Generate Gaussian peak
                    for i, T in enumerate(self.temp_range):
                        if temp_start <= T <= temp_end:
                            sigma = 15  # Peak width
                            heat_flow[i] += peak_height * np.exp(-((T - peak_temp) / sigma)**2)
                
                # Add baseline drift
                baseline_drift = 0.0001 * (self.temp_range - 20)
                heat_flow += baseline_drift
                
                # Add noise
                heat_flow += np.random.normal(0, 0.02, len(heat_flow))
                
                # Smooth
                from scipy.ndimage import gaussian_filter1d
                heat_flow = gaussian_filter1d(heat_flow, sigma=3)
                
                # Calculate enthalpy change (integrate heat flow)
                from scipy.integrate import cumulative_trapezoid
                enthalpy = cumulative_trapezoid(heat_flow, self.temp_range, initial=0)
                
                # Store data
                for i, T in enumerate(self.temp_range[::10]):  # Sample every 10 points
                    dta_data.append({
                        'Rubber_Content_%': rubber,
                        'Heating_Rate_C_per_min': heating_rate,
                        'Temperature_C': T,
                        'Heat_Flow_mW_per_mg': heat_flow[i*10],
                        'Enthalpy_Change_J_per_g': enthalpy[i*10],
                        'Time_min': (T - 20) / heating_rate,
                        'Sample_ID': f'DTA_{rubber}_{heating_rate}',
                        'Timestamp': datetime.now().isoformat()
                    })
        
        return pd.DataFrame(dta_data)
    
    def generate_kinetic_analysis(self):
        """
        Advanced kinetic analysis using multiple heating rates
        Kissinger and Ozawa methods for activation energy
        """
        kinetic_data = []
        
        for rubber in self.rubber_contents:
            for event, params in self.decomposition_events.items():
                
                # Skip events not relevant for rubber content
                if rubber == 0 and 'Rubber' in event:
                    continue
                
                peak_temps = []
                for heating_rate in self.heating_rates:
                    # Peak temperature shifts with heating rate
                    Tp = params['peak'] + heating_rate * 0.8 + np.random.normal(0, 1)
                    peak_temps.append(Tp + 273.15)  # Convert to Kelvin
                
                # Kissinger method: ln(β/Tp²) vs 1/Tp
                # Slope = -Ea/R
                beta_values = np.array(self.heating_rates)
                Tp_values = np.array(peak_temps)
                
                x_kissinger = 1000 / Tp_values  # 1000/T for better scaling
                y_kissinger = np.log(beta_values / Tp_values**2)
                
                # Linear fit
                slope_k, intercept_k = np.polyfit(x_kissinger, y_kissinger, 1)
                Ea_kissinger = -slope_k * 8.314  # kJ/mol (R = 8.314 J/mol·K)
                
                # Ozawa method: ln(β) vs 1/Tp
                y_ozawa = np.log(beta_values)
                slope_o, intercept_o = np.polyfit(x_kissinger, y_ozawa, 1)
                Ea_ozawa = -slope_o * 8.314 * 1.052  # Ozawa correction factor
                
                # Pre-exponential factor (Arrhenius)
                A_factor = np.exp(intercept_k + Ea_kissinger * 1000 / (8.314 * np.mean(Tp_values)))
                
                # Reaction order (n)
                if 'Water' in event:
                    n = 1.0  # First order
                elif 'Rubber' in event:
                    n = 1.5 + np.random.normal(0, 0.1)  # Complex kinetics
                else:
                    n = 1.2 + np.random.normal(0, 0.1)
                
                kinetic_data.append({
                    'Rubber_Content_%': rubber,
                    'Decomposition_Event': event,
                    'Peak_Temp_5C_per_min': peak_temps[0] - 273.15,
                    'Peak_Temp_10C_per_min': peak_temps[1] - 273.15,
                    'Peak_Temp_20C_per_min': peak_temps[2] - 273.15,
                    'Activation_Energy_Kissinger_kJ_mol': Ea_kissinger,
                    'Activation_Energy_Ozawa_kJ_mol': Ea_ozawa,
                    'Avg_Activation_Energy_kJ_mol': (Ea_kissinger + Ea_ozawa) / 2,
                    'Pre_exponential_Factor_1_per_s': A_factor,
                    'Reaction_Order': n,
                    'R_squared_Kissinger': 0.95 + np.random.normal(0, 0.02),
                    'R_squared_Ozawa': 0.94 + np.random.normal(0, 0.02),
                    'Timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(kinetic_data)
    
    def generate_mass_loss_summary(self):
        """
        Summary of mass losses at key temperatures
        Directly correlates with strength loss
        """
        summary_data = []
        
        key_temps = [105, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        
        for rubber in self.rubber_contents:
            for temp in key_temps:
                
                # Calculate cumulative mass loss
                mass_loss = 0
                
                # Free water (up to 105°C)
                if temp >= 105:
                    mass_loss += 2.5 - rubber * 0.05
                
                # CSH bound water (105-200°C)
                if temp >= 200:
                    mass_loss += 4.0 - rubber * 0.1
                
                # Rubber volatiles (200-350°C)
                if temp >= 350:
                    mass_loss += rubber * 0.15
                
                # Rubber pyrolysis (350-500°C)
                if temp >= 500:
                    mass_loss += rubber * 0.35
                
                # Portlandite (400-500°C)
                if temp >= 500:
                    mass_loss += 4.5 - rubber * 0.1
                
                # CSH decomposition (600-800°C)
                if temp >= 800:
                    mass_loss += 3.0 - rubber * 0.08
                
                # Calcite (600-850°C)
                if temp >= 850:
                    mass_loss += 2.0 + rubber * 0.02
                
                # Rubber char oxidation (500-700°C)
                if temp >= 700:
                    mass_loss += rubber * 0.1
                
                # Add variation
                mass_loss += np.random.normal(0, 0.3)
                
                # Residual mass
                residual_mass = 100 - mass_loss
                
                # Determine dominant decomposition at this temperature
                if temp <= 105:
                    dominant_process = 'Free Water Evaporation'
                elif temp <= 200:
                    dominant_process = 'CSH Dehydration'
                elif temp <= 350 and rubber > 0:
                    dominant_process = 'Rubber Volatile Release'
                elif temp <= 500:
                    if rubber > 5:
                        dominant_process = 'Rubber Pyrolysis'
                    else:
                        dominant_process = 'Portlandite Decomposition'
                elif temp <= 700 and rubber > 0:
                    dominant_process = 'Rubber Char Oxidation'
                elif temp <= 850:
                    dominant_process = 'CSH & Calcite Decomposition'
                else:
                    dominant_process = 'Residual Decomposition'
                
                summary_data.append({
                    'Rubber_Content_%': rubber,
                    'Temperature_C': temp,
                    'Total_Mass_Loss_%': mass_loss,
                    'Residual_Mass_%': residual_mass,
                    'Dominant_Process': dominant_process,
                    'Free_Water_Loss_%': min(2.5 - rubber * 0.05, mass_loss) if temp >= 105 else 0,
                    'Bound_Water_Loss_%': min(4.0 - rubber * 0.1, mass_loss - (2.5 - rubber * 0.05)) if temp >= 200 else 0,
                    'Rubber_Decomposition_%': rubber * 0.6 if temp >= 700 else (rubber * 0.5 if temp >= 500 else (rubber * 0.15 if temp >= 350 else 0)),
                    'Portlandite_Loss_%': 4.5 - rubber * 0.1 if temp >= 500 else 0,
                    'Timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(summary_data)
    
    def generate_evolved_gas_analysis(self):
        """
        Simulated evolved gas analysis (EGA) coupled with TGA
        Identifies gaseous products of decomposition
        """
        ega_data = []
        
        gases = ['H2O', 'CO2', 'CO', 'CH4', 'SO2', 'HCl']
        
        for rubber in self.rubber_contents:
            for temp in self.temp_range[::50]:  # Every 50°C
                
                gas_concentrations = {}
                
                # H2O evolution
                if temp <= 105:
                    gas_concentrations['H2O_ppm'] = 5000 + np.random.normal(0, 200)
                elif temp <= 200:
                    gas_concentrations['H2O_ppm'] = 8000 + np.random.normal(0, 300)
                elif 400 <= temp <= 500:
                    gas_concentrations['H2O_ppm'] = 10000 + np.random.normal(0, 400)  # Portlandite
                else:
                    gas_concentrations['H2O_ppm'] = 2000 + np.random.normal(0, 100)
                
                # CO2 evolution
                if 600 <= temp <= 850:
                    gas_concentrations['CO2_ppm'] = 15000 + np.random.normal(0, 500)  # Calcite
                elif rubber > 0 and 350 <= temp <= 500:
                    gas_concentrations['CO2_ppm'] = rubber * 500 + np.random.normal(0, 50)  # Rubber
                else:
                    gas_concentrations['CO2_ppm'] = 500 + np.random.normal(0, 50)
                
                # CO evolution (from rubber)
                if rubber > 0 and 400 <= temp <= 600:
                    gas_concentrations['CO_ppm'] = rubber * 200 + np.random.normal(0, 20)
                else:
                    gas_concentrations['CO_ppm'] = 50 + np.random.normal(0, 10)
                
                # CH4 evolution (from rubber pyrolysis)
                if rubber > 0 and 350 <= temp <= 500:
                    gas_concentrations['CH4_ppm'] = rubber * 100 + np.random.normal(0, 10)
                else:
                    gas_concentrations['CH4_ppm'] = 10 + np.random.normal(0, 5)
                
                # SO2 evolution (from rubber sulfur)
                if rubber > 0 and 300 <= temp <= 600:
                    gas_concentrations['SO2_ppm'] = rubber * 30 + np.random.normal(0, 5)
                else:
                    gas_concentrations['SO2_ppm'] = 5 + np.random.normal(0, 2)
                
                # HCl evolution (from PVC contamination in rubber)
                if rubber > 0 and 200 <= temp <= 400:
                    gas_concentrations['HCl_ppm'] = rubber * 5 + np.random.normal(0, 1)
                else:
                    gas_concentrations['HCl_ppm'] = 1 + np.random.normal(0, 0.5)
                
                # Ensure non-negative values
                for gas in gas_concentrations:
                    gas_concentrations[gas] = max(0, gas_concentrations[gas])
                
                ega_data.append({
                    'Rubber_Content_%': rubber,
                    'Temperature_C': temp,
                    **gas_concentrations,
                    'Total_Gas_Evolution_ppm': sum(gas_concentrations.values()),
                    'Timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(ega_data)
    
    def save_all_datasets(self):
        """Save all TGA/DTA analysis datasets"""
        print("\nGenerating TGA/DTA Analysis Datasets...")
        
        # Generate datasets
        tga_data = self.generate_tga_curves()
        dta_data = self.generate_dta_curves()
        kinetic_data = self.generate_kinetic_analysis()
        mass_loss_summary = self.generate_mass_loss_summary()
        ega_data = self.generate_evolved_gas_analysis()
        
        # Save to CSV files
        tga_data.to_csv('TGA_curves.csv', index=False)
        dta_data.to_csv('DTA_curves.csv', index=False)
        kinetic_data.to_csv('kinetic_analysis.csv', index=False)
        mass_loss_summary.to_csv('mass_loss_summary.csv', index=False)
        ega_data.to_csv('evolved_gas_analysis.csv', index=False)
        
        # Generate summary
        summary = {
            'Dataset': 'TGA/DTA Analysis for Rubberized Concrete',
            'Generated': datetime.now().isoformat(),
            'Total_TGA_Points': len(tga_data),
            'Total_DTA_Points': len(dta_data),
            'Total_Kinetic_Analyses': len(kinetic_data),
            'Temperature_Range_C': '20-1000',
            'Heating_Rates_C_per_min': self.heating_rates,
            'Rubber_Contents_%': self.rubber_contents,
            'Key_Decomposition_Events': list(self.decomposition_events.keys()),
            'Critical_Temperatures': {
                'Free_Water_Loss': '20-105°C',
                'CSH_Dehydration': '105-200°C',
                'Rubber_Pyrolysis': '350-500°C',
                'Portlandite_Decomposition': '400-500°C',
                'CSH_Decomposition': '600-800°C',
                'Calcite_Decomposition': '600-850°C'
            },
            'Max_Total_Mass_Loss_%': float(mass_loss_summary['Total_Mass_Loss_%'].max()),
            'Activation_Energy_Range_kJ_mol': f"{kinetic_data['Avg_Activation_Energy_kJ_mol'].min():.1f}-{kinetic_data['Avg_Activation_Energy_kJ_mol'].max():.1f}"
        }
        
        with open('TGA_DTA_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Generated {len(tga_data)} TGA data points")
        print(f"✓ Generated {len(dta_data)} DTA data points")
        print(f"✓ Generated {len(kinetic_data)} kinetic analyses")
        print(f"✓ Generated {len(mass_loss_summary)} mass loss summaries")
        print(f"✓ Generated {len(ega_data)} evolved gas analyses")
        
        return tga_data, dta_data, kinetic_data, mass_loss_summary, ega_data

if __name__ == "__main__":
    generator = TGADTADataGenerator()
    datasets = generator.save_all_datasets()
    print("\n✅ TGA/DTA Analysis Dataset Generation Complete!")