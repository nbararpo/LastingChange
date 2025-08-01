import dj_database_url
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, ttest_rel, ttest_ind
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
from datetime import datetime
import time
from flask import Flask

def create_app():
    app = Flask(__name__)
    # setup app, routes, config, etc.
    return app

app = create_app() 
warnings.filterwarnings('ignore')

# Helper function for effect magnitude (standalone)
def get_effect_magnitude(cohen_d):
    """Convert Cohen's d to magnitude category"""
    if cohen_d < 0.2:
        return "Negligible"
    elif cohen_d < 0.5:
        return "Small"
    elif cohen_d < 0.8:
        return "Medium"
    else:
        return "Large"

# Set page configuration
st.set_page_config(
    page_title="Omics Data Analysis Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .tab-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    
    .analysis-column {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .significance-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .persistence-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedOmicsAnalyzer:
    """Enhanced Omics Data Analyzer for Streamlit App"""
    
    def __init__(self):
        self.omics_data = None
        self.demographics_data = None
        self.merged_data = None
        self.protein_cols = []
        self.demographic_cols = []
        self.timepoints = []
        
    def load_data(self, omics_file, demographics_file):
        """Load and validate omics and demographics data"""
        try:
            # Load omics data
            if omics_file.name.endswith('.csv'):
                self.omics_data = pd.read_csv(omics_file)
            else:
                self.omics_data = pd.read_excel(omics_file)
            
            # Load demographics data
            if demographics_file.name.endswith('.csv'):
                self.demographics_data = pd.read_csv(demographics_file)
            else:
                self.demographics_data = pd.read_excel(demographics_file)
            
            # Validate data structure
            self._validate_data_structure()
            self._merge_datasets()
            
            return True, "Data loaded successfully!"
            
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def _validate_data_structure(self):
        """Validate the structure of uploaded data"""
        # Check omics data structure
        if self.omics_data.shape[1] < 3:
            raise ValueError("Omics data must have at least 3 columns (ID, Timepoint, and biomarkers)")
        
        # Rename columns for consistency
        omics_cols = list(self.omics_data.columns)
        self.omics_data.columns = ['ID', 'Timepoint'] + omics_cols[2:]
        
        # Ensure ID and Timepoint columns are properly formatted
        self.omics_data['ID'] = self.omics_data['ID'].astype(str)
        
        # Extract protein columns (from 3rd column onwards)
        self.protein_cols = list(self.omics_data.columns[2:])
        
        # Get unique timepoints and handle potential errors
        try:
            self.timepoints = sorted([tp for tp in self.omics_data['Timepoint'].unique() if pd.notna(tp)])
        except:
            # If sorting fails, just get unique values
            self.timepoints = list(self.omics_data['Timepoint'].unique())
            self.timepoints = [tp for tp in self.timepoints if pd.notna(tp)]
        
        # Check demographics data structure
        demo_cols = list(self.demographics_data.columns)
        
        # Ensure we have at least 11 columns for the expected structure
        if len(demo_cols) < 11:
            st.warning(f"⚠️ Demographics data has only {len(demo_cols)} columns. Expected structure:")
            st.write("Column 1: ID, Column 4: sex, Column 5: age, Columns 6-11: timepoints 1-6")
        
        # Ensure ID column is properly formatted (first column)
        self.demographics_data.iloc[:, 0] = self.demographics_data.iloc[:, 0].astype(str)
        
        # Set standardized column names for easier access
        if len(demo_cols) >= 11:
            # Create new column names
            new_demo_cols = demo_cols.copy()
            new_demo_cols[0] = 'ID'  # First column is always ID
            
            # Check if column 4 looks like sex data (common sex values)
            if len(demo_cols) >= 4:
                col4_values = self.demographics_data.iloc[:, 3].dropna().astype(str).str.upper().unique()
                sex_indicators = any(val in ['M', 'F', 'MALE', 'FEMALE', '1', '2', 'MAN', 'WOMAN'] for val in col4_values)
                if sex_indicators:
                    new_demo_cols[3] = 'sex'
                    self.sex_col = 'sex'
                else:
                    self.sex_col = None
                    st.warning(f"⚠️ Column 4 doesn't appear to contain sex/gender data. Found values: {', '.join(col4_values[:5])}")
            else:
                self.sex_col = None
            
            # Column 5 should be age - validate it's numeric
            if len(demo_cols) >= 5:
                try:
                    age_values = pd.to_numeric(self.demographics_data.iloc[:, 4], errors='coerce')
                    if age_values.notna().sum() > 0:  # At least some valid numeric values
                        new_demo_cols[4] = 'age'
                        self.age_col = 'age'
                    else:
                        self.age_col = None
                        st.warning("⚠️ Column 5 doesn't appear to contain numeric age data")
                except:
                    self.age_col = None
                    st.warning("⚠️ Column 5 cannot be processed as age data")
            else:
                self.age_col = None
            
            # Columns 6-11 are timepoints 1-6
            self.timepoint_demo_cols = []
            for i, tp in enumerate(range(1, 7)):  # timepoints 1-6
                col_idx = 5 + i  # columns 6-11 (0-indexed: 5-10)
                if col_idx < len(demo_cols):
                    new_demo_cols[col_idx] = f'timepoint_{tp}'
                    self.timepoint_demo_cols.append(f'timepoint_{tp}')
            
            # Apply new column names
            self.demographics_data.columns = new_demo_cols
            
        else:
            # Fallback: try to detect sex/age by column names for shorter data
            self.sex_col = None
            self.age_col = None
            self.timepoint_demo_cols = []
            
            # Standard column naming
            new_demo_cols = ['ID'] + [f'col_{i+2}' for i in range(len(demo_cols)-1)]
            
            # Look for sex and age columns by name (case insensitive)
            for i, col in enumerate(demo_cols):
                col_lower = col.lower()
                if col_lower in ['sex', 'gender']:
                    self.sex_col = col
                    new_demo_cols[i] = col
                elif col_lower in ['age']:
                    self.age_col = col
                    new_demo_cols[i] = col
                elif str(col).isdigit() and int(col) in [1, 2, 3, 4, 5, 6]:
                    self.timepoint_demo_cols.append(col)
                    new_demo_cols[i] = col
            
            self.demographics_data.columns = new_demo_cols
        
        # Extract other demographic columns (columns 2-3 and any beyond 11)
        self.demographic_cols = []
        for i, col in enumerate(self.demographics_data.columns):
            if (i == 1 or i == 2 or  # columns 2-3
                (i > 10 and col not in ['sex', 'age'] + self.timepoint_demo_cols)):  # beyond column 11
                self.demographic_cols.append(col)
        
        # Log what was found
        st.info(f"📊 Demographics structure detected:")
        st.info(f"  • Total columns: {len(self.demographics_data.columns)}")
        if self.sex_col and self.sex_col in self.demographics_data.columns:
            st.info(f"  • Sex column (col 4): {self.sex_col}")
        if self.age_col and self.age_col in self.demographics_data.columns:
            st.info(f"  • Age column (col 5): {self.age_col}")
        if self.timepoint_demo_cols:
            st.info(f"  • Timepoint columns (col 6-11): {', '.join(self.timepoint_demo_cols)}")
        if self.demographic_cols:
            st.info(f"  • Other demographic columns: {', '.join(self.demographic_cols)}")
    
    def _merge_datasets(self):
        """Merge omics and demographics data"""
        # If demographics has timepoint columns, we need to reshape it first
        if self.timepoint_demo_cols:
            st.info(f"🔄 Reshaping demographics data from wide to long format...")
            # Reshape demographics data from wide to long format
            demo_long_data = []
            
            for _, row in self.demographics_data.iterrows():
                participant_id = row['ID']
                base_demo = {}
                
                # Add sex and age if available
                if self.sex_col and self.sex_col in row.index and pd.notna(row[self.sex_col]):
                    base_demo[self.sex_col] = row[self.sex_col]
                if self.age_col and self.age_col in row.index and pd.notna(row[self.age_col]):
                    base_demo[self.age_col] = row[self.age_col]
                
                # Add other demographic columns
                for demo_col in self.demographic_cols:
                    if demo_col in row.index and pd.notna(row[demo_col]):
                        base_demo[demo_col] = row[demo_col]
                
                # Create rows for each timepoint (extract timepoint number from column name)
                for tp_col in self.timepoint_demo_cols:
                    if tp_col in row.index and pd.notna(row[tp_col]):
                        # Extract timepoint number from column name (e.g., 'timepoint_1' -> 1)
                        if '_' in tp_col:
                            timepoint_num = int(tp_col.split('_')[1])
                        else:
                            # Fallback for direct numeric column names
                            timepoint_num = int(tp_col)
                        
                        demo_row = {
                            'ID': participant_id,
                            'Timepoint': timepoint_num,
                            'demographic_value': row[tp_col]
                        }
                        demo_row.update(base_demo)
                        demo_long_data.append(demo_row)
            
            if demo_long_data:
                demographics_long = pd.DataFrame(demo_long_data)
                # Merge with omics data
                self.merged_data = pd.merge(self.omics_data, demographics_long, 
                                          on=['ID', 'Timepoint'], how='left')
                st.success(f"✅ Merged data: {len(self.merged_data)} samples with timepoint demographics")
            else:
                # No timepoint data, merge on ID only
                self.merged_data = pd.merge(self.omics_data, self.demographics_data, 
                                          on='ID', how='left')
                st.warning("⚠️ No valid timepoint demographic data found - merged on ID only")
        else:
            # Standard merge on ID only
            self.merged_data = pd.merge(self.omics_data, self.demographics_data, 
                                      on='ID', how='left')
            st.info("ℹ️ No timepoint columns found - standard merge on ID only")
    
    def get_data_summary(self):
        """Generate comprehensive data summary"""
        if self.merged_data is None:
            return None
        
        # Calculate missing data percentages safely
        omics_missing_pct = 0
        if self.protein_cols:
            omics_missing_pct = round((self.omics_data[self.protein_cols].isna().sum().sum() / 
                                     (len(self.omics_data) * len(self.protein_cols)) * 100), 2)
        
        # Calculate demographics missing data
        demo_cols_for_missing = []
        if self.sex_col and self.sex_col in self.demographics_data.columns:
            demo_cols_for_missing.append(self.sex_col)
        if self.age_col and self.age_col in self.demographics_data.columns:
            demo_cols_for_missing.append(self.age_col)
        demo_cols_for_missing.extend([col for col in self.timepoint_demo_cols if col in self.demographics_data.columns])
        demo_cols_for_missing.extend([col for col in self.demographic_cols if col in self.demographics_data.columns])
        
        demographics_missing_pct = 0
        if demo_cols_for_missing:
            demographics_missing_pct = round((self.demographics_data[demo_cols_for_missing].isna().sum().sum() / 
                                            (len(self.demographics_data) * len(demo_cols_for_missing)) * 100), 2)
        
        # Calculate completeness rate safely
        max_participants = max(self.omics_data['ID'].nunique(), 
                              self.demographics_data['ID'].nunique())
        completeness_rate = 0
        if max_participants > 0:
            completeness_rate = round((self.merged_data['ID'].nunique() / max_participants * 100), 2)
        
        # Demographics variables summary
        demo_vars = []
        if self.sex_col and self.sex_col in self.demographics_data.columns:
            demo_vars.append(f"{self.sex_col} (column 4)")
        if self.age_col and self.age_col in self.demographics_data.columns:
            demo_vars.append(f"{self.age_col} (column 5)")
        if self.timepoint_demo_cols:
            # Fix the timepoint display - handle columns that may not have underscores
            timepoint_display = []
            for col in self.timepoint_demo_cols:
                if col in self.demographics_data.columns:
                    if '_' in col:
                        timepoint_display.append(f"T{col.split('_')[1]}")
                    else:
                        timepoint_display.append(f"T{col}")
            demo_vars.append(f"Timepoint measures (columns 6-11): {', '.join(timepoint_display)}")
        demo_vars.extend([f"{col} (other)" for col in self.demographic_cols if col in self.demographics_data.columns])
        
        summary = {
            'omics': {
                'n_samples': len(self.omics_data),
                'n_participants': self.omics_data['ID'].nunique(),
                'n_biomarkers': len(self.protein_cols),
                'timepoints': self.timepoints,
                'n_timepoints': len(self.timepoints),
                'missing_data_pct': omics_missing_pct
            },
            'demographics': {
                'n_participants': len(self.demographics_data),
                'n_variables': len(demo_cols_for_missing),
                'variables': demo_vars,
                'sex_column': self.sex_col if self.sex_col in self.demographics_data.columns else None,
                'age_column': self.age_col if self.age_col in self.demographics_data.columns else None,
                'timepoint_columns': [col for col in self.timepoint_demo_cols if col in self.demographics_data.columns],
                'other_columns': [col for col in self.demographic_cols if col in self.demographics_data.columns],
                'missing_data_pct': demographics_missing_pct
            },
            'merged': {
                'n_samples': len(self.merged_data),
                'n_participants_with_both': self.merged_data['ID'].nunique(),
                'completeness_rate': completeness_rate
            }
        }
        
        return summary
    
    def analyze_demographics_by_timepoint(self, selected_variables=None):
        """Analyze demographics across timepoints"""
        if selected_variables is None:
            # Build available variables list
            available_vars = []
            if self.sex_col and self.sex_col in self.merged_data.columns:
                available_vars.append(self.sex_col)
            if self.age_col and self.age_col in self.merged_data.columns:
                available_vars.append(self.age_col)
            if 'demographic_value' in self.merged_data.columns:
                available_vars.append('demographic_value')
            available_vars.extend([col for col in self.demographic_cols if col in self.merged_data.columns])
            selected_variables = available_vars
        
        results = {}
        
        for var in selected_variables:
            # Handle display names vs actual column names
            actual_var = var.replace(' (timepoint measurements)', '') if 'timepoint measurements' in var else var
            
            if actual_var in self.merged_data.columns:
                # Handle categorical variables (like sex)
                if actual_var == self.sex_col:
                    # For categorical data, show counts and proportions by timepoint
                    crosstab = pd.crosstab(self.merged_data['Timepoint'], 
                                         self.merged_data[actual_var], 
                                         margins=True, normalize='index')
                    
                    # Chi-square test for independence
                    try:
                        chi2, p_value, dof, expected = stats.chi2_contingency(
                            pd.crosstab(self.merged_data['Timepoint'], self.merged_data[actual_var])
                        )
                        results[actual_var] = {
                            'type': 'categorical',
                            'crosstab': crosstab,
                            'chi2_statistic': chi2,
                            'chi2_p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        results[actual_var] = {
                            'type': 'categorical',
                            'crosstab': crosstab,
                            'chi2_statistic': None,
                            'chi2_p_value': None,
                            'significant': False
                        }
                
                else:
                    # For numeric variables (age, demographic_value, etc.)
                    var_data = self.merged_data.groupby('Timepoint')[actual_var].agg([
                        'count', 'mean', 'std', 'median', 'min', 'max'
                    ]).round(3)
                    
                    # Statistical test across timepoints
                    timepoint_groups = []
                    for tp in self.timepoints:
                        tp_data = self.merged_data[self.merged_data['Timepoint'] == tp][actual_var].dropna()
                        if len(tp_data) > 0:
                            timepoint_groups.append(tp_data)
                    
                    if len(timepoint_groups) > 1:
                        try:
                            f_stat, p_value = stats.f_oneway(*timepoint_groups)
                            results[actual_var] = {
                                'type': 'numeric',
                                'summary_stats': var_data,
                                'anova_f': f_stat,
                                'anova_p': p_value,
                                'significant': p_value < 0.05
                            }
                        except:
                            results[actual_var] = {
                                'type': 'numeric',
                                'summary_stats': var_data,
                                'anova_f': None,
                                'anova_p': None,
                                'significant': False
                            }
        
        return results
    
    def analyze_within_individual_differences_only(self, baseline_timepoint, selected_timepoint):
        """
        Simplified within-individual analysis:
        - Only calculate difference values between baseline and selected timepoint
        - Focus on individual change patterns (selected_time - baseline)
        """
        
        # Get data for the two timepoints
        baseline_data = self.merged_data[self.merged_data['Timepoint'] == baseline_timepoint]
        selected_data = self.merged_data[self.merged_data['Timepoint'] == selected_timepoint]
        
        # Find participants who have data at both timepoints
        common_ids = set(baseline_data['ID']).intersection(set(selected_data['ID']))
        
        if len(common_ids) == 0:
            return None, "No participants found with data at both timepoints"
        
        within_individual_results = {}
        
        for protein in self.protein_cols:
            # Get protein values for same individuals at both timepoints
            baseline_protein = baseline_data[baseline_data['ID'].isin(common_ids)].set_index('ID')[protein]
            selected_protein = selected_data[selected_data['ID'].isin(common_ids)].set_index('ID')[protein]
            
            # Align data by participant ID and remove missing values
            aligned_baseline = baseline_protein.reindex(common_ids).dropna()
            aligned_selected = selected_protein.reindex(aligned_baseline.index).dropna()
            
            if len(aligned_baseline) >= 3:  # Need at least 3 participants
                
                # Calculate individual differences: selected_time - baseline
                individual_differences = aligned_selected - aligned_baseline
                
                # Only proceed if differences are meaningful (not all zero)
                if individual_differences.std() > 1e-10:  # Avoid all-zero differences
                    
                    # Paired t-test on the individual differences
                    t_stat, p_value = stats.ttest_rel(aligned_selected, aligned_baseline)
                    
                    # Effect size (Cohen's d for paired samples)
                    cohen_d = (individual_differences.mean() / 
                              individual_differences.std()) if individual_differences.std() > 0 else 0
                    
                    # Count response patterns
                    n_increased = (individual_differences > 0).sum()
                    n_decreased = (individual_differences < 0).sum() 
                    n_unchanged = (individual_differences == 0).sum()
                    total_participants = len(individual_differences)
                    
                    # Calculate original means for reference
                    baseline_abs_mean = abs(aligned_baseline.mean())
                    selected_abs_mean = abs(aligned_selected.mean())
                    
                    # Store results
                    within_individual_results[protein] = {
                        'comparison': f"{baseline_timepoint} → {selected_timepoint}",
                        'n_participants': total_participants,
                        'individual_differences': individual_differences.values.tolist(),
                        
                        # Summary statistics of differences
                        'mean_difference': individual_differences.mean(),
                        'median_difference': individual_differences.median(),
                        'std_difference': individual_differences.std(),
                        
                        # Statistical test results
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohen_d': cohen_d,
                        'significant': p_value < 0.05,
                        'effect_size': self._get_effect_magnitude(abs(cohen_d)),
                        
                        # Direction and response patterns
                        'overall_direction': 'increase' if individual_differences.mean() > 0 else 'decrease',
                        'n_participants_increased': n_increased,
                        'n_participants_decreased': n_decreased,
                        'n_participants_unchanged': n_unchanged,
                        'percent_increased': (n_increased / total_participants * 100),
                        'percent_decreased': (n_decreased / total_participants * 100),
                        
                        # Consistency measure
                        'response_consistency': max(n_increased, n_decreased) / total_participants,
                        
                        # Original values for reference and display
                        'baseline_values': aligned_baseline.values.tolist(),
                        'selected_values': aligned_selected.values.tolist(),
                        'baseline_abs_mean': baseline_abs_mean,
                        'selected_abs_mean': selected_abs_mean,
                        'percent_change': (individual_differences.mean() / baseline_abs_mean * 100) if baseline_abs_mean != 0 else 0
                    }
        
        return within_individual_results, f"Analysis completed for {len(within_individual_results)} proteins"

    def analyze_significant_changes_enhanced(self, baseline_timepoint, target_timepoints, meaningful_threshold=2.0):
        """
        OPTIMIZED Enhanced significance analysis with both group-level and individual-level testing
        
        Group-level: Paired t-test testing if average change across all participants ≠ 0
        Individual-level: Binomial test for proportion of participants with meaningful changes
        
        Performance optimizations:
        - Pre-filter participants with complete data
        - Vectorized calculations where possible
        - Reduced redundant computations
        """
        all_results = {}
        
        for target_tp in target_timepoints:
            # OPTIMIZATION: Pre-filter and align data once
            baseline_data = self.merged_data[self.merged_data['Timepoint'] == baseline_timepoint]
            target_data = self.merged_data[self.merged_data['Timepoint'] == target_tp]
            
            # Find participants with data at both timepoints (once)
            common_ids = set(baseline_data['ID']).intersection(set(target_data['ID']))
            
            if len(common_ids) == 0:
                all_results[target_tp] = {
                    'within': {},
                    'across': {},
                    'message': "No participants found with data at both timepoints"
                }
                continue
            
            # Pre-filter data to common participants
            baseline_filtered = baseline_data[baseline_data['ID'].isin(common_ids)].set_index('ID')
            target_filtered = target_data[target_data['ID'].isin(common_ids)].set_index('ID')
            
            within_results = {}
            across_results = {}
            
            # OPTIMIZATION: Process only proteins with sufficient data
            valid_proteins = []
            for protein in self.protein_cols:
                if (protein in baseline_filtered.columns and protein in target_filtered.columns):
                    baseline_vals = baseline_filtered[protein].dropna()
                    target_vals = target_filtered[protein].reindex(baseline_vals.index).dropna()
                    if len(baseline_vals) >= 2:  # Minimum sample size
                        valid_proteins.append(protein)
            
            for protein in valid_proteins:
                # WITHIN-INDIVIDUAL ANALYSIS (Enhanced with both tests) - OPTIMIZED
                baseline_protein = baseline_filtered[protein].dropna()
                target_protein = target_filtered[protein].reindex(baseline_protein.index).dropna()
                
                if len(baseline_protein) >= 2:
                    # Calculate individual differences (vectorized)
                    individual_differences = target_protein - baseline_protein
                    
                    if individual_differences.std() > 1e-10:  # Avoid all-zero differences
                        # GROUP-LEVEL TEST: Paired t-test (optimized)
                        group_t_stat, group_p_value = stats.ttest_rel(target_protein, baseline_protein)
                        
                        # Effect size (Cohen's d for paired samples)
                        cohen_d = (individual_differences.mean() / 
                                  individual_differences.std()) if individual_differences.std() > 0 else 0
                        
                        # INDIVIDUAL-LEVEL TEST: Meaningful changes analysis (RELATIVE CHANGES)
                        # Calculate fold changes: target/baseline
                        baseline_positive = baseline_protein + 1e-6  # Add small value to avoid division by zero
                        fold_changes = target_protein / baseline_positive
                        
                        # Meaningful changes: fold change >= threshold OR fold change <= 1/threshold
                        meaningful_increases = fold_changes >= meaningful_threshold  # e.g., >= 2.0 (100% increase)
                        meaningful_decreases = fold_changes <= (1.0 / meaningful_threshold)  # e.g., <= 0.5 (50% decrease)
                        meaningful_changes = meaningful_increases | meaningful_decreases
                        
                        n_meaningful = meaningful_changes.sum()
                        n_total = len(fold_changes)
                        proportion_meaningful = n_meaningful / n_total
                        
                        # Binomial test (optimized)
                        try:
                            from scipy.stats import binomtest
                            binomial_result = binomtest(n_meaningful, n_total, 0.5)
                            individual_p_value = binomial_result.pvalue
                        except:
                            # Fallback calculation
                            from scipy.stats import binom
                            individual_p_value = 2 * (1 - binom.cdf(max(n_meaningful, n_total - n_meaningful) - 1, n_total, 0.5))
                        
                        # Count meaningful changes by direction (vectorized)
                        meaningful_increased = meaningful_increases.sum()
                        meaningful_decreased = meaningful_decreases.sum()
                        
                        # Determine dominant direction
                        meaningful_consistency = (meaningful_increased / n_meaningful if meaningful_increased > meaningful_decreased 
                                                else meaningful_decreased / n_meaningful) if n_meaningful > 0 else 0
                        
                        # Response patterns (vectorized) - keep absolute differences for overall patterns
                        n_increased = (individual_differences > 0).sum()
                        n_decreased = (individual_differences < 0).sum()
                        n_unchanged = (individual_differences == 0).sum()
                        
                        # Calculate reference values
                        baseline_abs_mean = abs(baseline_protein.mean())
                        target_abs_mean = abs(target_protein.mean())
                        
                        within_results[protein] = {
                            'comparison': f"{baseline_timepoint} → {target_tp}",
                            'n_participants': n_total,
                            'individual_differences': individual_differences.values.tolist(),
                            'fold_changes': fold_changes.values.tolist(),  # Add fold changes
                            
                            # GROUP-LEVEL RESULTS
                            'mean_difference': individual_differences.mean(),
                            'std_difference': individual_differences.std(),
                            'cohen_d': cohen_d,
                            'group_t_statistic': group_t_stat,
                            'group_p_value': group_p_value,
                            'group_significant': group_p_value < 0.05,
                            'effect_size': self._get_effect_magnitude(abs(cohen_d)),
                            
                            # INDIVIDUAL-LEVEL RESULTS (RELATIVE CHANGES)
                            'meaningful_threshold': meaningful_threshold,
                            'meaningful_threshold_description': f"≥{meaningful_threshold:.1f}x increase OR ≤{(1.0/meaningful_threshold):.1f}x decrease",
                            'n_meaningful_changes': n_meaningful,
                            'proportion_meaningful': proportion_meaningful,
                            'individual_p_value': individual_p_value,
                            'individual_significant': individual_p_value < 0.05,
                            'n_meaningful_increased': meaningful_increased,
                            'n_meaningful_decreased': meaningful_decreased,
                            'meaningful_consistency': meaningful_consistency,
                            'mean_fold_change': fold_changes.mean(),
                            'median_fold_change': fold_changes.median(),
                            
                            # OVERALL PATTERNS
                            'overall_direction': 'increase' if individual_differences.mean() > 0 else 'decrease',
                            'n_participants_increased': n_increased,
                            'n_participants_decreased': n_decreased,
                            'n_participants_unchanged': n_unchanged,
                            'percent_increased': (n_increased / n_total * 100),
                            'percent_decreased': (n_decreased / n_total * 100),
                            'response_consistency': max(n_increased, n_decreased) / n_total,
                            
                            # REFERENCE VALUES
                            'baseline_values': baseline_protein.values.tolist(),
                            'target_values': target_protein.values.tolist(),
                            'baseline_abs_mean': baseline_abs_mean,
                            'target_abs_mean': target_abs_mean,
                            'percent_change': (individual_differences.mean() / baseline_abs_mean * 100) if baseline_abs_mean != 0 else 0,
                            
                            # COMBINED SIGNIFICANCE
                            'significant': group_p_value < 0.05  # For compatibility
                        }
                
                # ACROSS-INDIVIDUAL ANALYSIS (Independent samples) - OPTIMIZED
                baseline_all = baseline_data[protein].dropna()
                target_all = target_data[protein].dropna()
                
                if len(baseline_all) > 5 and len(target_all) > 5:
                    # Independent t-test
                    across_t_stat, across_p_value = stats.ttest_ind(target_all, baseline_all)
                    
                    # Effect size (Cohen's d for independent samples) - optimized
                    pooled_std = np.sqrt(((len(baseline_all) - 1) * np.var(baseline_all, ddof=1) + 
                                         (len(target_all) - 1) * np.var(target_all, ddof=1)) / 
                                        (len(baseline_all) + len(target_all) - 2))
                    across_cohen_d = (np.mean(target_all) - np.mean(baseline_all)) / pooled_std if pooled_std > 0 else 0
                    
                    # Calculate means and CV (optimized)
                    baseline_abs_mean = abs(np.mean(baseline_all))
                    target_abs_mean = abs(np.mean(target_all))
                    baseline_cv = (np.std(baseline_all) / baseline_abs_mean * 100) if baseline_abs_mean > 0 else 0
                    target_cv = (np.std(target_all) / target_abs_mean * 100) if target_abs_mean > 0 else 0
                    
                    across_results[protein] = {
                        'n_baseline': len(baseline_all),
                        'n_target': len(target_all),
                        'baseline_abs_mean': baseline_abs_mean,
                        'target_abs_mean': target_abs_mean,
                        'baseline_cv': abs(baseline_cv),
                        'target_cv': abs(target_cv),
                        'mean_change': np.mean(target_all) - np.mean(baseline_all),
                        'percent_change': ((np.mean(target_all) - np.mean(baseline_all)) / baseline_abs_mean * 100) if baseline_abs_mean != 0 else 0,
                        't_statistic': across_t_stat,
                        'p_value': across_p_value,
                        'cohen_d': across_cohen_d,
                        'significant': across_p_value < 0.05,
                        'effect_size': self._get_effect_magnitude(abs(across_cohen_d)),
                        'direction': 'increase' if np.mean(target_all) > np.mean(baseline_all) else 'decrease'
                    }
            
            all_results[target_tp] = {
                'within': within_results,
                'across': across_results,
                'message': f"OPTIMIZED analysis completed for {len(within_results)} proteins"
            }
        
        return all_results

    def analyze_significant_changes(self, baseline_timepoint, target_timepoints):
        """
        Fixed version that only does within-individual difference analysis
        Removes variable scope issues
        """
        
        all_results = {}
        
        for target_tp in target_timepoints:
            
            # Within-individual analysis: only difference calculations
            within_results, message = self.analyze_within_individual_differences_only(
                baseline_timepoint, target_tp
            )
            
            # Keep across-individual analysis for comparison
            baseline_data = self.merged_data[self.merged_data['Timepoint'] == baseline_timepoint]
            target_data = self.merged_data[self.merged_data['Timepoint'] == target_tp]
            
            across_results = {}
            
            for protein in self.protein_cols:
                # ACROSS-INDIVIDUAL ANALYSIS (Independent comparison)
                baseline_all = baseline_data[protein].dropna()
                target_all = target_data[protein].dropna()
                
                if len(baseline_all) > 5 and len(target_all) > 5:
                    # Independent t-test
                    t_stat, p_value = stats.ttest_ind(target_all, baseline_all)
                    
                    # Effect size (Cohen's d for independent samples)
                    pooled_std = np.sqrt(((len(baseline_all) - 1) * np.var(baseline_all, ddof=1) + 
                                         (len(target_all) - 1) * np.var(target_all, ddof=1)) / 
                                        (len(baseline_all) + len(target_all) - 2))
                    cohen_d = (np.mean(target_all) - np.mean(baseline_all)) / pooled_std if pooled_std > 0 else 0
                    
                    # Calculate absolute means and CV
                    baseline_abs_mean = abs(np.mean(baseline_all))
                    target_abs_mean = abs(np.mean(target_all))
                    baseline_cv = (np.std(baseline_all) / baseline_abs_mean * 100) if baseline_abs_mean > 0 else 0
                    target_cv = (np.std(target_all) / target_abs_mean * 100) if target_abs_mean > 0 else 0
                    
                    across_results[protein] = {
                        'n_baseline': len(baseline_all),
                        'n_target': len(target_all),
                        'baseline_abs_mean': baseline_abs_mean,
                        'target_abs_mean': target_abs_mean,
                        'baseline_cv': abs(baseline_cv),
                        'target_cv': abs(target_cv),
                        'mean_change': np.mean(target_all) - np.mean(baseline_all),
                        'percent_change': ((np.mean(target_all) - np.mean(baseline_all)) / baseline_abs_mean * 100) if baseline_abs_mean != 0 else 0,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohen_d': cohen_d,
                        'significant': p_value < 0.05,
                        'effect_size': self._get_effect_magnitude(abs(cohen_d)),
                        'direction': 'increase' if np.mean(target_all) > np.mean(baseline_all) else 'decrease'
                    }
            
            if within_results:
                all_results[target_tp] = {
                    'within': within_results,
                    'across': across_results,
                    'message': message
                }
            else:
                all_results[target_tp] = {
                    'within': {},
                    'across': across_results,
                    'message': message
                }
        
        return all_results
    
    def analyze_persistence_enhanced_v2(self, baseline_tp=None, intermediate_tp=None, final_tps=None, significant_proteins=None):
        """
        CORRECTED Enhanced persistence analysis with proper logical flow:
        
        1. MEANINGFUL CHANGES ANALYSIS:
           - Within-individual: Relative changes (intermediate/baseline >= 2.0)
           - Across-individual: Two-sided t-test (baseline vs intermediate)
        
        2. STABILITY ANALYSIS:
           - CV analysis for stability (CV < 20%)
           - T-test for intermediate vs final timepoints
        
        3. PREDICTION ANALYSIS:
           - Use early changes (baseline→intermediate) to predict all final timepoints
           - Understand how long changes are stable and when they stop
        """
        if baseline_tp is None:
            baseline_tp = min(self.timepoints)
        if intermediate_tp is None and len(self.timepoints) >= 3:
            intermediate_tp = sorted(self.timepoints)[2]  # T3
        if final_tps is None and len(self.timepoints) >= 4:
            final_tps = sorted(self.timepoints)[3:]  # T4, T5, T6, etc.
        
        if not final_tps or intermediate_tp is None:
            return None, "Insufficient timepoints for persistence analysis"
        
        # Use only significant proteins if provided
        if significant_proteins is not None:
            proteins_to_analyze = [p for p in significant_proteins if p in self.protein_cols]
            if not proteins_to_analyze:
                return None, "No significant proteins found for persistence analysis"
        else:
            proteins_to_analyze = self.protein_cols
        
        # Get data for required timepoints
        baseline_data = self.merged_data[self.merged_data['Timepoint'] == baseline_tp]
        intermediate_data = self.merged_data[self.merged_data['Timepoint'] == intermediate_tp]
        
        persistence_results = {}
        
        for protein in proteins_to_analyze:
            protein_result = self._analyze_single_protein_persistence_corrected(
                protein, baseline_tp, intermediate_tp, final_tps,
                baseline_data, intermediate_data
            )
            if protein_result:
                persistence_results[protein] = protein_result
        
        analysis_type = "significant proteins" if significant_proteins is not None else "all proteins"
        return persistence_results, f"CORRECTED persistence analysis completed for {len(persistence_results)} {analysis_type}"
    
    def _analyze_single_protein_persistence_corrected(self, protein, baseline_tp, intermediate_tp, final_tps, baseline_data, intermediate_data):
        """
        CORRECTED persistence analysis for single protein following proper logical flow
        """
        
        # STEP 1: MEANINGFUL CHANGES ANALYSIS
        meaningful_changes_result = self._step1_meaningful_changes_analysis(
            protein, baseline_tp, intermediate_tp, baseline_data, intermediate_data
        )
        
        if not meaningful_changes_result:
            return None
        
        # STEP 2: STABILITY ANALYSIS for each final timepoint
        stability_results = {}
        prediction_results = {}
        
        for final_tp in final_tps:
            final_data = self.merged_data[self.merged_data['Timepoint'] == final_tp]
            
            # Step 2: Stability Analysis
            stability_result = self._step2_stability_analysis(
                protein, intermediate_tp, final_tp, intermediate_data, final_data,
                meaningful_changes_result['meaningful_participants']
            )
            
            # Step 3: Prediction Analysis
            prediction_result = self._step3_prediction_analysis(
                protein, baseline_tp, intermediate_tp, final_tp,
                baseline_data, intermediate_data, final_data,
                meaningful_changes_result['meaningful_participants']
            )
            
            if stability_result and prediction_result:
                stability_results[final_tp] = stability_result
                prediction_results[final_tp] = prediction_result
        
        if not stability_results or not prediction_results:
            return None
        
        # STEP 4: OVERALL PERSISTENCE ASSESSMENT
        overall_result = self._step4_overall_persistence_assessment(
            protein, meaningful_changes_result, stability_results, prediction_results
        )
        
        return overall_result
    
    def _step1_meaningful_changes_analysis(self, protein, baseline_tp, intermediate_tp, baseline_data, intermediate_data):
        """
        Step 1: Analyze meaningful changes (baseline → intermediate)
        """
        # Find participants with data at both timepoints
        common_ids = set(baseline_data['ID']).intersection(set(intermediate_data['ID']))
        
        if len(common_ids) < 10:  # Need sufficient sample size
            return None
        
        # WITHIN-INDIVIDUAL MEANINGFUL CHANGES (Relative)
        baseline_values = baseline_data[baseline_data['ID'].isin(common_ids)].set_index('ID')[protein].dropna()
        intermediate_values = intermediate_data[intermediate_data['ID'].isin(common_ids)].set_index('ID')[protein]
        
        # Align data
        aligned_intermediate = intermediate_values.reindex(baseline_values.index).dropna()
        aligned_baseline = baseline_values.reindex(aligned_intermediate.index)
        
        if len(aligned_baseline) < 10:
            return None
        
        # Calculate fold changes (intermediate/baseline)
        baseline_positive = aligned_baseline + 1e-6  # Avoid division by zero
        fold_changes = aligned_intermediate / baseline_positive
        
        # Meaningful changes: >= 2.0x increase OR <= 0.5x decrease
        meaningful_increases = fold_changes >= 2.0
        meaningful_decreases = fold_changes <= 0.5
        meaningful_changes_mask = meaningful_increases | meaningful_decreases
        
        meaningful_participants = aligned_baseline.index[meaningful_changes_mask].tolist()
        n_meaningful = len(meaningful_participants)
        
        if n_meaningful < 5:  # Need some meaningful changes
            return None
        
        # ACROSS-INDIVIDUAL MEANINGFUL CHANGES (Two-sided t-test)
        baseline_all = baseline_data[protein].dropna()
        intermediate_all = intermediate_data[protein].dropna()
        
        if len(baseline_all) < 10 or len(intermediate_all) < 10:
            return None
        
        # Two-sided t-test
        t_stat, p_value = stats.ttest_ind(intermediate_all, baseline_all)
        across_significant = p_value < 0.05
        
        return {
            'n_total_participants': len(aligned_baseline),
            'n_meaningful_participants': n_meaningful,
            'proportion_meaningful': n_meaningful / len(aligned_baseline),
            'meaningful_participants': meaningful_participants,
            'fold_changes': fold_changes.loc[meaningful_participants].values.tolist(),
            'baseline_values': aligned_baseline.loc[meaningful_participants].values.tolist(),
            'intermediate_values': aligned_intermediate.loc[meaningful_participants].values.tolist(),
            'across_t_stat': t_stat,
            'across_p_value': p_value,
            'across_significant': across_significant,
            'baseline_mean': baseline_all.mean(),
            'intermediate_mean': intermediate_all.mean(),
            'mean_change': intermediate_all.mean() - baseline_all.mean()
        }
    
    def _step2_stability_analysis(self, protein, intermediate_tp, final_tp, intermediate_data, final_data, meaningful_participants):
        """
        Step 2: Analyze stability (intermediate → final) for participants with meaningful early changes
        """
        if not meaningful_participants:
            return None
        
        # Get data for meaningful participants only
        intermediate_values = intermediate_data[intermediate_data['ID'].isin(meaningful_participants)].set_index('ID')[protein]
        final_values = final_data[final_data['ID'].isin(meaningful_participants)].set_index('ID')[protein]
        
        # Align data
        common_meaningful = set(intermediate_values.dropna().index).intersection(set(final_values.dropna().index))
        
        if len(common_meaningful) < 5:
            return None
        
        aligned_intermediate = intermediate_values.reindex(common_meaningful).dropna()
        aligned_final = final_values.reindex(aligned_intermediate.index).dropna()
        
        if len(aligned_intermediate) < 5:
            return None
        
        # CV ANALYSIS for stability
        final_cv = (aligned_final.std() / abs(aligned_final.mean()) * 100) if aligned_final.mean() != 0 else 100
        cv_stable = final_cv < 20.0
        
        # T-TEST for intermediate vs final (should be non-significant for stability)
        try:
            t_stat, p_value = stats.ttest_rel(aligned_final, aligned_intermediate)
            ttest_stable = p_value >= 0.05  # Non-significant = stable
        except:
            t_stat, p_value = 0, 1.0
            ttest_stable = True
        
        # INDIVIDUAL STABILITY: small changes from intermediate to final
        individual_changes = aligned_final - aligned_intermediate
        small_changes = abs(individual_changes) < 2.0  # Less than 2-fold change
        individual_stability = small_changes.mean()
        
        # OVERALL STABILITY SCORE
        stability_score = (
            0.4 * (1.0 if cv_stable else 0.0) +
            0.3 * (1.0 if ttest_stable else 0.0) +
            0.3 * individual_stability
        )
        
        return {
            'n_participants': len(aligned_intermediate),
            'final_cv': final_cv,
            'cv_stable': cv_stable,
            'ttest_stat': t_stat,
            'ttest_p_value': p_value,
            'ttest_stable': ttest_stable,
            'individual_stability': individual_stability,
            'stability_score': stability_score,
            'intermediate_mean': aligned_intermediate.mean(),
            'final_mean': aligned_final.mean(),
            'stability_change': aligned_final.mean() - aligned_intermediate.mean()
        }
    
    def _step3_prediction_analysis(self, protein, baseline_tp, intermediate_tp, final_tp, baseline_data, intermediate_data, final_data, meaningful_participants):
        """
        Step 3: Prediction analysis - Can early changes (baseline→intermediate) predict late changes (baseline→final)?
        """
        if not meaningful_participants:
            return None
        
        # Get data for meaningful participants
        baseline_values = baseline_data[baseline_data['ID'].isin(meaningful_participants)].set_index('ID')[protein]
        intermediate_values = intermediate_data[intermediate_data['ID'].isin(meaningful_participants)].set_index('ID')[protein]
        final_values = final_data[final_data['ID'].isin(meaningful_participants)].set_index('ID')[protein]
        
        # Find participants with all three timepoints
        common_all = (set(baseline_values.dropna().index) &
                     set(intermediate_values.dropna().index) &
                     set(final_values.dropna().index))
        
        if len(common_all) < 5:
            return None
        
        # Align data
        aligned_baseline = baseline_values.reindex(common_all).dropna()
        aligned_intermediate = intermediate_values.reindex(aligned_baseline.index).dropna()
        aligned_final = final_values.reindex(aligned_baseline.index).dropna()
        
        if len(aligned_baseline) < 5:
            return None
        
        # Calculate changes
        early_changes = aligned_intermediate - aligned_baseline  # baseline → intermediate
        late_changes = aligned_final - aligned_baseline        # baseline → final
        
        # PREDICTION CORRELATION
        try:
            if len(early_changes) > 1 and early_changes.std() > 1e-10 and late_changes.std() > 1e-10:
                prediction_correlation = np.corrcoef(early_changes, late_changes)[0, 1]
                if np.isnan(prediction_correlation):
                    prediction_correlation = 0
            else:
                prediction_correlation = 0
        except:
            prediction_correlation = 0
        
        # DIRECTION PERSISTENCE
        early_directions = np.sign(early_changes)
        late_directions = np.sign(late_changes)
        direction_persistence = (early_directions == late_directions).mean()
        
        # PREDICTION ACCURACY (based on correlation strength)
        prediction_accuracy = abs(prediction_correlation)
        
        # PREDICTION QUALITY
        if prediction_accuracy > 0.7:
            prediction_quality = "High"
        elif prediction_accuracy > 0.5:
            prediction_quality = "Moderate"
        else:
            prediction_quality = "Low"
        
        return {
            'n_participants': len(aligned_baseline),
            'prediction_correlation': prediction_correlation,
            'direction_persistence': direction_persistence,
            'prediction_accuracy': prediction_accuracy,
            'prediction_quality': prediction_quality,
            'early_changes': early_changes.values.tolist(),
            'late_changes': late_changes.values.tolist(),
            'mean_early_change': early_changes.mean(),
            'mean_late_change': late_changes.mean()
        }
    
    def _step4_overall_persistence_assessment(self, protein, meaningful_changes_result, stability_results, prediction_results):
        """
        Step 4: Overall persistence assessment combining all analyses
        """
        # Calculate average scores across all final timepoints
        stability_scores = [result['stability_score'] for result in stability_results.values()]
        prediction_accuracies = [result['prediction_accuracy'] for result in prediction_results.values()]
        
        avg_stability_score = np.mean(stability_scores) if stability_scores else 0
        avg_prediction_accuracy = np.mean(prediction_accuracies) if prediction_accuracies else 0
        
        # Count stable timepoints
        stable_timepoints = sum(1 for result in stability_results.values() 
                               if result['cv_stable'] and result['ttest_stable'])
        total_timepoints = len(stability_results)
        stability_proportion = stable_timepoints / total_timepoints if total_timepoints > 0 else 0
        
        # OVERALL PERSISTENCE SCORE
        overall_persistence_score = (
            0.4 * avg_stability_score +
            0.3 * avg_prediction_accuracy +
            0.3 * stability_proportion
        )
        
        # PERSISTENCE CATEGORY
        if overall_persistence_score > 0.7 and avg_prediction_accuracy > 0.6:
            category = "Highly Persistent & Predictive"
        elif overall_persistence_score > 0.5 and avg_prediction_accuracy > 0.4:
            category = "Moderately Persistent & Predictive"
        elif overall_persistence_score > 0.5:
            category = "Persistent but Low Prediction"
        elif avg_prediction_accuracy > 0.5:
            category = "Predictive but Not Persistent"
        else:
            category = "Low Persistence & Prediction"
        
        # WHEN DO CHANGES STOP BEING STABLE?
        stability_breakdown_tp = None
        for tp in sorted(stability_results.keys()):
            if not (stability_results[tp]['cv_stable'] and stability_results[tp]['ttest_stable']):
                stability_breakdown_tp = tp
                break
        
        return {
            # Step 1 results
            'meaningful_changes': meaningful_changes_result,
            
            # Step 2 & 3 results by timepoint
            'stability_by_timepoint': stability_results,
            'prediction_by_timepoint': prediction_results,
            
            # Overall assessment
            'overall_persistence_score': overall_persistence_score,
            'avg_stability_score': avg_stability_score,
            'avg_prediction_accuracy': avg_prediction_accuracy,
            'stability_proportion': stability_proportion,
            'stable_timepoints': stable_timepoints,
            'total_timepoints': total_timepoints,
            'category': category,
            
            # When changes stop being stable
            'stability_breakdown_timepoint': stability_breakdown_tp,
            'stable_until': stability_breakdown_tp - 1 if stability_breakdown_tp else max(stability_results.keys()),
            
            # Summary metrics
            'n_meaningful_participants': meaningful_changes_result['n_meaningful_participants'],
            'proportion_meaningful': meaningful_changes_result['proportion_meaningful'],
            'across_significant': meaningful_changes_result['across_significant']
        }
    
    def analyze_within_across_for_persistence(self, baseline_tp, intermediate_tp, final_tps, significant_proteins=None):
        """
        Analyze both within and across individual patterns for persistence analysis
        This provides comparative analysis for radar plots and visualizations
        
        Args:
            significant_proteins: List of proteins to analyze (if None, analyzes all proteins)
        """
        results = {}
        
        # Use only significant proteins if provided, otherwise use all proteins
        if significant_proteins is not None:
            proteins_to_analyze = [p for p in significant_proteins if p in self.protein_cols]
        else:
            proteins_to_analyze = self.protein_cols
        
        for final_tp in final_tps:
            # Get data for the timepoints
            baseline_data = self.merged_data[self.merged_data['Timepoint'] == baseline_tp]
            intermediate_data = self.merged_data[self.merged_data['Timepoint'] == intermediate_tp]
            final_data = self.merged_data[self.merged_data['Timepoint'] == final_tp]
            
            within_across_results = {}
            
            for protein in proteins_to_analyze:
                # WITHIN-INDIVIDUAL ANALYSIS
                within_result = self._analyze_within_individual_persistence(
                    protein, baseline_data, intermediate_data, final_data
                )
                
                # ACROSS-INDIVIDUAL ANALYSIS  
                across_result = self._analyze_across_individual_persistence(
                    protein, baseline_data, intermediate_data, final_data
                )
                
                if within_result and across_result:
                    within_across_results[protein] = {
                        'within': within_result,
                        'across': across_result
                    }
            
            results[final_tp] = within_across_results
        
        return results
    
    def _analyze_within_individual_persistence(self, protein, baseline_data, intermediate_data, final_data):
        """Within-individual persistence analysis for single protein"""
        # Find common participants across all timepoints
        common_ids = (set(baseline_data['ID']) & 
                     set(intermediate_data['ID']) & 
                     set(final_data['ID']))
        
        if len(common_ids) < 10:
            return None
        
        # Get aligned values
        baseline_vals = baseline_data[baseline_data['ID'].isin(common_ids)].set_index('ID')[protein]
        intermediate_vals = intermediate_data[intermediate_data['ID'].isin(common_ids)].set_index('ID')[protein]
        final_vals = final_data[final_data['ID'].isin(common_ids)].set_index('ID')[protein]
        
        # Align and clean
        aligned_baseline = baseline_vals.reindex(common_ids).dropna()
        aligned_intermediate = intermediate_vals.reindex(aligned_baseline.index).dropna()
        aligned_final = final_vals.reindex(aligned_baseline.index).dropna()
        
        if len(aligned_baseline) < 10:
            return None
        
        # Calculate individual changes
        early_changes = aligned_intermediate - aligned_baseline  # baseline → intermediate
        late_changes = aligned_final - aligned_baseline          # baseline → final
        stability_changes = aligned_final - aligned_intermediate  # intermediate → final
        
        # Only analyze meaningful early changes (≥2 units)
        meaningful_mask = np.abs(early_changes) >= 2.0
        if meaningful_mask.sum() < 5:
            return None
        
        meaningful_early = early_changes[meaningful_mask]
        meaningful_late = late_changes[meaningful_mask]
        meaningful_stability = stability_changes[meaningful_mask]
        
        # Within-individual metrics
        # 1. Direction persistence
        direction_same = (np.sign(meaningful_early) == np.sign(meaningful_late)).mean()
        
        # 2. Magnitude correlation (prediction)
        if len(meaningful_early) > 1:
            magnitude_correlation = np.corrcoef(meaningful_early, meaningful_late)[0, 1]
            if np.isnan(magnitude_correlation):
                magnitude_correlation = 0
        else:
            magnitude_correlation = 0
        
        # 3. Individual stability (small changes after intermediate)
        individual_stability = (np.abs(meaningful_stability) < 2.0).mean()
        
        # 4. Consistency score
        consistency_score = (direction_same + individual_stability) / 2
        
        # 5. Prediction accuracy
        prediction_accuracy = abs(magnitude_correlation)
        
        return {
            'n_participants': len(aligned_baseline),
            'n_meaningful': meaningful_mask.sum(),
            'meaningful_proportion': meaningful_mask.mean(),
            'direction_persistence': direction_same,
            'magnitude_correlation': magnitude_correlation,
            'individual_stability': individual_stability,
            'consistency_score': consistency_score,
            'prediction_accuracy': prediction_accuracy,
            'early_changes': meaningful_early.values.tolist(),
            'late_changes': meaningful_late.values.tolist(),
            'stability_changes': meaningful_stability.values.tolist(),
            'mean_early_change': meaningful_early.mean(),
            'mean_late_change': meaningful_late.mean(),
            'std_early_change': meaningful_early.std(),
            'std_late_change': meaningful_late.std()
        }
    
    def _analyze_across_individual_persistence(self, protein, baseline_data, intermediate_data, final_data):
        """Across-individual persistence analysis for single protein"""
        # Get all available data (different participants allowed)
        baseline_vals = baseline_data[protein].dropna()
        intermediate_vals = intermediate_data[protein].dropna()
        final_vals = final_data[protein].dropna()
        
        if len(baseline_vals) < 10 or len(intermediate_vals) < 10 or len(final_vals) < 10:
            return None
        
        # Population-level changes
        baseline_mean = baseline_vals.mean()
        intermediate_mean = intermediate_vals.mean()
        final_mean = final_vals.mean()
        
        early_change = intermediate_mean - baseline_mean      # baseline → intermediate
        late_change = final_mean - baseline_mean             # baseline → final
        stability_change = final_mean - intermediate_mean    # intermediate → final
        
        # Across-individual metrics
        # 1. Direction persistence (population level)
        early_direction = np.sign(early_change)
        late_direction = np.sign(late_change)
        direction_same = (early_direction == late_direction)
        
        # 2. CV analysis for stability
        baseline_cv = (baseline_vals.std() / abs(baseline_vals.mean()) * 100) if baseline_vals.mean() != 0 else 100
        intermediate_cv = (intermediate_vals.std() / abs(intermediate_vals.mean()) * 100) if intermediate_vals.mean() != 0 else 100
        final_cv = (final_vals.std() / abs(final_vals.mean()) * 100) if final_vals.mean() != 0 else 100
        
        # 3. Stability based on CV < 20%
        cv_stable = final_cv < 20.0
        
        # 4. Statistical significance
        from scipy.stats import ttest_ind
        try:
            _, p_baseline_intermediate = ttest_ind(intermediate_vals, baseline_vals)
            _, p_baseline_final = ttest_ind(final_vals, baseline_vals)
            _, p_intermediate_final = ttest_ind(final_vals, intermediate_vals)
            
            early_significant = p_baseline_intermediate < 0.05
            late_significant = p_baseline_final < 0.05
            stability_significant = p_intermediate_final < 0.05
        except:
            early_significant = False
            late_significant = False
            stability_significant = False
            p_baseline_intermediate = 1.0
            p_baseline_final = 1.0
            p_intermediate_final = 1.0
        
        # 5. Population predictability (early change predicts late change direction)
        prediction_accuracy = 1.0 if direction_same else 0.0
        
        # 6. Consistency score (combines direction + CV stability)
        consistency_score = (float(direction_same) + float(cv_stable)) / 2
        
        return {
            'n_baseline': len(baseline_vals),
            'n_intermediate': len(intermediate_vals), 
            'n_final': len(final_vals),
            'baseline_mean': baseline_mean,
            'intermediate_mean': intermediate_mean,
            'final_mean': final_mean,
            'early_change': early_change,
            'late_change': late_change,
            'stability_change': stability_change,
            'direction_persistence': float(direction_same),
            'baseline_cv': baseline_cv,
            'intermediate_cv': intermediate_cv,
            'final_cv': final_cv,
            'cv_stable': cv_stable,
            'early_significant': early_significant,
            'late_significant': late_significant,
            'stability_significant': stability_significant,
            'p_baseline_intermediate': p_baseline_intermediate,
            'p_baseline_final': p_baseline_final,
            'p_intermediate_final': p_intermediate_final,
            'consistency_score': consistency_score,
            'prediction_accuracy': prediction_accuracy
        }
    
    def _get_effect_magnitude(self, cohen_d):
        """Convert Cohen's d to magnitude category"""
        return get_effect_magnitude(cohen_d)
    
    def create_venn_diagram_data(self, within_results, across_results):
        """Create data for Venn diagram comparing within and across significant results"""
        within_significant = set(k for k, v in within_results.items() if v.get('significant', False))
        across_significant = set(k for k, v in across_results.items() if v.get('significant', False))
        
        only_within = within_significant - across_significant
        only_across = across_significant - within_significant
        both = within_significant & across_significant
        
        return {
            'only_within': only_within,
            'only_across': only_across,
            'both': both,
            'within_total': len(within_significant),
            'across_total': len(across_significant),
            'overlap': len(both)
        }

def main():
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = EnhancedOmicsAnalyzer()
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    # Main header
    st.markdown('<h1 class="main-header">🧬 Enhanced Omics Data Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar for file uploads
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploads
        omics_file = st.file_uploader(
            "Upload Omics Data",
            type=['csv', 'xlsx'],
            help="CSV or Excel file with ID (1st column), Timepoint (2nd column), and biomarkers"
        )
        
        demographics_file = st.file_uploader(
            "Upload Demographics Data", 
            type=['csv', 'xlsx'],
            help="CSV/Excel: Col 1=ID, Col 4=sex, Col 5=age, Col 6-11=timepoints 1-6"
        )
        
        # Load data button
        if st.button("🚀 Load Data", type="primary"):
            if omics_file and demographics_file:
                with st.spinner("Loading and processing data..."):
                    success, message = st.session_state.analyzer.load_data(omics_file, demographics_file)
                    if success:
                        st.session_state.data_loaded = True
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("Please upload both omics and demographics files.")
        
        # Data status
        if st.session_state.data_loaded:
            st.success("✅ Data loaded successfully!")
        else:
            st.info("ℹ️ Please upload data files to begin analysis.")
    
    # Main content with tabs
    if st.session_state.data_loaded:
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview", 
            "👥 Demographics Analysis", 
            "🔍 Biomarker Analysis\\Significant Analysis", 
            "📈 Biomarker Analysis\\Persistent Change-Long Term Effect"
        ])
        
        # Tab 1: Data Overview (keeping the existing implementation)
        with tab1:
            st.markdown('<div class="tab-header">📊 Data Overview & Summary</div>', unsafe_allow_html=True)
            
            # Get data summary
            summary = st.session_state.analyzer.get_data_summary()
            
            if summary:
                # Summary metrics at the top
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Participants", summary['omics']['n_participants'])
                    st.metric("Total Samples", summary['omics']['n_samples'])
                
                with col2:
                    st.metric("Biomarkers", summary['omics']['n_biomarkers'])
                    st.metric("Timepoints", summary['omics']['n_timepoints'])
                
                with col3:
                    st.metric("Demographics Variables", summary['demographics']['n_variables'])
                    st.metric("Data Completeness", f"{summary['merged']['completeness_rate']:.1f}%")
                
                with col4:
                    st.metric("Omics Missing Data", f"{summary['omics']['missing_data_pct']:.1f}%")
                    st.metric("Demographics Missing Data", f"{summary['demographics']['missing_data_pct']:.1f}%")
                
                st.divider()
                
                # Let user select which dataset to preview if multiple timepoints
                if len(st.session_state.analyzer.timepoints) > 1:
                    selected_timepoint = st.selectbox(
                        "Select Timepoint for Detailed Overview", 
                        st.session_state.analyzer.timepoints,
                        key="omics_preview_select"
                    )
                    selected_omics_df = st.session_state.analyzer.merged_data[
                        st.session_state.analyzer.merged_data['Timepoint'] == selected_timepoint
                    ]
                else:
                    selected_timepoint = st.session_state.analyzer.timepoints[0]
                    selected_omics_df = st.session_state.analyzer.merged_data
                
                # Create sub-tabs for different views
                omics_preview_tab1, omics_preview_tab2, omics_preview_tab3 = st.tabs(["Overview", "Distributions", "Omics Categories"])
                
                # Tab 1: Overview
                with omics_preview_tab1:
                    st.subheader(f"Omics Data Overview: Timepoint {selected_timepoint}")
                    
                    # Split into columns
                    overview_col1, overview_col2 = st.columns([1, 1])
                    
                    with overview_col1:
                        # Basic dataset information
                        st.write(f"**Number of Records:** {selected_omics_df.shape[0]}")
                        st.write(f"**Number of Variables:** {selected_omics_df.shape[1]}")
                        
                        # Date range if available
                        if 'Date' in selected_omics_df.columns:
                            min_date = selected_omics_df['Date'].min().strftime('%Y-%m-%d')
                            max_date = selected_omics_df['Date'].max().strftime('%Y-%m-%d')
                            st.write(f"**Collection Period:** {min_date} to {max_date}")
                        
                        # Get number of patients
                        if 'ID' in selected_omics_df.columns:
                            num_patients = selected_omics_df['ID'].nunique()
                            st.write(f"**Number of Participants:** {num_patients}")
                        
                        # Missing data summary
                        protein_cols = st.session_state.analyzer.protein_cols
                        if protein_cols:
                            missing_percent = round((selected_omics_df[protein_cols].isna().sum().sum() / 
                                                   (selected_omics_df.shape[0] * len(protein_cols)) * 100), 1)
                            st.write(f"**Missing Data:** {missing_percent}%")
                    
                    with overview_col2:
                        # Categories of biomarkers
                        categories = {
                            'Proteins Synaptic Plasticity': ['BDNF', 'CREB1', 'CAMK2A', 'CAMK2B', "SYN1", "SYN2", "RPLP1"],
                            'Proteins Glial Support': ['MBP', 'MOG', 'PLP1', 'S100B', "S100A13"],
                            'Proteins Structural Plasticity': ['GAP43', 'ACTB', 'TUBB3', 'CFL1'],
                            'Proteins Neurogenesis': ['NES', 'SOX2', 'NEUROD1', 'DCX'],
                            'Proteins ECM Remodeling': ['GRIN1', "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GABRA1", "GABRB2", "GABRG2", "CACNA1C", "INSYN1", "ARHGEF9", "GPHN"],
                            'Metabolic Hormones': ["DHEA", 'Cortisol', 'Corticosterone'],
                            'Inflammatory': ['IL6', 'IL10', 'IL11', 'hscrp', 'CRP', 'TNF_alpha'],
                            'Electrolytes': ['Calcium', 'Na', 'K', 'co2'],
                            'Blood': ['hemoglobin', 'WBC', 'albumin']
                        }
                        
                        st.write("**Biomarker Categories:**")
                        for category, markers in categories.items():
                            available = [m for m in markers if m in selected_omics_df.columns]
                            if available:
                                st.write(f"- **{category}:** {', '.join(available)} ({len(available)} markers)")
                    
                    # Data preview section
                    st.subheader("Data Preview")
                    st.dataframe(selected_omics_df.head(10), use_container_width=True)
                    
                    # Summary statistics for numeric columns
                    st.subheader("Summary Statistics")
                    numeric_cols = selected_omics_df.select_dtypes(include=['float64', 'int64']).columns
                    numeric_cols = [col for col in numeric_cols if col not in ['ID', 'Timepoint']]
                    
                    if numeric_cols:
                        # Create a formatted summary table
                        omics_summary = pd.DataFrame()
                        
                        for col in numeric_cols:
                            col_data = selected_omics_df[col]
                            col_summary = pd.DataFrame({
                                'Biomarker': [col],
                                'Mean ± SD': [f"{col_data.mean():.3f} ± {col_data.std():.3f}"],
                                'Median [IQR]': [f"{col_data.median():.3f} [{col_data.quantile(0.25):.3f}-{col_data.quantile(0.75):.3f}]"],
                                'Range': [f"{col_data.min():.3f} - {col_data.max():.3f}"],
                                'Missing (%)': [f"{col_data.isna().sum()} ({col_data.isna().sum()/len(col_data)*100:.1f}%)"]
                            })
                            omics_summary = pd.concat([omics_summary, col_summary])
                        
                        # Reset index for clean display
                        omics_summary = omics_summary.reset_index(drop=True)
                        
                        # Display the formatted summary
                        st.dataframe(omics_summary, use_container_width=True)
                
                # Tab 2: Distributions
                with omics_preview_tab2:
                    st.subheader("Biomarker Distributions")
                    
                    # Get numeric columns for visualization
                    numeric_cols = selected_omics_df.select_dtypes(include=['float64', 'int64']).columns
                    numeric_cols = [col for col in numeric_cols if col not in ['ID', 'Timepoint']]
                    
                    if numeric_cols:
                        # Allow selection of biomarker category
                        categories = {
                            'All Biomarkers': numeric_cols,
                            'Proteins Synaptic Plasticity': [col for col in numeric_cols if col in ['BDNF', 'CREB1', 'CAMK2A', 'CAMK2B', "SYN1", "SYN2", "RPLP1"]],
                            'Proteins Glial Support': [col for col in numeric_cols if col in ['MBP', 'MOG', 'PLP1', 'S100B', "S100A13"]],
                            'Proteins Structural Plasticity': [col for col in numeric_cols if col in ['GAP43', 'ACTB', 'TUBB3', 'CFL1']],
                            'Proteins Neurogenesis': [col for col in numeric_cols if col in ['NES', 'SOX2', 'NEUROD1', 'DCX']],
                            'Proteins ECM Remodeling': [col for col in numeric_cols if col in ['GRIN1', "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GABRA1", "GABRB2", "GABRG2", "CACNA1C", "INSYN1", "ARHGEF9", "GPHN"]],
                            'Metabolic Hormones': [col for col in numeric_cols if col in ["DHEA", 'Cortisol', 'Corticosterone']],                             
                            'Inflammatory': [col for col in numeric_cols if col in ['IL6', 'IL10', 'IL11', 'hscrp', 'CRP', 'TNF_alpha']],
                            'Electrolytes': [col for col in numeric_cols if col in ['Calcium', 'Na', 'K', 'co2']],
                            'Blood': [col for col in numeric_cols if col in ['hemoglobin', 'WBC', 'albumin']]
                        }
                        
                        # Filter out empty categories
                        categories = {k: v for k, v in categories.items() if v}
                        
                        selected_category = st.selectbox(
                            "Select Biomarker Category",
                            categories.keys(),
                            key="biomarker_category"
                        )
                        
                        if selected_category in categories and categories[selected_category]:
                            # Create a multiselect for biomarkers in that category
                            selected_biomarkers = st.multiselect(
                                "Select Biomarkers to Display",
                                categories[selected_category],
                                default=[categories[selected_category][0]] if categories[selected_category] else [],
                                key="selected_biomarkers"
                            )
                            
                            if selected_biomarkers:
                                # Create distribution plots
                                n_plots = len(selected_biomarkers)
                                n_cols = min(3, n_plots)
                                n_rows = (n_plots + n_cols - 1) // n_cols
                                
                                fig = make_subplots(
                                    rows=n_rows, cols=n_cols,
                                    subplot_titles=selected_biomarkers,
                                    vertical_spacing=0.08
                                )
                                
                                for idx, biomarker in enumerate(selected_biomarkers):
                                    row = idx // n_cols + 1
                                    col = idx % n_cols + 1
                                    
                                    values = selected_omics_df[biomarker].dropna()
                                    if len(values) > 0:
                                        fig.add_trace(
                                            go.Histogram(x=values, name=biomarker, showlegend=False),
                                            row=row, col=col
                                        )
                                
                                fig.update_layout(height=300*n_rows, title_text="Biomarker Distributions")
                                st.plotly_chart(fig, use_container_width=True)
                
                # Tab 3: Omics Categories
                with omics_preview_tab3:
                    st.subheader("Omics Categories Analysis")
                    
                    # Categories of biomarkers with detailed analysis
                    categories = {
                        'Proteins Synaptic Plasticity': ['BDNF', 'CREB1', 'CAMK2A', 'CAMK2B', "SYN1", "SYN2", "RPLP1"],
                        'Proteins Glial Support': ['MBP', 'MOG', 'PLP1', 'S100B', "S100A13"],
                        'Proteins Structural Plasticity': ['GAP43', 'ACTB', 'TUBB3', 'CFL1'],
                        'Proteins Neurogenesis': ['NES', 'SOX2', 'NEUROD1', 'DCX'],
                        'Proteins ECM Remodeling': ['GRIN1', "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GABRA1", "GABRB2", "GABRG2", "CACNA1C", "INSYN1", "ARHGEF9", "GPHN"],
                        'Metabolic Hormones': ["DHEA", 'Cortisol', 'Corticosterone'],
                        'Inflammatory': ['IL6', 'IL10', 'IL11', 'hscrp', 'CRP', 'TNF_alpha'],
                        'Electrolytes': ['Calcium', 'Na', 'K', 'co2'],
                        'Blood': ['hemoglobin', 'WBC', 'albumin']
                    }
                    
                    # Category coverage analysis
                    category_coverage = []
                    for category, markers in categories.items():
                        available = [m for m in markers if m in selected_omics_df.columns]
                        coverage = len(available) / len(markers) * 100 if markers else 0
                        category_coverage.append({
                            'Category': category,
                            'Total_Markers': len(markers),
                            'Available_Markers': len(available),
                            'Coverage_Percent': coverage,
                            'Available_List': ', '.join(available) if available else 'None'
                        })
                    
                    coverage_df = pd.DataFrame(category_coverage)
                    
                    # Display coverage table
                    st.subheader("Category Coverage Analysis")
                    st.dataframe(coverage_df, use_container_width=True)
                    
                    # Coverage visualization
                    fig_coverage = px.bar(
                        coverage_df, 
                        x='Category', 
                        y='Coverage_Percent',
                        title='Biomarker Category Coverage',
                        labels={'Coverage_Percent': 'Coverage (%)'},
                        color='Coverage_Percent',
                        color_continuous_scale='viridis'
                    )
                    fig_coverage.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_coverage, use_container_width=True)
                    
                    # Detailed category analysis
                    st.subheader("Detailed Category Analysis")
                    
                    for category, markers in categories.items():
                        available = [m for m in markers if m in selected_omics_df.columns]
                        if available:
                            with st.expander(f"{category} ({len(available)}/{len(markers)} markers)"):
                                st.write(f"**Available markers:** {', '.join(available)}")
                                
                                # Basic statistics for available markers
                                category_data = selected_omics_df[available]
                                category_stats = category_data.describe().T
                                st.dataframe(category_stats, use_container_width=True)
                                
                                # Correlation heatmap if more than 1 marker
                                if len(available) > 1:
                                    st.write("**Correlation Matrix:**")
                                    corr_matrix = category_data.corr()
                                    fig_corr = px.imshow(
                                        corr_matrix,
                                        title=f"{category} - Correlation Matrix",
                                        color_continuous_scale='RdBu_r',
                                        aspect='auto'
                                    )
                                    st.plotly_chart(fig_corr, use_container_width=True)
        
        # Tab 2: Demographics Analysis (keeping the existing implementation)
        with tab2:
            st.markdown('<div class="tab-header">👥 Demographics Analysis</div>', unsafe_allow_html=True)
            
            # Build available demographic variables
            available_demo_vars = []
            if (hasattr(st.session_state.analyzer, 'sex_col') and 
                st.session_state.analyzer.sex_col and 
                st.session_state.analyzer.sex_col in st.session_state.analyzer.merged_data.columns):
                available_demo_vars.append(st.session_state.analyzer.sex_col)
            if (hasattr(st.session_state.analyzer, 'age_col') and 
                st.session_state.analyzer.age_col and 
                st.session_state.analyzer.age_col in st.session_state.analyzer.merged_data.columns):
                available_demo_vars.append(st.session_state.analyzer.age_col)
            if 'demographic_value' in st.session_state.analyzer.merged_data.columns:
                available_demo_vars.append('demographic_value (timepoint measurements)')
            if hasattr(st.session_state.analyzer, 'demographic_cols'):
                available_cols = [col for col in st.session_state.analyzer.demographic_cols 
                                if col in st.session_state.analyzer.merged_data.columns]
                available_demo_vars.extend(available_cols)
            
            # Select demographic variables for analysis
            if available_demo_vars:
                selected_demo_vars = st.multiselect(
                    "Select demographic variables to analyze:",
                    available_demo_vars,
                    default=available_demo_vars[:3] if len(available_demo_vars) >= 3 else available_demo_vars
                )
                
                if selected_demo_vars:
                    # Analyze demographics
                    with st.spinner("Analyzing demographics across timepoints..."):
                        demo_results = st.session_state.analyzer.analyze_demographics_by_timepoint(selected_demo_vars)
                    
                    # Display results
                    for var in selected_demo_vars:
                        # Handle display name vs actual column name
                        actual_var = var.replace(' (timepoint measurements)', '') if 'timepoint measurements' in var else var
                        display_name = var
                        
                        if actual_var in demo_results:
                            st.subheader(f"📊 Analysis: {display_name}")
                            
                            result = demo_results[actual_var]
                            
                            if result.get('type') == 'categorical':
                                # Handle categorical variables (like sex)
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # Create stacked bar chart for categorical data
                                    crosstab_data = result['crosstab'].copy()
                                    
                                    # Safely remove 'All' totals if they exist
                                    if 'All' in crosstab_data.index:
                                        crosstab_data = crosstab_data.drop('All', axis=0)
                                    if 'All' in crosstab_data.columns:
                                        crosstab_data = crosstab_data.drop('All', axis=1)
                                    
                                    # Convert to long format for plotting
                                    plot_data = []
                                    for timepoint in crosstab_data.index:
                                        for category in crosstab_data.columns:
                                            plot_data.append({
                                                'Timepoint': timepoint,
                                                'Category': category,
                                                'Proportion': crosstab_data.loc[timepoint, category]
                                            })
                                    
                                    plot_df = pd.DataFrame(plot_data)
                                    
                                    if not plot_df.empty:
                                        fig = px.bar(
                                            plot_df,
                                            x='Timepoint',
                                            y='Proportion',
                                            color='Category',
                                            title=f'{display_name} Distribution Across Timepoints',
                                            labels={'Proportion': 'Proportion'}
                                        )
                                        fig.update_layout(height=400)
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.warning("No data available for plotting")
                                
                                with col2:
                                    # Statistical summary for categorical
                                    st.write("**Statistical Summary:**")
                                    if result['chi2_p_value'] is not None:
                                        st.metric("Chi-square p-value", f"{result['chi2_p_value']:.4f}")
                                        significance = "Significant" if result['significant'] else "Not Significant"
                                        st.metric("Result", significance)
                                    
                                    st.write("**Proportions by Timepoint:**")
                                    # Display crosstab without 'All' totals if they exist
                                    display_crosstab = result['crosstab'].copy()
                                    if 'All' in display_crosstab.index:
                                        display_crosstab = display_crosstab.drop('All', axis=0)
                                    st.dataframe(display_crosstab, use_container_width=True)
                            
                            else:
                                # Handle numeric variables (age, demographic_value, etc.)
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    # Create box plot for numeric data
                                    fig = px.box(
                                        st.session_state.analyzer.merged_data,
                                        x='Timepoint',
                                        y=actual_var,
                                        title=f'{display_name} Distribution Across Timepoints',
                                        points="all"
                                    )
                                    fig.update_layout(height=400)
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                with col2:
                                    # Statistical summary for numeric
                                    st.write("**Statistical Summary:**")
                                    if result['anova_p'] is not None:
                                        st.metric("ANOVA p-value", f"{result['anova_p']:.4f}")
                                        significance = "Significant" if result['significant'] else "Not Significant"
                                        st.metric("Result", significance)
                                    
                                    st.write("**Summary Statistics:**")
                                    st.dataframe(result['summary_stats'], use_container_width=True)
                            
                            st.divider()
                else:
                    st.info("Please select at least one demographic variable to analyze.")
            else:
                st.warning("No demographic variables found. Please check your demographics data structure.")
        
        # Tab 3: Enhanced Significant Analysis
        with tab3:
            st.markdown('<div class="tab-header">🔍 Biomarker Analysis - Significant Analysis</div>', unsafe_allow_html=True)
            
            # Enhanced explanation with corrections for RELATIVE CHANGES
            st.markdown('<div class="significance-box">', unsafe_allow_html=True)
            st.markdown("**Enhanced Analysis with Corrected Statistical Interpretations (RELATIVE CHANGES):**")
            st.markdown("- **Group-Level Test**: Paired t-test testing if average change across all participants ≠ 0")
            st.markdown("- **Individual-Level Test**: Binomial test for proportion of participants with meaningful **relative changes**")
            st.markdown("- **Meaningful Change Threshold**: **Fold changes** ≥threshold (e.g., 2.0x = 100% increase) OR ≤1/threshold (e.g., 0.5x = 50% decrease)")
            st.markdown("- **Biological Relevance**: Relative changes are more meaningful across different baseline levels")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Timepoint selection
            col1, col2, col3 = st.columns(3)
            
            with col1:
                baseline_tp = st.selectbox(
                    "Baseline Timepoint:",
                    st.session_state.analyzer.timepoints,
                    index=0,
                    key="sig_baseline_corrected"
                )
            
            with col2:
                target_tps = st.multiselect(
                    "Target Timepoints for Comparison:",
                    [tp for tp in st.session_state.analyzer.timepoints if tp != baseline_tp],
                    default=[tp for tp in st.session_state.analyzer.timepoints if tp != baseline_tp][:2],
                    key="sig_targets_corrected"
                )
            
            with col3:
                meaningful_threshold = st.number_input(
                    "Meaningful Change Threshold (Fold Change):",
                    min_value=1.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    help="Relative change considered meaningful (e.g., 2.0 = 100% increase, 0.5 = 50% decrease)",
                    key="meaningful_threshold"
                )
            
            if st.button("🔍 Run Enhanced Corrected Analysis", type="primary", key="run_sig_analysis_corrected"):
                if target_tps:
                    with st.spinner("Running enhanced corrected analysis..."):
                        # Run enhanced significance analysis with corrected statistics
                        sig_results = st.session_state.analyzer.analyze_significant_changes_enhanced(
                            baseline_tp, target_tps, meaningful_threshold
                        )
                        
                        # Store results in session state for use in persistence analysis
                        st.session_state.significant_results = sig_results
                    
                    # Display results for each target timepoint
                    for target_tp in target_tps:
                        if target_tp in sig_results:
                            st.subheader(f"📊 Enhanced Analysis: {baseline_tp} → {target_tp}")
                            
                            within_results = sig_results[target_tp]['within']
                            across_results = sig_results[target_tp]['across']
                            
                            # Get baseline and target data for this comparison (needed for visualizations)
                            baseline_data = st.session_state.analyzer.merged_data[
                                st.session_state.analyzer.merged_data['Timepoint'] == baseline_tp
                            ]
                            target_data = st.session_state.analyzer.merged_data[
                                st.session_state.analyzer.merged_data['Timepoint'] == target_tp
                            ]
                            
                            # Enhanced Venn diagram with corrected significance
                            venn_data = st.session_state.analyzer.create_venn_diagram_data(within_results, across_results)
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Within-Individual Significant", venn_data['within_total'])
                            with col2:
                                st.metric("Across-Individual Significant", venn_data['across_total'])
                            with col3:
                                st.metric("Overlap (Both)", venn_data['overlap'])
                            with col4:
                                total_unique = len(venn_data['only_within'] | venn_data['only_across'] | venn_data['both'])
                                st.metric("Total Unique Significant", total_unique)
                            
                            # Corrected analysis tabs with proper organization
                            within_tab, across_tab, interpretation_tab = st.tabs([
                                "👤 Within-Individual Analysis", 
                                "🌐 Across-Individual (Group Level) Analysis", 
                                "📋 Interpretation Guide"
                            ])
                            
                            # WITHIN-INDIVIDUAL ANALYSIS TAB - Individual trajectories and meaningful changes
                            with within_tab:
                                st.info("👤 **Within-Individual Analysis**: Individual change patterns and meaningful relative changes for same participants")
                                
                                # Focus on individual-level meaningful changes only
                                individual_significant = {k: v for k, v in within_results.items() if v.get('individual_significant', False)}
                                
                                st.subheader("📊 Individual Meaningful Relative Changes Analysis")
                                st.info(f"**Question**: Do individual participants show meaningful relative changes (≥{meaningful_threshold:.1f}x increase OR ≤{(1.0/meaningful_threshold):.1f}x decrease)?")
                                
                                if individual_significant:
                                    st.success(f"Found {len(individual_significant)} proteins with significant individual meaningful relative changes!")
                                    
                                    # Create summary table for individual-significant
                                    individual_summary = []
                                    for protein, data in list(individual_significant.items()): #[:500]
                                        individual_summary.append({
                                            'Protein': protein,
                                            'N_Participants': data['n_participants'],
                                            'Meaningful_Threshold': data.get('meaningful_threshold_description', f"≥{data['meaningful_threshold']:.1f}x"),
                                            'N_Meaningful_Changes': f"{data['n_meaningful_changes']}/{data['n_participants']}",
                                            'Proportion_Meaningful': f"{data['proportion_meaningful']:.3f}",
                                            'Binomial_P_value': f"{data['individual_p_value']:.4f}",
                                            'Meaningful_Increased': data['n_meaningful_increased'],
                                            'Meaningful_Decreased': data['n_meaningful_decreased'],
                                            'Mean_Fold_Change': f"{data.get('mean_fold_change', 1.0):.2f}x",
                                            'Individual_Consistency': f"{data['meaningful_consistency']:.2f}"
                                        })
                                    
                                    individual_df = pd.DataFrame(individual_summary)
                                    st.dataframe(individual_df, use_container_width=True)
                                    
                                    st.info(f"""
                                    **Individual-Level Relative Change Metrics:**
                                    - **Meaningful_Threshold**: ≥{meaningful_threshold:.1f}x increase (e.g., 2.0x = 100% increase) OR ≤{(1.0/meaningful_threshold):.1f}x decrease (e.g., 0.5x = 50% decrease)
                                    - **N_Meaningful_Changes**: Count of participants with meaningful relative changes
                                    - **Proportion_Meaningful**: Fraction with meaningful relative changes
                                    - **Binomial_P_value**: Tests if proportion significantly > 50% (chance)
                                    - **Mean_Fold_Change**: Average fold change across all participants
                                    - **Individual_Consistency**: Proportion moving in dominant direction
                                    """)
                                    
                                    # Individual trajectory visualization
                                    if len(individual_significant) > 0:
                                        st.subheader("📈 Individual Relative Change Patterns")
                                        
                                        selected_protein_ind = st.selectbox(
                                            "Select protein to view individual relative changes:",
                                            list(individual_significant.keys())[:10],
                                            key=f"individual_protein_select_{target_tp}"
                                        )
                                        
                                        if selected_protein_ind:
                                            protein_data = individual_significant[selected_protein_ind]
                                            
                                            # Get fold changes and create meaningful mask
                                            fold_changes = protein_data['fold_changes']
                                            meaningful_increases_mask = [x >= meaningful_threshold for x in fold_changes]
                                            meaningful_decreases_mask = [x <= (1.0/meaningful_threshold) for x in fold_changes]
                                            
                                            # Create figure with fold changes
                                            fig_meaningful = go.Figure()
                                            
                                            # Add fold changes as bars
                                            participant_numbers = list(range(1, len(fold_changes) + 1))
                                            colors = []
                                            
                                            for i, (fold_change, is_increase, is_decrease) in enumerate(zip(fold_changes, meaningful_increases_mask, meaningful_decreases_mask)):
                                                if is_increase:
                                                    colors.append('#FF6B6B')  # Meaningful increase
                                                elif is_decrease:
                                                    colors.append('#4ECDC4')  # Meaningful decrease 
                                                else:
                                                    colors.append('#95A5A6')  # Not meaningful
                                            
                                            fig_meaningful.add_trace(go.Bar(
                                                x=participant_numbers,
                                                y=fold_changes,
                                                marker_color=colors,
                                                text=[f'{x:.2f}x' for x in fold_changes],
                                                textposition='auto',
                                                hovertemplate='Participant %{x}<br>Fold Change: %{y:.2f}x<extra></extra>',
                                                name='Fold Changes'
                                            ))
                                            
                                            # Add meaningful change threshold lines
                                            fig_meaningful.add_hline(
                                                y=meaningful_threshold, 
                                                line_dash="dash", 
                                                line_color="green",
                                                annotation_text=f"Meaningful Increase: ≥{meaningful_threshold:.1f}x"
                                            )
                                            fig_meaningful.add_hline(
                                                y=(1.0/meaningful_threshold), 
                                                line_dash="dash", 
                                                line_color="green",
                                                annotation_text=f"Meaningful Decrease: ≤{(1.0/meaningful_threshold):.1f}x"
                                            )
                                            fig_meaningful.add_hline(y=1.0, line_color="gray", line_width=1, annotation_text="No Change (1.0x)")
                                            
                                            fig_meaningful.update_layout(
                                                title=f"Individual Relative Change Patterns: {selected_protein_ind}<br>"
                                                      f"{protein_data['n_meaningful_increased']} meaningful increases (≥{meaningful_threshold:.1f}x), "
                                                      f"{protein_data['n_meaningful_decreased']} meaningful decreases (≤{(1.0/meaningful_threshold):.1f}x)<br>"
                                                      f"Binomial p-value: {protein_data['individual_p_value']:.4f}",
                                                xaxis_title="Participant",
                                                yaxis_title=f"Fold Change (Target/Baseline)",
                                                height=500,
                                                showlegend=False
                                            )
                                            
                                            st.plotly_chart(fig_meaningful, use_container_width=True)
                                            
                                            # Additional fold change statistics
                                            st.subheader("📋 Fold Change Statistics")
                                            
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Mean Fold Change", f"{protein_data.get('mean_fold_change', 1.0):.2f}x")
                                            with col2:
                                                st.metric("Median Fold Change", f"{protein_data.get('median_fold_change', 1.0):.2f}x")
                                            with col3:
                                                increase_percent = (protein_data['n_meaningful_increased'] / protein_data['n_participants'] * 100)
                                                st.metric("% Meaningful Increases", f"{increase_percent:.1f}%")
                                            with col4:
                                                decrease_percent = (protein_data['n_meaningful_decreased'] / protein_data['n_participants'] * 100)
                                                st.metric("% Meaningful Decreases", f"{decrease_percent:.1f}%")
                                
                                else:
                                    st.warning("No proteins showed significant individual meaningful relative changes.")
                            
                            # ACROSS-INDIVIDUAL (GROUP LEVEL) ANALYSIS TAB
                            with across_tab:
                                st.info("🌐 **Across-Individual Analysis**: Group/population level changes and statistical significance")
                                
                                # Group-level significance from within-individual analysis (paired t-test)
                                group_significant_paired = {k: v for k, v in within_results.items() if v.get('group_significant', False)}
                                
                                # Independent samples significance  
                                across_significant_independent = {k: v for k, v in across_results.items() if v.get('significant', False)}
                                
                                # Show both group-level analyses
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.subheader("📊 Group Analysis (Independent)")
                                    st.info("**Independent t-test**: Different participants at different times")
                                    
                                    if across_significant_independent:
                                        st.success(f"{len(across_significant_independent)} proteins with significant group changes (independent)")
                                        
                                        # Streamlined summary
                                        independent_summary = []
                                        for protein, data in list(across_significant_independent.items()): #[:10] all
                                            independent_summary.append({
                                                'Protein': protein,
                                                'N_Baseline': data['n_baseline'],
                                                'N_Target': data['n_target'],
                                                'Mean_Change': f"{data['mean_change']:.3f}",
                                                'Cohen_d': f"{data['cohen_d']:.3f}",
                                                'P_value': f"{data['p_value']:.4f}",
                                                'Effect_Size': data['effect_size']
                                            })
                                        
                                        independent_df = pd.DataFrame(independent_summary)
                                        st.dataframe(independent_df, use_container_width=True)
                                    else:
                                        st.warning("No significant independent group changes.")
                                
                                with col2:
                                    st.subheader("📊 Group Analysis (Paired)")
                                    st.info("**Paired t-test**: Average change across same participants ≠ 0?")
                                    
                                    if group_significant_paired:
                                        st.success(f"{len(group_significant_paired)} proteins with significant group changes (paired)")
                                        
                                        # Streamlined summary
                                        paired_summary = []
                                        for protein, data in list(group_significant_paired.items()): #[:10] not 10 changed to all
                                            paired_summary.append({
                                                'Protein': protein,
                                                'N_Pairs': data['n_participants'],
                                                'Mean_Change': f"{data['mean_difference']:.3f}",
                                                'Cohen_d': f"{data['cohen_d']:.3f}",
                                                'P_value': f"{data['group_p_value']:.4f}",
                                                'Effect_Size': data['effect_size']
                                            })
                                        
                                        paired_df = pd.DataFrame(paired_summary)
                                        st.dataframe(paired_df, use_container_width=True)
                                    else:
                                        st.warning("No significant paired group changes.")
                               
                                # Combined group-level visualization
                                st.subheader("📈 Group-Level Change Patterns")
                                
                                # Choose which analysis to visualize
                                analysis_type = st.radio(
                                    "Select group analysis to visualize:",
                                    ["Paired Group Analysis", "Independent Group Analysis"],
                                    key=f"group_analysis_type_{target_tp}"
                                )
                                
                                if analysis_type == "Independent Group Analysis" and across_significant_independent:
                                    selected_protein_indep = st.selectbox(
                                        "Select protein for independent group visualization:",
                                        list(across_significant_independent.keys()), #changed to all, [:10]
                                        key=f"independent_group_protein_select_{target_tp}"
                                    )
                                    
                                    if selected_protein_indep:
                                        protein_data_indep = across_significant_independent[selected_protein_indep]
                                        
                                        # Get raw data for independent analysis
                                        baseline_data_viz = baseline_data[selected_protein_indep].dropna()
                                        target_data_viz = target_data[selected_protein_indep].dropna()
                                        
                                        # Box plot comparison
                                        fig_independent = go.Figure()
                                        
                                        fig_independent.add_trace(go.Box(
                                            y=baseline_data_viz,
                                            name=f'Baseline (T{baseline_tp})',
                                            marker_color='lightblue'
                                        ))
                                        
                                        fig_independent.add_trace(go.Box(
                                            y=target_data_viz,
                                            name=f'Target (T{target_tp})',
                                            marker_color='lightcoral'
                                        ))
                                        
                                        fig_independent.update_layout(
                                            title=f"Independent Group Comparison: {selected_protein_indep}<br>"
                                                  f"Mean Change: {protein_data_indep['mean_change']:.3f}, "
                                                  f"P-value: {protein_data_indep['p_value']:.4f}",
                                            yaxis_title="Protein Level",
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_independent, use_container_width=True)
                                
                                elif analysis_type == "Paired Group Analysis" and group_significant_paired:
                                    selected_protein_paired = st.selectbox(
                                        "Select protein for paired group visualization:",
                                        list(group_significant_paired.keys())[:10],
                                        key=f"paired_group_protein_select_{target_tp}"
                                    )
                                    
                                    if selected_protein_paired:
                                        protein_data_paired = group_significant_paired[selected_protein_paired]
                                        
                                        # Get individual differences for paired analysis
                                        individual_differences = protein_data_paired['individual_differences']
                                        
                                        # Create histogram of individual differences
                                        fig_paired = go.Figure()
                                        
                                        fig_paired.add_trace(go.Histogram(
                                            x=individual_differences,
                                            nbinsx=20,
                                            name='Individual Differences',
                                            marker_color='lightgreen'
                                        ))
                                        
                                        # Add mean line
                                        mean_diff = protein_data_paired['mean_difference']
                                        fig_paired.add_vline(
                                            x=mean_diff,
                                            line_dash="dash",
                                            line_color="red",
                                            annotation_text=f"Mean: {mean_diff:.3f}"
                                        )
                                        
                                        fig_paired.update_layout(
                                            title=f"Paired Group Analysis: {selected_protein_paired}<br>"
                                                  f"Mean Change: {protein_data_paired['mean_difference']:.3f}, "
                                                  f"P-value: {protein_data_paired['group_p_value']:.4f}",
                                            xaxis_title="Individual Difference (Target - Baseline)",
                                            yaxis_title="Frequency",
                                            height=400
                                        )
                                        
                                        st.plotly_chart(fig_paired, use_container_width=True)
                            
                               # Add download button for summary table
                         
                            # INTERPRETATION GUIDE TAB - Corrected conceptual organization
                            with interpretation_tab:
                                st.subheader("📋 Corrected Statistical Interpretation Guide")
                                
                                st.markdown("""
                                ### 👤 **Within-Individual Analysis**
                                Focuses on **individual change patterns** and **meaningful relative changes**.
                                
                                **Binomial Test for Meaningful Relative Changes:**
                                - H₀: Proportion of meaningful individual relative changes = 50% (chance)
                                - H₁: Proportion of meaningful individual relative changes ≠ 50%
                                - **Meaningful Changes**: Fold changes ≥ threshold (e.g., ≥2.0x = 100% increase) OR ≤ 1/threshold (e.g., ≤0.5x = 50% decrease)
                                - **Purpose**: Identify biomarkers where many individuals show clinically meaningful relative responses
                                - **Advantage**: Accounts for different baseline levels across participants
                                
                                ### 🌐 **Across-Individual (Group Level) Analysis**
                                Focuses on **population/group level changes** and **statistical significance**.
                                
                                #### **Paired Group Analysis:**
                                - **Paired t-test**: Tests if average change across participants ≠ 0
                                - **Same participants** at different timepoints
                                - **Stronger causal inference** but smaller sample sizes
                                
                                #### **Independent Group Analysis:**
                                - **Independent t-test**: Compares group means at different timepoints  
                                - **Different participants** allowed at each timepoint
                                - **Larger sample sizes** possible but weaker causal inference
                                
                                ### 🎯 **Clinical Interpretation Hierarchy**
                                
                                1. **🌟 Individual + Group Significant (Highest Priority)**:
                                   - Many individuals show meaningful **relative changes** AND group effect detected
                                   - Best candidates for clinical biomarkers
                                
                                2. **👤 Individual Significant Only**:
                                   - High proportion of meaningful **relative changes**
                                   - Excellent for personalized medicine applications
                                
                                3. **🌐 Group Significant Only**:
                                   - Population-level effect but few meaningful individual relative changes
                                   - Useful for epidemiological studies
                                
                                4. **❌ Neither Significant**:
                                   - No evidence for meaningful relative changes or group-level changes
                                   - Lower priority for clinical development
                                
                                ### ✅ **Why Relative Changes (Fold Changes) Are Better**
                                
                                - **Biological Relevance**: 2x increase meaningful regardless of baseline (10→20 vs 100→200)
                                - **Standardized Interpretation**: Fold changes comparable across different proteins
                                - **Clinical Applicability**: Relative changes more clinically interpretable
                                - **Baseline Independent**: Accounts for natural variation in baseline levels
                                
                                ### 📊 **Example Interpretation**
                                
                                - **2.0x threshold**: 100% increase considered meaningful
                                - **0.5x threshold**: 50% decrease considered meaningful  
                                - **Participant A**: Baseline=10, Target=25 → 2.5x (meaningful increase)
                                - **Participant B**: Baseline=100, Target=180 → 1.8x (not meaningful)
                                
                                ### ⚡ **Performance Optimization**
                                
                                - **Filtered Analysis**: Only analyzes significant proteins from previous step
                                - **Streamlined Calculations**: Reduced redundant computations
                                - **Efficient Data Matching**: Optimized participant matching for paired analysis
                                """)
                            
                            st.divider()
                
                else:
                    st.warning("Please select at least one target timepoint for comparison.")
            
            else:
                st.info("👆 Select timepoints and click 'Run Enhanced Corrected Analysis' to begin")
        
        # Tab 4: Enhanced Persistent Change - Long Term Effect with Within/Across Analysis
        with tab4:
            st.markdown('<div class="tab-header">📈 Biomarker Analysis - Persistent Change (Long Term Effect)</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="persistence-box">', unsafe_allow_html=True)
            st.markdown("**Focus**: Persistence analysis of significant biomarkers")
            st.markdown("**Analysis**: CV < 20% + stability criteria from intermediate to final")
            st.markdown("**Features**: Enhanced prediction and persistence scoring with within/across individual analysis")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Enhanced options for persistence analysis
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                baseline_tp_persist = st.selectbox(
                    "Baseline Timepoint:",
                    st.session_state.analyzer.timepoints,
                    index=0,
                    key="persist_baseline"
                )
            
            with col2:
                intermediate_tp_persist = st.selectbox(
                    "Intermediate Timepoint:",
                    [tp for tp in st.session_state.analyzer.timepoints if tp != baseline_tp_persist],
                    index=0 if len(st.session_state.analyzer.timepoints) > 1 else 0,
                    key="persist_intermediate"
                )
            
            with col3:
                final_tps_persist = st.multiselect(
                    "Final Timepoints:",
                    [tp for tp in st.session_state.analyzer.timepoints if tp not in [baseline_tp_persist, intermediate_tp_persist]],
                    default=[tp for tp in st.session_state.analyzer.timepoints if tp not in [baseline_tp_persist, intermediate_tp_persist]][:2],
                    key="persist_finals"
                )
            
            with col4:
                # Option to analyze only significant proteins
                analyze_only_significant = st.checkbox(
                    "Analyze Only Significant Proteins",
                    value=True,
                    help="If checked, analyzes only proteins that showed significant changes in previous analysis. If unchecked, analyzes all proteins.",
                    key="analyze_only_significant"
                )
            
            # Instructions based on selection
            if analyze_only_significant:
                st.info("🎯 **Recommended**: Analyzing only significant proteins ensures persistence analysis focuses on meaningful changes identified in the previous step.")
            else:
                st.warning("⚠️ **Note**: Analyzing all proteins may include biomarkers without significant changes, potentially inflating persistence counts.")
            
            if st.button("📈 Run Enhanced Long Term Effects", type="primary", key="run_persist_analysis"):
                if final_tps_persist:
                    with st.spinner("Step 1/3: Identifying proteins with meaningful changes (baseline→intermediate)..."):
                        # STEP 1: Run baseline→intermediate analysis to identify meaningful & significant proteins
                        baseline_to_intermediate_results = st.session_state.analyzer.analyze_significant_changes_enhanced(
                            baseline_tp_persist, [intermediate_tp_persist], 2.0
                        )
                        
                        # Extract proteins that meet BOTH criteria: fold change ≥2 AND p<0.05
                        if intermediate_tp_persist in baseline_to_intermediate_results:
                            baseline_inter_data = baseline_to_intermediate_results[intermediate_tp_persist]
                            baseline_inter_within = baseline_inter_data.get('within', {})
                            
                            # Identify proteins with BOTH meaningful individual changes AND statistical significance
                            meaningful_and_significant_proteins = []
                            
                            for protein, data in baseline_inter_within.items():
                                # Check for meaningful individual changes (fold change ≥2)
                                has_meaningful_individual = data.get('individual_significant', False)
                                # Check for statistical significance (group p-value < 0.05) 
                                has_statistical_significance = data.get('group_p_value', 1.0) < 0.05
                                
                                # Include only proteins that meet BOTH criteria
                                if has_meaningful_individual and has_statistical_significance:
                                    meaningful_and_significant_proteins.append(protein)
                            
                            if len(meaningful_and_significant_proteins) == 0:
                                st.error("❌ No proteins found with BOTH meaningful fold changes (≥2) AND statistical significance (p<0.05) from baseline→intermediate.")
                                st.stop()
                            else:
                                st.success(f"✅ Found {len(meaningful_and_significant_proteins)} proteins with BOTH meaningful changes (≥2 fold) AND significance (p<0.05)")
                                
                                # Show breakdown of criteria
                                total_meaningful = sum(1 for v in baseline_inter_within.values() if v.get('individual_significant', False))
                                total_significant = sum(1 for v in baseline_inter_within.values() if v.get('group_p_value', 1.0) < 0.05)
                                
                                st.info(f"""
                                **Selection Criteria for Persistence Analysis:**
                                - Proteins with meaningful individual changes (≥2 fold): {total_meaningful}
                                - Proteins with statistical significance (p<0.05): {total_significant}
                                - **Proteins meeting BOTH criteria**: {len(meaningful_and_significant_proteins)}
                                """)
                        else:
                            st.error("❌ Failed to analyze baseline→intermediate changes. Please check your data.")
                            st.stop()
                    
                    with st.spinner("Step 2/3: Running persistence analysis on qualified proteins..."):
                        # STEP 2: Run persistence analysis ONLY on proteins that met both criteria
                        persistence_results, message = st.session_state.analyzer.analyze_persistence_enhanced_v2(
                            baseline_tp_persist, intermediate_tp_persist, final_tps_persist, meaningful_and_significant_proteins
                        )
                        # Run within/across analysis for persistence
                        within_across_results = st.session_state.analyzer.analyze_within_across_for_persistence(
                            baseline_tp_persist, intermediate_tp_persist, final_tps_persist, meaningful_and_significant_proteins
                        )
                    
                    if persistence_results:
                        st.success(message)
                        
                        # Show analysis scope information
                        total_proteins = len(st.session_state.analyzer.protein_cols)
                        analyzed_proteins = len(persistence_results)
                        
                        if analyze_only_significant:
                            significant_count = len(meaningful_and_significant_proteins) if meaningful_and_significant_proteins else 0
                            st.info(f"📊 **Analysis Scope**: {analyzed_proteins} persistent proteins found from {significant_count} significant proteins (out of {total_proteins} total biomarkers)")
                        else:
                            st.info(f"📊 **Analysis Scope**: {analyzed_proteins} persistent proteins found from {total_proteins} total biomarkers")
                        
                        # SUMMARY PLOT - Number of Persistent Features
                        st.subheader("📊 Persistence Summary - Within vs Across Individuals")
                        
                        # Calculate persistence counts for summary
                        if within_across_results:
                            within_persistent_counts = {}
                            across_persistent_counts = {}
                            
                            for final_tp in final_tps_persist:
                                if final_tp in within_across_results:
                                    # Count within-individual persistent features
                                    within_count = 0
                                    across_count = 0
                                    
                                    for protein, data in within_across_results[final_tp].items():
                                        # Within-individual persistence criteria
                                        if data.get('within'):
                                            within_data = data['within']
                                            # Consider persistent if consistency score > 0.6 and direction persistence > 0.5
                                            if (within_data.get('consistency_score', 0) > 0.6 and 
                                                within_data.get('direction_persistence', 0) > 0.5):
                                                within_count += 1
                                        
                                        # Across-individual persistence criteria  
                                        if data.get('across'):
                                            across_data = data['across']
                                            # Consider persistent if CV stable and direction persistent
                                            if (across_data.get('cv_stable', False) and 
                                                across_data.get('direction_persistence', 0) > 0.5):
                                                across_count += 1
                                    
                                    within_persistent_counts[f'T{final_tp}'] = within_count
                                    across_persistent_counts[f'T{final_tp}'] = across_count
                            
                            # Create summary bar plot
                            if within_persistent_counts or across_persistent_counts:
                                timepoints = list(within_persistent_counts.keys())
                                within_counts = list(within_persistent_counts.values())
                                across_counts = list(across_persistent_counts.values())
                                
                                fig_summary = go.Figure()
                                
                                # Add within-individual bars
                                fig_summary.add_trace(go.Bar(
                                    name='Within-Individual Persistent',
                                    x=timepoints,
                                    y=within_counts,
                                    marker_color='lightblue',
                                    text=within_counts,
                                    textposition='auto'
                                ))
                                
                                # Add across-individual bars
                                fig_summary.add_trace(go.Bar(
                                    name='Across-Individual Persistent',
                                    x=timepoints,
                                    y=across_counts,
                                    marker_color='lightcoral',
                                    text=across_counts,
                                    textposition='auto'
                                ))
                                
                                fig_summary.update_layout(
                                    title=f'Number of Persistent Biomarkers by Final Timepoint<br>'
                                          f'Baseline: T{baseline_tp_persist}, Intermediate: T{intermediate_tp_persist}',
                                    xaxis_title='Final Timepoint',
                                    yaxis_title='Number of Persistent Biomarkers',
                                    barmode='group',
                                    height=400,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_summary, use_container_width=True)
                                
                                # Add summary statistics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    total_within = sum(within_counts) if within_counts else 0
                                    st.metric("Total Within-Individual Persistent", total_within)
                                
                                with col2:
                                    total_across = sum(across_counts) if across_counts else 0
                                    st.metric("Total Across-Individual Persistent", total_across)
                                
                                with col3:
                                    avg_within = np.mean(within_counts) if within_counts else 0
                                    st.metric("Average Within per Timepoint", f"{avg_within:.1f}")
                                
                                with col4:
                                    avg_across = np.mean(across_counts) if across_counts else 0
                                    st.metric("Average Across per Timepoint", f"{avg_across:.1f}")
                                
                                # Interpretation note
                                st.info("""
                                **Persistence Criteria:**
                                - **Within-Individual**: Consistency score > 0.6 AND Direction persistence > 0.5
                                - **Across-Individual**: CV < 20% (stable) AND Direction persistence > 0.5
                                - **Higher counts** indicate more biomarkers showing stable, persistent patterns
                                """)
                        
                        st.divider()
                        
                        # Create tabs for different analysis views
                        persistence_overview_tab, within_persist_tab, across_persist_tab, radar_plot_tab = st.tabs([
                            "📊 Overview & Summary",
                            "👥 Within-Individual Persistence", 
                            "🌐 Across-Individual Persistence",
                            "🎯 Radar Plot Analysis"
                        ])
                        
                        # OVERVIEW TAB with Enhanced Prediction Analysis and CORRECTED Persistence
                        with persistence_overview_tab:
                            st.subheader("📊 CORRECTED Persistence Analysis Results Overview")
                            
                            # CORRECTED METHODOLOGY EXPLANATION
                            st.info("""
                            **CORRECTED 4-Step Persistence Analysis:**
                            1. **Meaningful Changes**: Within-individual relative changes (≥2x) + Across-individual t-test
                            2. **Stability Analysis**: CV < 20% + Non-significant t-test (intermediate→final)  
                            3. **Prediction Analysis**: Early changes (baseline→intermediate) predict all final timepoints
                            4. **Overall Assessment**: When do changes stop being stable?
                            """)
                            
                            # PREDICTION ANALYSIS SECTION
                            st.subheader("🔮 Prediction Analysis: Can Early Changes Predict Late Changes?")
                            st.info("""
                            **Enhanced Prediction Question**: Can changes from baseline→intermediate predict changes from baseline→ALL final timepoints?
                            - **Early Changes**: Baseline → Intermediate (e.g., T1 → T3)  
                            - **Late Changes**: Baseline → Each Final (e.g., T1 → T4, T1 → T5, T1 → T6)
                            - **Stability Timeline**: When do changes stop being predictable/stable?
                            """)
                            
                            # Calculate enhanced prediction statistics across all proteins - FIXED
                            prediction_stats = []
                            stability_timeline = {}
                            high_prediction_proteins = []
                            moderate_prediction_proteins = []
                            low_prediction_proteins = []
                            
                            for protein, data in persistence_results.items():
                                if 'prediction_by_timepoint' in data:
                                    # Track when stability breaks down - FIXED
                                    breakdown_tp = data.get('stability_breakdown_timepoint', 'Never')
                                    stable_until = data.get('stable_until', 'All timepoints')
                                    
                                    # Convert to string for consistent handling
                                    breakdown_key = str(breakdown_tp) if breakdown_tp is not None else 'Never'
                                    
                                    if breakdown_key not in stability_timeline:
                                        stability_timeline[breakdown_key] = 0
                                    stability_timeline[breakdown_key] += 1
                                    
                                    # Get prediction data for each final timepoint
                                    for final_tp, pred_data in data['prediction_by_timepoint'].items():
                                        pred_accuracy = pred_data.get('prediction_accuracy', 0)
                                        
                                        # Ensure prediction accuracy is numeric
                                        if isinstance(pred_accuracy, str):
                                            try:
                                                pred_accuracy = float(pred_accuracy)
                                            except:
                                                pred_accuracy = 0.0
                                        
                                        prediction_stats.append({
                                            'Protein': protein,
                                            'Final_Timepoint': f'T{final_tp}',
                                            'Prediction_Accuracy': pred_accuracy,
                                            'Prediction_Quality': pred_data.get('prediction_quality', 'Low'),
                                            'Direction_Persistence': pred_data.get('direction_persistence', 0),
                                            'Stable_Until': f'T{stable_until}',
                                            'Breakdown_At': f'T{breakdown_tp}' if breakdown_tp != 'Never' and breakdown_tp is not None else 'Never'
                                        })
                                        
                                        # Categorize proteins by prediction quality
                                        protein_label = f"{protein} → T{final_tp}"
                                        if pred_accuracy > 0.7:
                                            high_prediction_proteins.append(f"{protein_label} (r={pred_accuracy:.3f})")
                                        elif pred_accuracy > 0.5:
                                            moderate_prediction_proteins.append(f"{protein_label} (r={pred_accuracy:.3f})")
                                        else:
                                            low_prediction_proteins.append(f"{protein_label} (r={pred_accuracy:.3f})")
                            
                            if prediction_stats:
                                # Enhanced prediction summary metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("High Prediction (r>0.7)", len(high_prediction_proteins))
                                with col2:
                                    st.metric("Moderate Prediction (0.5<r≤0.7)", len(moderate_prediction_proteins))
                                with col3:
                                    st.metric("Low Prediction (r≤0.5)", len(low_prediction_proteins))
                                with col4:
                                    # Fixed average prediction calculation
                                    if prediction_stats:
                                        numeric_predictions = []
                                        for p in prediction_stats:
                                            pred_acc = p['Prediction_Accuracy']
                                            if isinstance(pred_acc, str):
                                                try:
                                                    numeric_predictions.append(float(pred_acc))
                                                except:
                                                    numeric_predictions.append(0.0)
                                            else:
                                                numeric_predictions.append(float(pred_acc) if pred_acc is not None else 0.0)
                                        
                                        avg_prediction = np.mean(numeric_predictions) if numeric_predictions else 0.0
                                        st.metric("Average Prediction Accuracy", f"{avg_prediction:.3f}")
                                    else:
                                        st.metric("Average Prediction Accuracy", "0.000")
                                
                                # Stability timeline analysis
                                st.subheader("📅 Stability Timeline: When Do Changes Stop Being Stable?")
                                
                                if stability_timeline:
                                    timeline_df = pd.DataFrame([
                                        {'Timepoint': k, 'N_Proteins_Breakdown': v} 
                                        for k, v in stability_timeline.items()
                                    ])
                                    
                                    fig_timeline = px.bar(
                                        timeline_df,
                                        x='Timepoint',
                                        y='N_Proteins_Breakdown',
                                        title='When Do Biomarkers Stop Being Stable?',
                                        labels={'N_Proteins_Breakdown': 'Number of Biomarkers', 'Timepoint': 'Stability Breaks Down At'}
                                    )
                                    st.plotly_chart(fig_timeline, use_container_width=True)
                                    
                                    # Interpretation
                                    never_breakdown = stability_timeline.get('Never', 0)
                                    st.info(f"""
                                    **Stability Timeline Interpretation:**
                                    - **{never_breakdown} biomarkers** remain stable across all timepoints
                                    - **Breakdown pattern** shows when stability deteriorates
                                    - **Earlier breakdown** = shorter persistence window
                                    """)
                                
                                # Prediction correlation distribution plot
                                pred_df = pd.DataFrame(prediction_stats)
                                fig_pred_dist = px.histogram(
                                    pred_df, 
                                    x='Prediction_Accuracy',
                                    color='Prediction_Quality',
                                    title='Distribution of Prediction Accuracies<br>(Baseline→Intermediate predicting Baseline→All Finals)',
                                    labels={'Prediction_Accuracy': 'Prediction Accuracy (r)'},
                                    nbins=20
                                )
                                fig_pred_dist.add_vline(x=0.7, line_dash="dash", line_color="green", 
                                                       annotation_text="High Prediction (r=0.7)")
                                fig_pred_dist.add_vline(x=0.5, line_dash="dash", line_color="orange", 
                                                       annotation_text="Moderate Prediction (r=0.5)")
                                st.plotly_chart(fig_pred_dist, use_container_width=True)
                                
                                # Show top predictive proteins with timeline
                                if high_prediction_proteins:
                                    st.success("🎯 **High Prediction Biomarkers** (Early changes strongly predict late changes):")
                                    for protein in high_prediction_proteins[:10]:
                                        st.write(f"• {protein}")
                                
                                # Detailed prediction and stability table
                                st.subheader("📋 Detailed Prediction & Stability Analysis")
                                pred_table = pred_df.sort_values('Prediction_Accuracy', ascending=False)
                                st.dataframe(pred_table.head(15), use_container_width=True)
                            
                            st.divider()
                            
                            # Create summary table with CORRECTED methodology
                            persistence_summary = []
                            for protein, data in list(persistence_results.items()): # all proteins not first [:20]
                                # Safely extract values with proper type checking
                                n_meaningful = data.get('n_meaningful_participants', 0)
                                n_total = data.get('meaningful_changes', {}).get('n_total_participants', 1)
                                
                                # Ensure numeric values
                                if isinstance(n_meaningful, str):
                                    try:
                                        n_meaningful = int(n_meaningful)
                                    except:
                                        n_meaningful = 0
                                
                                if isinstance(n_total, str):
                                    try:
                                        n_total = int(n_total)
                                    except:
                                        n_total = 1
                                
                                # Calculate proportion safely
                                meaningful_proportion = n_meaningful / max(n_total, 1)
                                
                                persistence_summary.append({
                                    'Protein': protein,
                                    'Category': data.get('category', 'Unknown'),
                                    'Overall_Score': data.get('overall_persistence_score', 0),
                                    'Stability_Score': data.get('avg_stability_score', 0),
                                    'Prediction_Score': data.get('avg_prediction_accuracy', 0),
                                    'Meaningful_Participants_Count': n_meaningful,
                                    'Total_Participants': n_total,
                                    'Meaningful_Proportion': meaningful_proportion,
                                    'Meaningful_Participants_Display': f"{n_meaningful}/{n_total}",
                                    'Stable_Until': f"T{data.get('stable_until', 'Unknown')}",
                                    'Breakdown_At': f"T{data['stability_breakdown_timepoint']}" if data.get('stability_breakdown_timepoint') else "Never"
                                })
                            
                            persistence_df = pd.DataFrame(persistence_summary)
                            
                            # Format display columns
                            display_df = persistence_df.copy()
                            display_df['Overall_Score'] = display_df['Overall_Score'].apply(lambda x: f"{x:.3f}")
                            display_df['Stability_Score'] = display_df['Stability_Score'].apply(lambda x: f"{x:.3f}")
                            display_df['Prediction_Score'] = display_df['Prediction_Score'].apply(lambda x: f"{x:.3f}")
                            display_df['Meaningful_Proportion'] = display_df['Meaningful_Proportion'].apply(lambda x: f"{x:.3f}")
                            
                            # Select columns for display
                            display_columns = ['Protein', 'Category', 'Overall_Score', 'Stability_Score', 
                                             'Prediction_Score', 'Meaningful_Participants_Display', 'Stable_Until', 'Breakdown_At']
                            
                            st.dataframe(display_df[display_columns].rename(columns={
                                'Meaningful_Participants_Display': 'Meaningful_Participants'
                            }), use_container_width=True)
                            
                            # Add download button for summary table
                            st.subheader("📥 Download Summary Table")
                            
                            # Prepare download data with all columns
                            download_df = persistence_df.copy()
                            download_df['Overall_Score'] = download_df['Overall_Score'].apply(lambda x: f"{x:.6f}")
                            download_df['Stability_Score'] = download_df['Stability_Score'].apply(lambda x: f"{x:.6f}")
                            download_df['Prediction_Score'] = download_df['Prediction_Score'].apply(lambda x: f"{x:.6f}")
                            download_df['Meaningful_Proportion'] = download_df['Meaningful_Proportion'].apply(lambda x: f"{x:.6f}")
                            
                            # Create CSV data
                            csv_data = download_df.to_csv(index=False)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="📊 Download Complete Summary (CSV)",
                                    data=csv_data,
                                    file_name=f"persistence_analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download complete persistence analysis results with all metrics"
                                )
                            
                            with col2:
                                # Also provide a simplified version
                                simplified_df = display_df[['Protein', 'Category', 'Overall_Score', 
                                                          'Meaningful_Participants_Display', 'Stable_Until']].copy()
                                simplified_csv = simplified_df.to_csv(index=False)
                                
                                st.download_button(
                                    label="📋 Download Simplified Summary (CSV)",
                                    data=simplified_csv,
                                    file_name=f"persistence_summary_simplified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    help="Download simplified summary with key metrics only"
                                )
                            
                            # Category distribution with enhanced interpretation
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                category_counts = persistence_df['Category'].value_counts()
                                fig_category = px.pie(
                                    values=category_counts.values,
                                    names=category_counts.index,
                                    title="Distribution of Persistence Categories<br>(CORRECTED Methodology)"
                                )
                                st.plotly_chart(fig_category, use_container_width=True)
                            
                            with col2:
                                # Enhanced scatter plot with stability timeline - FIXED
                                fig_scores = px.scatter(
                                    persistence_df,
                                    x='Stability_Score',
                                    y='Prediction_Score',
                                    color='Category',
                                    size='Meaningful_Proportion',  # Use numeric proportion instead of string
                                    size_max=20,
                                    hover_data=['Protein', 'Stable_Until', 'Breakdown_At', 'Meaningful_Participants_Display'],
                                    title='Stability vs PREDICTION Scores<br>(CORRECTED: With Stability Timeline)',
                                    labels={
                                        'Stability_Score': 'Stability Score',
                                        'Prediction_Score': 'Prediction Score',
                                        'Meaningful_Proportion': 'Proportion Meaningful'
                                    }
                                )
                                fig_scores.add_hline(y=0.7, line_dash="dash", line_color="green", 
                                                    annotation_text="High Prediction")
                                fig_scores.add_vline(x=0.7, line_dash="dash", line_color="blue", 
                                                    annotation_text="High Stability")
                                fig_scores.update_layout(height=400)
                                st.plotly_chart(fig_scores, use_container_width=True)
                        
                        # WITHIN-INDIVIDUAL PERSISTENCE TAB
                        with within_persist_tab:
                            st.subheader("👥 Within-Individual Persistence Analysis")
                            
                            if within_across_results:
                                # Select final timepoint for analysis
                                selected_final_tp = st.selectbox(
                                    "Select final timepoint for within-individual analysis:",
                                    final_tps_persist,
                                    key="within_final_tp_select"
                                )
                                
                                if selected_final_tp in within_across_results:
                                    within_data = within_across_results[selected_final_tp]
                                    
                                    # Filter proteins with within-individual data
                                    within_proteins = {k: v['within'] for k, v in within_data.items() if v.get('within')}
                                    
                                    if within_proteins:
                                        st.success(f"Found {len(within_proteins)} proteins with within-individual persistence data!")
                                        
                                        # Create within-individual summary table
                                        within_summary = []
                                        for protein, data in list(within_proteins.items()): # keep all [:15]
                                            within_summary.append({
                                                'Protein': protein,
                                                'N_Participants': data['n_participants'],
                                                'N_Meaningful': data['n_meaningful'],
                                                'Direction_Persistence': f"{data['direction_persistence']:.3f}",
                                                'Magnitude_Correlation': f"{data['magnitude_correlation']:.3f}",
                                                'Individual_Stability': f"{data['individual_stability']:.3f}",
                                                'Consistency_Score': f"{data['consistency_score']:.3f}",
                                                'Prediction_Accuracy': f"{data['prediction_accuracy']:.3f}"
                                            })
                                        
                                        within_df = pd.DataFrame(within_summary)
                                        st.dataframe(within_df, use_container_width=True)
                                        
                                        st.info("""
                                        **Within-Individual Metrics Explanation:**
                                        - **Direction_Persistence**: Proportion maintaining same change direction
                                        - **Magnitude_Correlation**: Correlation between early and late changes
                                        - **Individual_Stability**: Proportion with small changes after intermediate
                                        - **Consistency_Score**: Overall within-individual consistency
                                        - **Prediction_Accuracy**: How well early changes predict late changes
                                        """)
                                        
                                        # Individual trajectory visualization
                                        st.subheader("📊 Individual Trajectory Analysis")
                                        
                                        selected_protein_within = st.selectbox(
                                            "Select protein for trajectory visualization:",
                                            list(within_proteins.keys()), #keep all [:10]
                                            key="within_protein_trajectory"
                                        )
                                        
                                        if selected_protein_within:
                                            within_protein_data = within_proteins[selected_protein_within]
                                            
                                            # Create trajectory plot
                                            fig_trajectory = go.Figure()
                                            
                                            # Get change data
                                            early_changes = within_protein_data['early_changes']
                                            late_changes = within_protein_data['late_changes']
                                            
                                            # Add trajectory lines for each participant
                                            for i, (early, late) in enumerate(zip(early_changes, late_changes)):
                                                fig_trajectory.add_trace(go.Scatter(
                                                    x=[baseline_tp_persist, intermediate_tp_persist, selected_final_tp],
                                                    y=[0, early, late],  # Start at 0 (baseline), then early change, then late change
                                                    mode='lines+markers',
                                                    name=f'Participant {i+1}',
                                                    showlegend=False,
                                                    line=dict(width=1, color='rgba(100,100,100,0.5)'),
                                                    marker=dict(size=4)
                                                ))
                                            
                                            # Add mean trajectory
                                            mean_early = np.mean(early_changes)
                                            mean_late = np.mean(late_changes)
                                            fig_trajectory.add_trace(go.Scatter(
                                                x=[baseline_tp_persist, intermediate_tp_persist, selected_final_tp],
                                                y=[0, mean_early, mean_late],
                                                mode='lines+markers',
                                                name='Mean Trajectory',
                                                line=dict(width=4, color='red'),
                                                marker=dict(size=8, color='red')
                                            ))
                                            
                                            fig_trajectory.update_layout(
                                                title=f"Individual Trajectories: {selected_protein_within}<br>"
                                                      f"Direction Persistence: {within_protein_data['direction_persistence']:.3f}, "
                                                      f"Prediction Accuracy: {within_protein_data['prediction_accuracy']:.3f}",
                                                xaxis_title="Timepoint",
                                                yaxis_title="Change from Baseline",
                                                height=500
                                            )
                                            
                                            st.plotly_chart(fig_trajectory, use_container_width=True)
                                    else:
                                        st.warning("No within-individual persistence data available for selected timepoint.")
                                else:
                                    st.warning("Selected timepoint not available in within-individual analysis.")
                            else:
                                st.warning("No within-individual persistence analysis data available.")
                        
                        # ACROSS-INDIVIDUAL PERSISTENCE TAB
                        with across_persist_tab:
                            st.subheader("🌐 Across-Individual Persistence Analysis")
                            
                            if within_across_results:
                                # Select final timepoint for analysis
                                selected_final_tp_across = st.selectbox(
                                    "Select final timepoint for across-individual analysis:",
                                    final_tps_persist,
                                    key="across_final_tp_select"
                                )
                                
                                if selected_final_tp_across in within_across_results:
                                    across_data = within_across_results[selected_final_tp_across]
                                    
                                    # Filter proteins with across-individual data
                                    across_proteins = {k: v['across'] for k, v in across_data.items() if v.get('across')}
                                    
                                    if across_proteins:
                                        st.success(f"Found {len(across_proteins)} proteins with across-individual persistence data!")
                                        
                                        # Create across-individual summary table
                                        across_summary = []
                                        for protein, data in list(across_proteins.items()):
                                            across_summary.append({
                                                'Protein': protein,
                                                'Baseline_Mean': f"{data['baseline_mean']:.3f}",
                                                'Intermediate_Mean': f"{data['intermediate_mean']:.3f}",
                                                'Final_Mean': f"{data['final_mean']:.3f}",
                                                'Early_Change': f"{data['early_change']:.3f}",
                                                'Late_Change': f"{data['late_change']:.3f}",
                                                'Direction_Persistence': f"{data['direction_persistence']:.3f}",
                                                'Final_CV': f"{data['final_cv']:.1f}%",
                                                'CV_Stable': "Yes" if data['cv_stable'] else "No",
                                                'Consistency_Score': f"{data['consistency_score']:.3f}"
                                            })
                                        
                                        across_df = pd.DataFrame(across_summary)
                                        st.dataframe(across_df, use_container_width=True)
                                        
                                        st.info("""
                                        **Across-Individual Metrics Explanation:**
                                        - **Early/Late_Change**: Population-level changes at different timepoints
                                        - **Direction_Persistence**: Whether population direction is maintained
                                        - **Final_CV**: Coefficient of variation at final timepoint (lower = more stable)
                                        - **CV_Stable**: Whether CV < 20% (stable criterion)
                                        - **Consistency_Score**: Overall across-individual consistency
                                        """)
                                        
                                        # Population trajectory visualization
                                        st.subheader("📊 Population Trajectory Analysis")
                                        
                                        selected_protein_across = st.selectbox(
                                            "Select protein for population trajectory:",
                                            list(across_proteins.keys())[:20],
                                            key="across_protein_trajectory"
                                        )
                                        
                                        if selected_protein_across:
                                            across_protein_data = across_proteins[selected_protein_across]
                                            
                                            # Create population trajectory plot
                                            fig_pop_trajectory = go.Figure()
                                            
                                            # Population means
                                            timepoints = [baseline_tp_persist, intermediate_tp_persist, selected_final_tp_across]
                                            means = [
                                                across_protein_data['baseline_mean'],
                                                across_protein_data['intermediate_mean'],
                                                across_protein_data['final_mean']
                                            ]
                                            
                                            # Add mean trajectory
                                            fig_pop_trajectory.add_trace(go.Scatter(
                                                x=timepoints,
                                                y=means,
                                                mode='lines+markers',
                                                name='Population Mean',
                                                line=dict(width=4, color='blue'),
                                                marker=dict(size=10, color='blue')
                                            ))
                                            
                                            # Add CV bands (showing variability)
                                            cvs = [
                                                across_protein_data['baseline_cv'],
                                                across_protein_data['intermediate_cv'],
                                                across_protein_data['final_cv']
                                            ]
                                            
                                            # Calculate bands based on CV
                                            upper_band = [mean * (1 + cv/100) for mean, cv in zip(means, cvs)]
                                            lower_band = [mean * (1 - cv/100) for mean, cv in zip(means, cvs)]
                                            
                                            fig_pop_trajectory.add_trace(go.Scatter(
                                                x=timepoints + timepoints[::-1],
                                                y=upper_band + lower_band[::-1],
                                                fill='toself',
                                                fillcolor='rgba(0,0,255,0.2)',
                                                line=dict(color='rgba(255,255,255,0)'),
                                                name='CV Variability Band',
                                                showlegend=True
                                            ))
                                            
                                            # Add stability threshold line
                                            if across_protein_data['cv_stable']:
                                                stability_color = 'green'
                                                stability_text = 'Stable (CV < 20%)'
                                            else:
                                                stability_color = 'red'
                                                stability_text = 'Unstable (CV ≥ 20%)'
                                            
                                            fig_pop_trajectory.update_layout(
                                                title=f"Population Trajectory: {selected_protein_across}<br>"
                                                      f"{stability_text}, Direction Persistence: {across_protein_data['direction_persistence']:.3f}",
                                                xaxis_title="Timepoint",
                                                yaxis_title="Population Mean Level",
                                                height=500
                                            )
                                            
                                            st.plotly_chart(fig_pop_trajectory, use_container_width=True)
                                    else:
                                        st.warning("No across-individual persistence data available for selected timepoint.")
                                else:
                                    st.warning("Selected timepoint not available in across-individual analysis.")
                            else:
                                st.warning("No across-individual persistence analysis data available.")
                        
                        # RADAR PLOT ANALYSIS TAB
                        with radar_plot_tab:
                            st.subheader("🎯 Radar Plot Analysis - Biomarker Profiles")
                            
                            if persistence_results and within_across_results:
                                # Select proteins for radar plot
                                available_proteins = list(persistence_results.keys())
                                selected_proteins_radar = st.multiselect(
                                    "Select proteins for radar plot comparison (max 5 recommended):",
                                    available_proteins,
                                    default=available_proteins[:10] if len(available_proteins) >= 3 else available_proteins,
                                    key="radar_protein_select"
                                )
                                
                                # Select final timepoint for radar analysis
                                selected_final_tp_radar = st.selectbox(
                                    "Select final timepoint for radar analysis:",
                                    final_tps_persist,
                                    key="radar_final_tp_select"
                                )
                                
                                if selected_proteins_radar and selected_final_tp_radar in within_across_results:
                                    # Create radar plot
                                    fig_radar = go.Figure()
                                    
                                    # Define radar plot categories
                                    categories = [
                                        'Persistence Score',
                                        'Prediction Score', 
                                        'Within Direction Persistence',
                                        'Within Individual Stability',
                                        'Across Direction Persistence',
                                        'Across CV Stability',
                                        'Overall Consistency'
                                    ]
                                    
                                    colors = ['red', 'blue', 'green', 'orange', 'purple', "pink", "brown", "gray", "teal", "olive"]
                                    
                                    for i, protein in enumerate(selected_proteins_radar[:10]):  # Limit to 5 proteins
                                        if protein in persistence_results:
                                            persist_data = persistence_results[protein]
                                            
                                            # Get within/across data if available
                                            within_data = None
                                            across_data = None
                                            if selected_final_tp_radar in within_across_results:
                                                if protein in within_across_results[selected_final_tp_radar]:
                                                    within_data = within_across_results[selected_final_tp_radar][protein].get('within')
                                                    across_data = within_across_results[selected_final_tp_radar][protein].get('across')
                                            
                                            # Safely extract numeric values with type checking
                                            def safe_float(value, default=0.0):
                                                try:
                                                    if isinstance(value, str):
                                                        return float(value)
                                                    return float(value) if value is not None else default
                                                except (ValueError, TypeError):
                                                    return default
                                            
                                            # Calculate radar values (normalize to 0-1 scale) - FIXED
                                            avg_persistence = safe_float(persist_data.get('avg_persistence_score', 0))
                                            avg_prediction = safe_float(persist_data.get('avg_prediction_accuracy', 0))
                                            
                                            values = [
                                                avg_persistence,
                                                avg_prediction,
                                                safe_float(within_data.get('direction_persistence', 0) if within_data else 0),
                                                safe_float(within_data.get('individual_stability', 0) if within_data else 0),
                                                safe_float(across_data.get('direction_persistence', 0) if across_data else 0),
                                                1.0 if across_data and across_data.get('cv_stable', False) else 0.0,
                                                (avg_persistence + avg_prediction) / 2
                                            ]
                                            
                                            # Add trace for this protein
                                            fig_radar.add_trace(go.Scatterpolar(
                                                r=values + [values[0]],  # Close the polygon
                                                theta=categories + [categories[0]],  # Close the polygon
                                                fill='toself',
                                                name=protein,
                                                line_color=colors[i % len(colors)]
                                            ))
                                    
                                    fig_radar.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 1]
                                            )
                                        ),
                                        title=f"Biomarker Persistence & Prediction Profile<br>Final Timepoint: T{selected_final_tp_radar}",
                                        height=600,
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_radar, use_container_width=True)
                                    
                                    # Interpretation guide for radar plot
                                    st.info("""
                                    **Radar Plot Interpretation:**
                                    - **Outer edge (1.0)**: Optimal performance in that dimension
                                    - **Center (0.0)**: Poor performance in that dimension
                                    - **Larger area**: Better overall biomarker profile
                                    - **Regular shape**: Balanced performance across dimensions
                                    - **Irregular shape**: Strong in some areas, weak in others
                                    """)
                                    
                                    # Summary table for selected proteins
                                    st.subheader("📋 Selected Proteins Summary")
                                    
                                    radar_summary = []
                                    for protein in selected_proteins_radar:
                                        if protein in persistence_results:
                                            persist_data = persistence_results[protein]
                                            
                                            # Safely extract values with type checking
                                            def safe_float(value, default=0.0):
                                                try:
                                                    if isinstance(value, str):
                                                        return float(value)
                                                    return float(value) if value is not None else default
                                                except (ValueError, TypeError):
                                                    return default
                                            
                                            def safe_int(value, default=0):
                                                try:
                                                    if isinstance(value, str):
                                                        return int(value)
                                                    return int(value) if value is not None else default
                                                except (ValueError, TypeError):
                                                    return default
                                            
                                            n_meaningful = safe_int(persist_data.get('n_meaningful_early_changes', 0))
                                            n_total = safe_int(persist_data.get('n_total_participants', 1))
                                            
                                            radar_summary.append({
                                                'Protein': protein,
                                                'Category': persist_data.get('category', 'Unknown'),
                                                'Overall_Score': f"{safe_float(persist_data.get('overall_enhanced_score', 0)):.3f}",
                                                'Persistence': f"{safe_float(persist_data.get('avg_persistence_score', 0)):.3f}",
                                                'Prediction': f"{safe_float(persist_data.get('avg_prediction_accuracy', 0)):.3f}",
                                                'Meaningful_Participants': f"{n_meaningful}/{n_total}"
                                            })
                                    
                                    radar_summary_df = pd.DataFrame(radar_summary)
                                    st.dataframe(radar_summary_df, use_container_width=True)
                                
                                # Add download button for summary
                                    st.subheader("📥 Download Radar Plot Analysis")
                                    
                                    radar_csv =  across_df.to_csv(index=False)
                                    st.download_button(
                                        label="📊 Download Summary (CSV)",
                                        data=radar_csv,
                                        file_name=f"across_df{selected_final_tp_radar}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        help="Download across_df biomarker comparison results"
                                    )
                                    presistence_csv =  persistence_df.to_csv(index=False)
                                    st.download_button(
                                        label="📊 Download Summary (CSV)",
                                        data= presistence_csv,
                                        file_name=f"persistence_df{selected_final_tp_radar}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv",
                                        help="Download persistence_df biomarker comparison results"
                                    )
                                
                                else:
                                    st.warning("Please select proteins and ensure timepoint data is available.")
                            else:
                                st.warning("Persistence results not available for radar plot analysis.")
                    
                    else:
                        st.warning("Insufficient data for persistence analysis.")
                
                else:
                    st.warning("Please select at least one final timepoint for analysis.")
            
            else:
                st.info("👆 Select timepoints and click 'Run Enhanced Long Term Effects' to begin")
    
    else:
        # Instructions when no data is loaded
        st.info("""
        ### 🚀 Welcome to the Enhanced Omics Data Analysis Platform - Corrected Version!
        
        **Key Corrections Made:**
        
        **Statistical Interpretation Fixes:**
        - **Group-Level P-Value**: Now correctly described as testing average change across ALL participants
        - **Individual-Level Analysis**: Added proper binomial testing for meaningful changes (≥threshold)
        - **Separate Significance Types**: Group vs Individual significance clearly distinguished
        - **Clinical Relevance Hierarchy**: Both significant > Group only > Individual only > Neither
        
        **Enhanced Features:**
        - **Meaningful Change Threshold**: Customizable threshold (default: ≥2 units) for clinical significance
        - **Dual Statistical Testing**: Both population-level and individual-level insights
        - **Corrected Interpretations**: Clear explanations of what each test actually measures
        - **Clinical Priority Ranking**: Helps identify most clinically relevant biomarkers
        
        **Data Upload Requirements:**
        
        1. **Omics Data**: CSV/Excel with ID, Timepoint, and biomarker columns
        2. **Demographics Data**: Specific structure with ID, sex (col 4), age (col 5), timepoints (col 6-11)
        
        **Analysis Workflow:**
        1. **Data Overview**: Understand your dataset structure
        2. **Demographics**: Check participant characteristics 
        3. **Significant Analysis (CORRECTED)**: Identify biomarkers with both group and individual significance
        4. **Persistence Analysis**: Find long-term stable biomarkers
        
        The corrected version provides more accurate statistical interpretations and better clinical guidance!
        """)

if __name__ == "__main__":
    main()
