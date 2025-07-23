import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class ProteinStabilityAnalyzer:
    def __init__(self, data_path=None, df=None):
        """
        Initialize the analyzer with protein data
        
        Parameters:
        data_path (str): Path to CSV file
        df (DataFrame): Pre-loaded DataFrame
        """
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            # Generate simulated data matching your structure
            self.df = self._generate_simulated_data()
        
        self.protein_cols = [col for col in self.df.columns if col.startswith('Protein_') or col.startswith('protein_')]
        self.baseline_time = 1
        self.comparison_times = [3, 4, 5, 6]
        
    def _generate_simulated_data(self):
        """Generate simulated data matching your structure"""
        np.random.seed(42)
        
        # Parameters matching your data
        n_samples = 680
        n_unique_ids = 172
        n_timepoints = 6
        n_proteins = 100  # Reduced for demo, adjust as needed
        
        data = []
        final_id = 1
        
        for unique_id in range(1, n_unique_ids + 1):
            # Create individual-specific baseline effects
            individual_effect = np.random.normal(1, 0.2)
            
            for timepoint in range(1, n_timepoints + 1):
                row = {
                    'FinalID': final_id,
                    'ID': unique_id,
                    'Timepoints': timepoint
                }
                
                # Time-dependent effects
                time_effect = 1 + (timepoint - 1) * 0.05 * np.random.normal(1, 0.1)
                
                # Generate protein values with varying stability
                for protein_idx in range(1, n_proteins + 1):
                    protein_name = f'Protein_{protein_idx}'
                    
                    # Protein-specific baseline
                    protein_baseline = np.random.normal(2, 0.5)
                    
                    # Stability coefficient (some proteins more stable than others)
                    stability = np.random.beta(2, 1)  # Skewed towards stable
                    
                    # Measurement noise
                    noise = np.random.normal(0, 0.1)
                    
                    # Final value calculation
                    value = (protein_baseline * individual_effect * time_effect * stability) + noise
                    row[protein_name] = max(0, value)  # Ensure non-negative
                
                data.append(row)
                final_id += 1
        
        return pd.DataFrame(data)
    
    def calculate_within_individual_stability(self):
        """
        Calculate stability coefficients for each protein within each individual
        comparing Time 3-6 vs baseline (Time 1)
        """
        stability_results = {}
        
        for protein in self.protein_cols:
            stability_results[protein] = {}
            
            for individual_id in self.df['ID'].unique():
                individual_data = self.df[self.df['ID'] == individual_id]
                
                # Get baseline value (Time 1)
                baseline_row = individual_data[individual_data['Timepoints'] == self.baseline_time]
                if baseline_row.empty:
                    continue
                    
                baseline_value = baseline_row[protein].iloc[0]
                
                # Get comparison timepoints (3-6)
                comparison_values = []
                for time in self.comparison_times:
                    time_row = individual_data[individual_data['Timepoints'] == time]
                    if not time_row.empty:
                        comparison_values.append(time_row[protein].iloc[0])
                
                if len(comparison_values) > 1:
                    # Calculate correlation between baseline and comparison periods
                    baseline_repeated = [baseline_value] * len(comparison_values)
                    correlation, p_value = pearsonr(baseline_repeated, comparison_values)
                    
                    # Alternative: Calculate coefficient of variation
                    cv = np.std(comparison_values) / np.mean(comparison_values) if np.mean(comparison_values) != 0 else np.inf
                    stability_coef = 1 / (1 + cv)  # Higher values = more stable
                    
                    stability_results[protein][individual_id] = {
                        'correlation': correlation if not np.isnan(correlation) else 0,
                        'stability_coefficient': stability_coef,
                        'baseline_value': baseline_value,
                        'comparison_values': comparison_values,
                        'cv': cv
                    }
        
        return stability_results
    
    def calculate_across_individual_stability(self):
        """
        Calculate stability coefficients across all individuals
        """
        across_stability = {}
        
        for protein in self.protein_cols:
            across_stability[protein] = {}
            
            # Get baseline values across all individuals
            baseline_data = self.df[self.df['Timepoints'] == self.baseline_time]
            baseline_values = baseline_data[protein].values
            
            for time in self.comparison_times:
                time_data = self.df[self.df['Timepoints'] == time]
                time_values = time_data[protein].values
                
                if len(baseline_values) > 0 and len(time_values) > 0:
                    # Calculate correlation between baseline and each timepoint
                    min_len = min(len(baseline_values), len(time_values))
                    correlation, p_value = pearsonr(baseline_values[:min_len], time_values[:min_len])
                    
                    across_stability[protein][time] = {
                        'correlation': correlation if not np.isnan(correlation) else 0,
                        'p_value': p_value,
                        'n_samples': min_len
                    }
        
        return across_stability
    
    def calculate_effect_sizes(self):
        """
        Calculate Cohen's d effect sizes comparing each timepoint to baseline
        """
        effect_sizes = {}
        
        for protein in self.protein_cols:
            effect_sizes[protein] = {}
            
            # Get baseline values
            baseline_data = self.df[self.df['Timepoints'] == self.baseline_time]
            baseline_values = baseline_data[protein].values
            baseline_mean = np.mean(baseline_values)
            baseline_std = np.std(baseline_values)
            
            for time in self.comparison_times:
                time_data = self.df[self.df['Timepoints'] == time]
                time_values = time_data[protein].values
                time_mean = np.mean(time_values)
                time_std = np.std(time_values)
                
                # Calculate Cohen's d
                pooled_std = np.sqrt(((len(baseline_values) - 1) * baseline_std**2 + 
                                    (len(time_values) - 1) * time_std**2) / 
                                   (len(baseline_values) + len(time_values) - 2))
                
                cohens_d = (time_mean - baseline_mean) / pooled_std if pooled_std != 0 else 0
                
                effect_sizes[protein][time] = {
                    'cohens_d': cohens_d,
                    'baseline_mean': baseline_mean,
                    'time_mean': time_mean,
                    'baseline_std': baseline_std,
                    'time_std': time_std
                }
        
        return effect_sizes
    
    def create_stability_heatmap(self, within_stability, top_n_proteins=20):
        """
        Create heatmap of stability coefficients
        """
        # Prepare data for heatmap
        proteins = list(within_stability.keys())[:top_n_proteins]
        individuals = list(range(1, min(21, len(self.df['ID'].unique()) + 1)))  # Top 20 individuals
        
        heatmap_data = []
        for protein in proteins:
            row = []
            for individual in individuals:
                if individual in within_stability[protein]:
                    row.append(within_stability[protein][individual]['stability_coefficient'])
                else:
                    row.append(0)
            heatmap_data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=[f'ID_{i}' for i in individuals],
                   yticklabels=[p.replace('Protein_', 'P') for p in proteins],
                   cmap='RdYlBu_r', 
                   center=0.5,
                   annot=False,
                   fmt='.2f',
                   cbar_kws={'label': 'Stability Coefficient'})
        
        plt.title('Within-Individual Protein Stability Coefficients\n(Time 3-6 vs Baseline)', fontsize=14)
        plt.xlabel('Individual ID', fontsize=12)
        plt.ylabel('Proteins', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def create_effect_size_distribution(self, effect_sizes):
        """
        Create distribution plot of effect sizes
        """
        all_effect_sizes = []
        for protein in effect_sizes:
            for time in effect_sizes[protein]:
                all_effect_sizes.append(effect_sizes[protein][time]['cohens_d'])
        
        plt.figure(figsize=(10, 6))
        plt.hist(all_effect_sizes, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', label='No Effect')
        plt.axvline(x=0.2, color='orange', linestyle='--', label='Small Effect')
        plt.axvline(x=0.5, color='green', linestyle='--', label='Medium Effect')
        plt.axvline(x=0.8, color='purple', linestyle='--', label='Large Effect')
        
        plt.xlabel('Cohen\'s d (Effect Size)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Effect Sizes\n(Comparison Times vs Baseline)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def create_individual_trajectories(self, top_n_proteins=10, top_n_individuals=10):
        """
        Create individual trajectory plots for top proteins
        """
        proteins = self.protein_cols[:top_n_proteins]
        individuals = self.df['ID'].unique()[:top_n_individuals]
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for i, protein in enumerate(proteins):
            ax = axes[i]
            
            for individual in individuals:
                individual_data = self.df[self.df['ID'] == individual].sort_values('Timepoints')
                timepoints = individual_data['Timepoints'].values
                values = individual_data[protein].values
                
                ax.plot(timepoints, values, alpha=0.3, color='blue')
            
            # Add mean trajectory
            mean_trajectory = []
            for time in sorted(self.df['Timepoints'].unique()):
                time_data = self.df[self.df['Timepoints'] == time]
                mean_trajectory.append(time_data[protein].mean())
            
            ax.plot(sorted(self.df['Timepoints'].unique()), mean_trajectory, 
                   color='red', linewidth=3, label='Mean')
            
            ax.set_title(f'{protein}', fontsize=10)
            ax.set_xlabel('Timepoint')
            ax.set_ylabel('Protein Level')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        plt.suptitle('Individual Protein Trajectories Over Time', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_across_individual_plot(self, across_stability):
        """
        Create plot showing across-individual stability
        """
        proteins = list(across_stability.keys())[:20]  # Top 20 proteins
        timepoints = self.comparison_times
        
        correlations = []
        for protein in proteins:
            protein_corrs = []
            for time in timepoints:
                if time in across_stability[protein]:
                    protein_corrs.append(across_stability[protein][time]['correlation'])
                else:
                    protein_corrs.append(0)
            correlations.append(protein_corrs)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlations, 
                   xticklabels=[f'Time {t}' for t in timepoints],
                   yticklabels=[p.replace('Protein_', 'P') for p in proteins],
                   cmap='RdYlBu_r', 
                   center=0,
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation with Baseline'})
        
        plt.title('Across-Individual Protein Stability\n(Correlation with Baseline)', fontsize=14)
        plt.xlabel('Timepoint', fontsize=12)
        plt.ylabel('Proteins', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def generate_summary_statistics(self, within_stability, across_stability, effect_sizes):
        """
        Generate comprehensive summary statistics
        """
        print("=" * 60)
        print("PROTEIN STABILITY ANALYSIS SUMMARY")
        print("=" * 60)
        
        # Data overview
        print(f"Dataset Overview:")
        print(f"  • Total samples: {len(self.df)}")
        print(f"  • Unique individuals: {len(self.df['ID'].unique())}")
        print(f"  • Timepoints: {sorted(self.df['Timepoints'].unique())}")
        print(f"  • Proteins analyzed: {len(self.protein_cols)}")
        print()
        
        # Within-individual stability
        all_within_stability = []
        for protein in within_stability:
            for individual in within_stability[protein]:
                all_within_stability.append(within_stability[protein][individual]['stability_coefficient'])
        
        print(f"Within-Individual Stability:")
        print(f"  • Mean stability coefficient: {np.mean(all_within_stability):.3f} ± {np.std(all_within_stability):.3f}")
        print(f"  • Range: {np.min(all_within_stability):.3f} - {np.max(all_within_stability):.3f}")
        print(f"  • Highly stable proteins (>0.8): {np.sum(np.array(all_within_stability) > 0.8)}")
        print()
        
        # Across-individual stability
        all_across_correlations = []
        for protein in across_stability:
            for time in across_stability[protein]:
                all_across_correlations.append(across_stability[protein][time]['correlation'])
        
        print(f"Across-Individual Stability:")
        print(f"  • Mean correlation with baseline: {np.mean(all_across_correlations):.3f} ± {np.std(all_across_correlations):.3f}")
        print(f"  • Range: {np.min(all_across_correlations):.3f} - {np.max(all_across_correlations):.3f}")
        print(f"  • Strong correlations (>0.7): {np.sum(np.array(all_across_correlations) > 0.7)}")
        print()
        
        # Effect sizes
        all_effect_sizes = []
        for protein in effect_sizes:
            for time in effect_sizes[protein]:
                all_effect_sizes.append(abs(effect_sizes[protein][time]['cohens_d']))
        
        print(f"Effect Sizes:")
        print(f"  • Mean absolute effect size: {np.mean(all_effect_sizes):.3f} ± {np.std(all_effect_sizes):.3f}")
        print(f"  • Small effects (0.2-0.5): {np.sum((np.array(all_effect_sizes) >= 0.2) & (np.array(all_effect_sizes) < 0.5))}")
        print(f"  • Medium effects (0.5-0.8): {np.sum((np.array(all_effect_sizes) >= 0.5) & (np.array(all_effect_sizes) < 0.8))}")
        print(f"  • Large effects (>0.8): {np.sum(np.array(all_effect_sizes) >= 0.8)}")
        print()
        
        return {
            'within_stability_stats': {
                'mean': np.mean(all_within_stability),
                'std': np.std(all_within_stability),
                'min': np.min(all_within_stability),
                'max': np.max(all_within_stability)
            },
            'across_stability_stats': {
                'mean': np.mean(all_across_correlations),
                'std': np.std(all_across_correlations),
                'min': np.min(all_across_correlations),
                'max': np.max(all_across_correlations)
            },
            'effect_size_stats': {
                'mean': np.mean(all_effect_sizes),
                'std': np.std(all_effect_sizes),
                'min': np.min(all_effect_sizes),
                'max': np.max(all_effect_sizes)
            }
        }
    
    def run_complete_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("Starting protein stability analysis...")
        
        # Calculate stability coefficients
        print("1. Calculating within-individual stability coefficients...")
        within_stability = self.calculate_within_individual_stability()
        
        print("2. Calculating across-individual stability coefficients...")
        across_stability = self.calculate_across_individual_stability()
        
        print("3. Calculating effect sizes...")
        effect_sizes = self.calculate_effect_sizes()
        
        print("4. Generating summary statistics...")
        summary_stats = self.generate_summary_statistics(within_stability, across_stability, effect_sizes)
        
        print("5. Creating visualizations...")
        self.create_stability_heatmap(within_stability)
        self.create_effect_size_distribution(effect_sizes)
        self.create_individual_trajectories()
        self.create_across_individual_plot(across_stability)
        
        return {
            'within_stability': within_stability,
            'across_stability': across_stability,
            'effect_sizes': effect_sizes,
            'summary_stats': summary_stats
        }

# Usage example
if __name__ == "__main__":
    # Initialize analyzer with your data
    # Option 1: Load from CSV file
    # analyzer = ProteinStabilityAnalyzer(data_path='your_protein_data.csv')
    
    # Option 2: Use simulated data (for demonstration)
    analyzer = ProteinStabilityAnalyzer()
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    # Access individual results
    within_stability = results['within_stability']
    across_stability = results['across_stability']
    effect_sizes = results['effect_sizes']
    summary_stats = results['summary_stats']
    
    print("\nAnalysis complete! Results are available in the 'results' dictionary.")
    print("Key components:")
    print("  • within_stability: Individual-level stability coefficients")
    print("  • across_stability: Population-level stability correlations")
    print("  • effect_sizes: Cohen's d effect sizes vs baseline")
    print("  • summary_stats: Comprehensive statistical summary")