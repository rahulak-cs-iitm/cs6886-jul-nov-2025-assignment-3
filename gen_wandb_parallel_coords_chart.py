# wandb_compression_sweep.py - WandB plotting for compression pipeline sweep
import os
import sys
import subprocess
import argparse
import time
import itertools
import wandb
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any

# Import your configurable compression (assuming it's available)
try:
    from configurable_compression import run_configurable_compression
except ImportError:
    print("Warning: configurable_compression.py not found, will use subprocess calls")
    run_configurable_compression = None

class WandBCompressionSweep:
    """
    WandB sweep manager for compression pipeline visualization.
    """
    
    def __init__(self, project_name="mobilenetv2-compression-analysis", entity=None):
        self.project_name = project_name
        self.entity = entity
        self.results = []
        
    def run_single_experiment(self, weight_bits: int, activation_bits: int, 
                            pruned_model_path: str, experiment_id: int) -> Dict[str, Any]:
        """
        Run a single compression experiment and log to WandB.
        """
        
        # Initialize WandB run
        run_name = f"exp_{experiment_id:02d}_w{weight_bits}a{activation_bits}"
        
        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            config={
                "weight_bits": weight_bits,
                "activation_bits": activation_bits,
                "model": "MobileNetV2",
                "dataset": "CIFAR-10",
                "compression_method": "symmetric_quantization",
                "experiment_id": experiment_id,
                "bit_config": f"{weight_bits}W_{activation_bits}A"
            },
            reinit=True,
            tags=["compression", "quantization", f"{weight_bits}bit-weights", f"{activation_bits}bit-activations"]
        )
        
        print(f"\nEXPERIMENT {experiment_id}: {weight_bits}-bit weights, {activation_bits}-bit activations")
        print(f"WandB Run: {run.name}")
        
        try:
            # Run compression using the configurable script
            if run_configurable_compression:
                # Direct function call
                results = run_configurable_compression(
                    pruned_model_path=pruned_model_path,
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    save_model=False,  # Don't save models in sweep
                    evaluate_accuracy=True,
                    output_dir="./temp_checkpoints"
                )
            else:
                # Subprocess call to configurable_compression.py
                cmd = [
                    "python", "configurable_compression.py",
                    "--weight_bits", str(weight_bits),
                    "--activation_bits", str(activation_bits),
                    "--pruned_model", pruned_model_path,
                    "--no_save"  # Don't save models in sweep
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise Exception(f"Subprocess failed: {result.stderr}")
                
                # Parse results from stdout (simplified - would need actual parsing)
                results = self._parse_subprocess_output(result.stdout)
            
            if results is None:
                raise Exception("No results returned from compression")
            
            # Calculate additional metrics
            efficiency_score = (results.get('quantized_accuracy', 0) / 100) * results.get('total_theoretical_compression', 1)
            accuracy_per_compression = results.get('quantized_accuracy', 0) / results.get('total_theoretical_compression', 1)
            
            # Log comprehensive metrics to WandB
            metrics = {
                # Configuration
                'weight_bits': weight_bits,
                'activation_bits': activation_bits,
                'bit_product': weight_bits * activation_bits,
                'bit_sum': weight_bits + activation_bits,
                'bit_config': f"{weight_bits}W_{activation_bits}A",
                
                # Model structure
                'total_params': results.get('total_params', 0),
                'weight_params': results.get('weight_params', 0),
                'sparsity_ratio': results.get('weight_sparsity', 0),
                
                # Size metrics
                'baseline_size_mb': results.get('baseline_size_mb', 0),
                'theoretical_size_mb': results.get('sparse_size_mb', 0),
                'size_reduction_mb': results.get('baseline_size_mb', 0) - results.get('sparse_size_mb', 0),
                'size_reduction_percent': ((results.get('baseline_size_mb', 1) - results.get('sparse_size_mb', 0)) / results.get('baseline_size_mb', 1)) * 100,
                
                # Compression ratios
                'weight_compression_ratio': results.get('quantization_compression', 1),
                'sparse_compression_ratio': results.get('sparse_compression', 1),
                'activation_compression_ratio': results.get('activation_compression', 1),
                'total_compression_ratio': results.get('total_theoretical_compression', 1),
                
                # Accuracy metrics
                'original_accuracy': results.get('original_accuracy', 0),
                'quantized_accuracy': results.get('quantized_accuracy', 0),
                'accuracy_drop': results.get('accuracy_drop', 0),
                'accuracy_retention': results.get('accuracy_retention', 100),
                
                # Efficiency metrics
                'efficiency_score': efficiency_score,
                'accuracy_per_compression': accuracy_per_compression,
                'compression_per_bit': results.get('total_theoretical_compression', 1) / (weight_bits + activation_bits),
                
                # Performance metrics
                'quantization_time': results.get('quantization_time', 0),
                
                # Quality categories for analysis
                'compression_category': self._categorize_compression(results.get('total_theoretical_compression', 1)),
                'accuracy_category': self._categorize_accuracy(results.get('quantized_accuracy', 0)),
                'efficiency_category': self._categorize_efficiency(efficiency_score),
            }
            
            # Log metrics
            wandb.log(metrics)
            
            # Log custom plots
            self._log_custom_plots(wandb, metrics, weight_bits, activation_bits)
            
            print(f"✅ Logged metrics: {metrics['quantized_accuracy']:.2f}% accuracy, {metrics['total_compression_ratio']:.1f}x compression")
            
            # Store for local analysis
            self.results.append(metrics)
            
            return metrics
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            # Log failure to WandB
            wandb.log({
                'weight_bits': weight_bits,
                'activation_bits': activation_bits,
                'experiment_failed': True,
                'error_message': str(e)
            })
            return None
            
        finally:
            wandb.finish()
    
    def _parse_subprocess_output(self, output: str) -> Dict[str, Any]:
        """
        Parse subprocess output (simplified - implement based on actual output format).
        """
        # This is a placeholder - implement actual parsing based on your output format
        return {
            'quantized_accuracy': 93.0,  # Placeholder
            'total_theoretical_compression': 8.0,  # Placeholder
            'baseline_size_mb': 8.5,  # Placeholder
            'sparse_size_mb': 1.0,  # Placeholder
        }
    
    def _categorize_compression(self, compression_ratio: float) -> str:
        """Categorize compression ratio for analysis."""
        if compression_ratio >= 20:
            return "Ultra-High"
        elif compression_ratio >= 10:
            return "High"
        elif compression_ratio >= 5:
            return "Medium"
        elif compression_ratio >= 2:
            return "Low"
        else:
            return "Minimal"
    
    def _categorize_accuracy(self, accuracy: float) -> str:
        """Categorize accuracy for analysis."""
        if accuracy >= 93:
            return "Excellent"
        elif accuracy >= 90:
            return "Good"
        elif accuracy >= 85:
            return "Fair"
        else:
            return "Poor"
    
    def _categorize_efficiency(self, efficiency: float) -> str:
        """Categorize efficiency score."""
        if efficiency >= 10:
            return "Excellent"
        elif efficiency >= 5:
            return "Good"
        elif efficiency >= 2:
            return "Fair"
        else:
            return "Poor"
    
    def _log_custom_plots(self, wandb_run, metrics: Dict[str, Any], weight_bits: int, activation_bits: int):
        """
        Log custom plots and visualizations to WandB.
        """
        
        # Create a simple scatter plot data point
        wandb_run.log({
            "compression_vs_accuracy": wandb.plot.scatter(
                wandb.Table(data=[[metrics['total_compression_ratio'], metrics['quantized_accuracy']]], 
                           columns=["compression_ratio", "accuracy"]),
                "compression_ratio", "accuracy",
                title=f"Compression vs Accuracy ({weight_bits}W_{activation_bits}A)"
            )
        })
    
    def run_comprehensive_sweep(self, pruned_model_path: str, 
                              bit_configurations: List[Tuple[int, int]] = None,
                              max_experiments: int = None) -> pd.DataFrame:
        """
        Run comprehensive sweep across bit configurations.
        """
        
        if bit_configurations is None:
            # Default comprehensive configurations
            bit_configurations = [
                # Baseline and reference
                (32, 32),   # No quantization
                (32, 16), (32, 8),   # Only activation quantization
                (16, 32), (8, 32),   # Only weight quantization
                
                # Balanced configurations
                (16, 16),   # Moderate compression
                (12, 12),   # Custom precision
                (8, 8),     # Standard quantization
                
                # Mixed precision
                (16, 8), (8, 16),    # Mixed precision variants
                (12, 8), (8, 12),    # Custom mixed precision
                
                # Aggressive compression
                (8, 6), (6, 8),      # Aggressive activation/weight
                (6, 6),              # Ultra-low precision
                (8, 4), (4, 8),      # Very aggressive
                (4, 4),              # Maximum compression
            ]
        
        if max_experiments and len(bit_configurations) > max_experiments:
            bit_configurations = bit_configurations[:max_experiments]
        
        print(f"STARTING WANDB COMPRESSION SWEEP")
        print(f"Project: {self.project_name}")
        print(f"Configurations: {len(bit_configurations)}")
        print(f"Model: {pruned_model_path}")
        print(f"{'='*80}")
        
        # Verify model exists
        if not os.path.exists(pruned_model_path):
            print(f"Error: Pruned model not found at {pruned_model_path}")
            return pd.DataFrame()
        
        # Run experiments
        successful_experiments = 0
        failed_experiments = 0
        
        for i, (weight_bits, activation_bits) in enumerate(bit_configurations, 1):
            print(f"\n{'='*60}")
            print(f"EXPERIMENT {i}/{len(bit_configurations)}")
            print(f"Configuration: {weight_bits}-bit weights, {activation_bits}-bit activations")
            print(f"{'='*60}")
            
            try:
                start_time = time.time()
                result = self.run_single_experiment(
                    weight_bits=weight_bits,
                    activation_bits=activation_bits,
                    pruned_model_path=pruned_model_path,
                    experiment_id=i
                )
                
                experiment_time = time.time() - start_time
                
                if result:
                    successful_experiments += 1
                    print(f"Success in {experiment_time:.1f}s")
                else:
                    failed_experiments += 1
                    print(f"Failed in {experiment_time:.1f}s")
                
                # Brief pause between experiments
                time.sleep(2)
                
            except KeyboardInterrupt:
                print(f"\nScript interrupted by user")
                break
            except Exception as e:
                failed_experiments += 1
                print(f"Unexpected error: {e}")
                continue
        
        # Create results DataFrame
        if self.results:
            results_df = pd.DataFrame(self.results)
            
            # Save results locally
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            csv_path = f"./logs/wandb_stats/wandb_compression_results_{timestamp}.csv"
            results_df.to_csv(csv_path, index=False)
            
            print(f"\n{'='*80}")
            print("SWEEP SUMMARY")
            print(f"{'='*80}")
            print(f"  Successful experiments: {successful_experiments}")
            print(f"  Failed experiments: {failed_experiments}")
            print(f"Results saved to: {csv_path}")
            
            # Print best configurations
            if successful_experiments > 0:
                best_accuracy = results_df.loc[results_df['quantized_accuracy'].idxmax()]
                best_compression = results_df.loc[results_df['total_compression_ratio'].idxmax()]
                best_efficiency = results_df.loc[results_df['efficiency_score'].idxmax()]
                
                print(f"\nBEST CONFIGURATIONS:")
                print(f"   Accuracy: {best_accuracy['quantized_accuracy']:.2f}% ({best_accuracy['bit_config']})")
                print(f"   Compression: {best_compression['total_compression_ratio']:.1f}x ({best_compression['bit_config']})")
                print(f"   Efficiency: {best_efficiency['efficiency_score']:.2f} ({best_efficiency['bit_config']})")
            
            print(f"\nWANDB VISUALIZATION:")
            print(f"   Project: {self.project_name}")
            print(f"   URL: https://wandb.ai/{self.entity or 'your-username'}/{self.project_name}")
            print(f"   Recommended panels:")
            print(f"     • Parallel Coordinates: weight_bits, activation_bits, quantized_accuracy, total_compression_ratio")
            print(f"     • Scatter: total_compression_ratio vs quantized_accuracy")
            print(f"     • Line Plot: efficiency_score over experiment_id")
            print(f"     • Bar Chart: compression_category distribution")
            
            return results_df
        else:
            print(f"\nNo successful experiments to analyze")
            return pd.DataFrame()

def main():
    """
    Main function with command-line interface.
    """
    
    parser = argparse.ArgumentParser(
        description='WandB Compression Sweep for MobileNetV2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full sweep with default configurations
    python wandb_compression_sweep.py
    
    # Custom bit ranges
    python wandb_compression_sweep.py --weight_bits 32 16 8 4 --activation_bits 32 16 8
    
    # Limited number of experiments
    python wandb_compression_sweep.py --max_experiments 10
    
    # Custom WandB project
    python wandb_compression_sweep.py --project "my-compression-study" --entity "my-team"
        """
    )
    
    parser.add_argument('--pruned_model', type=str,
                       default="./checkpoints/mobilenetv2_cifar10_pruned.pth",
                       help='Path to the pruned model')
    
    parser.add_argument('--project', type=str,
                       default="mobilenetv2-compression-analysis",
                       help='WandB project name')
    
    parser.add_argument('--entity', type=str,
                       help='WandB entity (username/team)')
    
    parser.add_argument('--weight_bits', type=int, nargs='+',
                       default=[32, 16, 12, 8, 6, 4],
                       help='Weight bit-widths to test')
    
    parser.add_argument('--activation_bits', type=int, nargs='+',
                       default=[32, 16, 12, 8, 6, 4],
                       help='Activation bit-widths to test')
    
    parser.add_argument('--custom_configs', type=str,
                       help='Custom configurations: "32,32;16,8;8,4"')
    
    parser.add_argument('--max_experiments', type=int,
                       help='Maximum number of experiments to run')
    
    parser.add_argument('--dry_run', action='store_true',
                       help='Print configurations without running experiments')
    
    args = parser.parse_args()
    
    # Determine bit configurations
    if args.custom_configs:
        configs = []
        for pair in args.custom_configs.split(';'):
            w_bits, a_bits = map(int, pair.split(','))
            configs.append((w_bits, a_bits))
        bit_configurations = configs
    else:
        # Generate all combinations
        bit_configurations = list(itertools.product(args.weight_bits, args.activation_bits))
    
    # Apply max experiments limit
    if args.max_experiments:
        bit_configurations = bit_configurations[:args.max_experiments]
    
    # Print configuration
    print(f"WandB Compression Sweep Configuration:")
    print(f"  Project: {args.project}")
    print(f"  Entity: {args.entity or 'default'}")
    print(f"  Model: {args.pruned_model}")
    print(f"  Configurations: {len(bit_configurations)}")
    print(f"  Bit configs: {bit_configurations}")
    
    if args.dry_run:
        print("\nDRY RUN - No experiments will be executed")
        return
    
    # Verify model exists
    if not os.path.exists(args.pruned_model):
        print(f"Error: Model not found at {args.pruned_model}")
        return
    
    # Initialize sweep manager
    sweep_manager = WandBCompressionSweep(
        project_name=args.project,
        entity=args.entity
    )
    
    # Run sweep
    try:
        results_df = sweep_manager.run_comprehensive_sweep(
            pruned_model_path=args.pruned_model,
            bit_configurations=bit_configurations,
            max_experiments=args.max_experiments
        )
        
        print(f"\nSweep completed successfully!")
        print(f"Check your WandB dashboard for interactive visualizations")
        
    except KeyboardInterrupt:
        print(f"\nScript interrupted by user")
    except Exception as e:
        print(f"Sweep failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

