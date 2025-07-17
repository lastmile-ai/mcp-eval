import os
import argparse
import json
from typing import List

from core_trace_process import create_trace_dataset, get_tools_info, separate_traces_by_server
from predictors import ToolPredictor
from mcp_eval.metrics import process_spans

def optimize_with_dspy(predictor: ToolPredictor, list_of_available_tools: list, train_examples: list, test_examples: list, optimizer_type: str, optimizer_kwargs: dict) -> dict:
    """Use DSPy optimizers to improve tool selection"""
    
    # Baseline evaluation
    print("\n=== Baseline Evaluation ===")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(test_examples)}")

    # Optimize docstrings using available tools and examples
    print("\n=== Optimizing Docstrings ===")
    optimization_report = predictor.optimize_docstrings(
        examples=train_examples,
        tools_list=list_of_available_tools
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    return {
        "optimizer": optimizer_type, 
        "train_count": len(train_examples), 
        "val_count": len(test_examples),
        "optimization_report": optimization_report
    }


def save_results(results: dict, output_path: str) -> None:
    """Save optimization results to a file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_path}")

def process_server_optimization(server_name: str, trace_files: List[str], processed_files: List[str], args) -> dict:
    """Process optimization for a specific server"""
    print(f"\n=== Processing server: {server_name} ===")
    
    examples = create_trace_dataset(trace_files, processed_files)
    
    if args.limit:
        examples = examples[:args.limit]
    
    # Extract available tools from the first trace file
    list_of_available_tools = []
    if trace_files:
        list_of_available_tools = get_tools_info(trace_files[0])
        print(f"Loaded {len(list_of_available_tools)} available tools for {server_name}")
        
    # Simple train/test split
    train_size = int(len(examples) * args.train_ratio)
    train_examples = examples[:train_size]
    test_examples = examples[train_size:]
    predictor = ToolPredictor(model_name=args.model)

    print(f"Loaded {len(examples)} examples for {server_name}")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Testing examples: {len(test_examples)}")
    
    # Simplified optimizer kwargs for docstring optimization only
    optimizer_kwargs = {}
    
    # Run optimization
    results = optimize_with_dspy(
        predictor=predictor,
        list_of_available_tools=list_of_available_tools,
        train_examples=train_examples,
        test_examples=test_examples,
        optimizer_type=args.optimizer,
        optimizer_kwargs=optimizer_kwargs
    )
    
    # Add server-specific info to results
    results['server_name'] = server_name
    results['optimizer'] = args.optimizer
    results['optimizer_kwargs'] = optimizer_kwargs
    
    return results


def main(args) -> None:
    """Main function to run the tool selection optimization with DSPy"""
    print(f"Loading trace dataset from {args.trace_directory}")
    
    # Read directory content and find trace file pairs
    if not os.path.exists(args.trace_directory):
        raise ValueError(f"Directory {args.trace_directory} does not exist")
    
    trace_files = []
    processed_files = []
    
    for filename in os.listdir(args.trace_directory):
        if filename.endswith('_trace.jsonl'):
            # Extract base name without '_trace.jsonl'
            base_name = filename[:-12]  # Remove '_trace.jsonl'
            
            trace_path = os.path.join(args.trace_directory, filename)
            processed_path = os.path.join(args.trace_directory, f"{base_name}.json")
            
            if os.path.exists(processed_path):
                trace_files.append(trace_path)
                processed_files.append(processed_path)
            else:
                print(f"Warning: No matching processed file found for {filename}")
    
    print(f"Found {len(trace_files)} trace file pairs in {args.trace_directory}")
    
    # Handle server-specific processing
    if args.server_name:
        # Process only the specified server
        print(f"Processing only server: {args.server_name}")
        
        # Separate traces by server
        server_traces = separate_traces_by_server(trace_files)
        
        if args.server_name not in server_traces:
            print(f"Error: Server '{args.server_name}' not found in trace files")
            print(f"Available servers: {list(server_traces.keys())}")
            return
        
        # Get matching processed files for this server
        server_trace_files = server_traces[args.server_name]
        server_processed_files = []
        
        for trace_file in server_trace_files:
            # Find corresponding processed file
            for trace_path, processed_path in zip(trace_files, processed_files):
                if trace_path == trace_file:
                    server_processed_files.append(processed_path)
                    break
        
        # Process the specific server
        results = process_server_optimization(args.server_name, server_trace_files, server_processed_files, args)
        
        # Save results if output path provided
        if args.output:
            output_path = args.output.replace('.json', f'_{args.server_name}.json')
            save_results(results, output_path)
        
        # Save optimization report as separate JSON file
        if 'optimization_report' in results and results['optimization_report']:
            report_path = args.output.replace('.json', f'_{args.server_name}_report.json') if args.output else f'{args.server_name}_optimization_report.json'
            save_results(results['optimization_report'], report_path)
            print(f"Optimization report saved to {report_path}")
    
    else:
        # Process all servers separately
        print("Processing all servers separately")
        
        # Separate traces by server
        server_traces = separate_traces_by_server(trace_files)
        
        print(f"Found {len(server_traces)} servers: {list(server_traces.keys())}")
        
        all_results = {}
        
        for server_name, server_trace_files in server_traces.items():
            # Get matching processed files for this server
            server_processed_files = []
            
            for trace_file in server_trace_files:
                # Find corresponding processed file
                for trace_path, processed_path in zip(trace_files, processed_files):
                    if trace_path == trace_file:
                        server_processed_files.append(processed_path)
                        break
            
            # Process this server
            results = process_server_optimization(server_name, server_trace_files, server_processed_files, args)
            all_results[server_name] = results
            
            # Save individual server results
            if args.output:
                output_path = args.output.replace('.json', f'_{server_name}.json')
                save_results(results, output_path)
            
            # Save optimization report for this server
            if 'optimization_report' in results and results['optimization_report']:
                report_path = args.output.replace('.json', f'_{server_name}_report.json') if args.output else f'{server_name}_optimization_report.json'
                save_results(results['optimization_report'], report_path)
                print(f"Optimization report saved to {report_path}")
        
        # Save combined results
        if args.output:
            combined_path = args.output.replace('.json', '_combined.json')
            save_results(all_results, combined_path)
            print(f"Combined results saved to {combined_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Selection Optimizer with DSPy")
    parser.add_argument("--trace-directory", type=str,
                      help="Directory containing trace files", 
                      default='examples/mcp_server_fetch/test-reports')
    parser.add_argument("--output", type=str, default=None,
                      help="Path to save results JSON")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini",
                      help="Model to use for optimization")
    parser.add_argument("--optimizer", type=str, default="bootstrap",
                      help="Optimizer type (fewshot, bootstrap, mipro)")
    parser.add_argument("--limit", type=int, default=50,
                      help="Limit number of examples to load")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                      help="Ratio of data to use for training")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--server-name", type=str, default=None,
                      help="Specific server name to optimize docstrings for")
    
    # FewShot parameters
    parser.add_argument("--k", type=int, default=5,
                      help="Number of examples for few-shot learning")
    
    # Bootstrap parameters
    parser.add_argument("--num-bootstrapped", type=int, default=5,
                      help="Number of bootstrapped example sets")
    parser.add_argument("--max-demos", type=int, default=3,
                      help="Maximum demonstrations per bootstrapped set")
    
    # MIPRO parameters
    parser.add_argument("--num-epochs", type=int, default=3,
                      help="Number of epochs for MIPRO")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for MIPRO")
    
    args = parser.parse_args()
    main(args)