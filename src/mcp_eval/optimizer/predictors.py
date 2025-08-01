import os
import dspy
from typing import List, Dict, Any, Tuple

class ToolCall(dspy.Module):
    """Module representing a tool call prediction"""
    query = dspy.InputField(desc="The query to be answered by the tool.")
    available_tools = dspy.InputField(desc="The list of available tools.")
    selected_tool = dspy.OutputField(desc="The tool selected based on the query.")
    tool_arguments = dspy.OutputField(desc="The extracted arguments for the tool call.")


class DocstringImprover(dspy.Signature):
    """Signature for improving tool docstrings based on failed examples"""
    tool_name = dspy.InputField(desc="Name of the tool to optimize docstring for")
    original_docstring = dspy.InputField(desc="Original docstring of the tool")
    failed_examples = dspy.InputField(desc="List of examples where the tool selection failed")
    correct_examples = dspy.InputField(desc="List of examples where the tool was correctly selected")
    improved_docstring = dspy.OutputField(desc="Improved docstring that better describes the tool's purpose")


class ToolPredictor(dspy.Module):
    """Module for optimizing tool docstrings to improve tool selection accuracy"""
    def __init__(self, model_name: str = "openai/gpt-4o", temperature: float = 0.2):
        super().__init__()
        
        # Configure LLM
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in the .env file.")
        
        # Initialize the language model
        self.lm = dspy.LM(model_name, temperature=temperature)
        dspy.configure(lm=self.lm)
        
        self.tool_selector = dspy.ChainOfThought(
            "query, available_tools -> selected_tool, tool_arguments"
        )
        
        self.docstring_improver = dspy.ChainOfThought(DocstringImprover)        
        self.optimized_docstrings = {}
    
    def predict_tool(self, query: str, available_tools: List[Dict[str, Any]]) -> Tuple[str, Dict]:
        """
        Predict which tool should be used for a given query.
        
        Args:
            query: The user query
            available_tools: List of available tools with names and descriptions
            
        Returns:
            Tuple of (selected_tool_name, tool_arguments)
        """
        # Update tools with optimized docstrings if available
        updated_tools = []
        for tool in available_tools:
            tool_copy = tool.copy()
            tool_name = tool_copy.get('name', '')
            if tool_name in self.optimized_docstrings:
                tool_copy['description'] = self.optimized_docstrings[tool_name]
            updated_tools.append(tool_copy)
        
        # Make prediction
        prediction = self.tool_selector(
            query=query,
            available_tools=updated_tools
        )
        
        return prediction.selected_tool, prediction.tool_arguments
    

    
    def train_tool_selector_agent_with_optimizer(
        self, 
        available_tools: List[Dict[str, Any]],
        train_examples: List[dspy.Example], 
        dev_examples: List[dspy.Example],
        optimizer_class=dspy.teleprompt.BootstrapFewShot,
        metric=None,
        **optimizer_kwargs):
        """
        Train the tool selector and docstring improver using a DSPy optimizer.
        
        Args:
            train_examples: Examples to use for training
            dev_examples: Examples to use for validation
            optimizer_class: DSPy optimizer class to use
            metric: Metric to optimize for
            **optimizer_kwargs: Additional arguments for the optimizer
            
        Returns:
            Optimized predictor module
        """
        
        print("=== Step 1: Optimizing Tool Selector ===")
        
        # Create default metric if none provided
        if metric is None:
            def default_metric(example, pred, trace=None):
                return 1.0 if pred.selected_tool == example.correct_tool else 0.0
            metric = default_metric
        
        # Create and configure the optimizer for tool selector
        if optimizer_class == dspy.teleprompt.LabeledFewShot:
            # LabeledFewShot doesn't use metric parameter
            optimizer = optimizer_class(**optimizer_kwargs)
        else:
            optimizer = optimizer_class(
                metric=metric,
                **optimizer_kwargs
            )
        
        # Compile the tool selector
        optimized_selector = optimizer.compile(
            student=self.tool_selector,
            trainset=train_examples,
            # valset=dev_examples
        )
        
        # Update the selector with the optimized version
        self.tool_selector = optimized_selector
        
        print("=== Step 2: Predicting Tools for All Examples ===")
        
        # Combine all examples for comprehensive analysis
        predictions = []
        
        # Predict tool and arguments for all examples
        for example in train_examples:
            try:
                # Use the optimized tool selector to make predictions
                prediction = self.tool_selector(
                    query=example.query,
                    available_tools=available_tools
                )
                
                predictions.append({
                    'example': example,
                    'predicted_tool': prediction.selected_tool,
                    'correct_tool': example.correct_tool,
                    'is_correct': prediction.selected_tool == example.correct_tool,
                    'tool_arguments': prediction.tool_arguments
                })
            except Exception as e:
                print(f"Error predicting for example: {e}")
                predictions.append({
                    'example': example,
                    'predicted_tool': None,
                    'correct_tool': example.correct_tool,
                    'is_correct': False,
                    'tool_arguments': None
                })
        
        # Calculate accuracy metric
        correct_predictions = sum(1 for p in predictions if p['is_correct'])
        accuracy = correct_predictions / len(predictions) if predictions else 0.0
        print(f"Tool selector accuracy: {accuracy:.4f} ({correct_predictions}/{len(predictions)})")
        
        print("=== Step 3: Labeling Examples as Successful/Failed ===")
        
        # Label examples based on tool selection accuracy
        successful_examples = [p for p in predictions if p['is_correct']]
        failed_examples = [p for p in predictions if not p['is_correct']]
        
        print(f"Successful examples: {len(successful_examples)}")
        print(f"Failed examples: {len(failed_examples)}")
        
        print("=== Step 4: Extracting Tools and Grouping Examples ===")
        
        # Group examples by tool for docstring optimization
        tool_examples = {}
        
        for prediction in predictions:
            tool_name = prediction['correct_tool']  # Group by correct tool
            if tool_name not in tool_examples:
                tool_examples[tool_name] = {
                    'tool_name': tool_name,
                    'successful_examples': [],
                    'failed_examples': [],
                    'original_docstring': None
                }
            
            if prediction['is_correct']:
                tool_examples[tool_name]['successful_examples'].append(prediction['example'])
            else:
                tool_examples[tool_name]['failed_examples'].append(prediction['example'])
        
        # Extract original docstrings for each tool
        for tool_name in tool_examples:
            # Find the tool in available tools and get its original docstring
            for example in dev_examples:
                for tool_info in example.available_tools:
                    if tool_info.get('name') == tool_name:
                        tool_examples[tool_name]['original_docstring'] = tool_info.get('description', '')
                        break
                if tool_examples[tool_name]['original_docstring'] is not None:
                    break
        
        print(f"Found examples for {len(tool_examples)} unique tools")
        
    def optimize_docstrings(self, examples: List[dspy.Example], tools_list: List[Dict[str, Any]]):
        """Optimize docstrings for tools based on successful and failed examples"""
        print("=== Starting Docstring Optimization ===")
        print(f"Processing {len(tools_list)} tools with {len(examples)} examples")
        
        # Store optimization report data
        self.optimization_report = {}
        
        # Optimize docstrings for tools that have both failed and successful examples
        for tool in tools_list:
            try:
                tool_name = tool.get('name', '')
                tool_docstring = tool.get('description', '')
                tool_schema = tool.get('input_schema', {})
                failed_queries = []
                successful_queries = []
                
                # Analyze examples to find successful and failed cases for this tool
                for example in examples:
                    # Check if this tool was used in the example
                    if any([True for tool in example.get('tool_calls') if tool_name in tool.name]):
                        # Determine if the example was successful                        
                        if example.is_successful:
                            successful_queries.append(example.user_query)
                        else:
                            failed_queries.append(example.user_query)
                
                # Initialize report entry for this tool
                self.optimization_report[tool_name] = {
                    'original_docstring': tool_docstring,
                    'input_schema': tool_schema,
                    'successful_examples': successful_queries,
                    'failed_examples': failed_queries,
                    'optimized_docstring': None,
                    'optimization_attempted': False
                }
                
                # Only optimize if we have both successful and failed examples
                if len(successful_queries) > 0 or len(failed_queries) > 0:
                    print(f"Optimizing {tool_name}: {len(successful_queries)} successful, {len(failed_queries)} failed")
                    
                    # Show original docstring
                    print(f"\n--- ORIGINAL DOCSTRING for {tool_name} ---")
                    print(f"{tool_docstring}")
                    print("--- END ORIGINAL DOCSTRING ---")
                    
                    # Use docstring improver to generate better docstring
                    improved_docstring = self.docstring_improver(
                        tool_name=tool_name,
                        original_docstring=tool_docstring,
                        failed_examples=str(failed_queries),  # Limit to 3 examples
                        correct_examples=str(successful_queries)  # Limit to 3 examples
                    )
                    
                    # Store the optimized docstring
                    optimized_text = improved_docstring.improved_docstring
                    self.optimized_docstrings[tool_name] = optimized_text
                    self.optimization_report[tool_name]['optimized_docstring'] = optimized_text
                    self.optimization_report[tool_name]['optimization_attempted'] = True
                    
                    # Show optimized docstring
                    print(f"\n--- OPTIMIZED DOCSTRING for {tool_name} ---")
                    print(f"{optimized_text}")
                    print("--- END OPTIMIZED DOCSTRING ---\n")
                    
                    print(f"✓ Optimized docstring for tool: {tool_name}")
                else:
                    print(f"⊘ Skipping {tool_name}: no relevant examples found")
                
            except Exception as e:
                print(f"✗ Error optimizing docstring for {tool_name}: {e}")
                if tool_name in self.optimization_report:
                    self.optimization_report[tool_name]['optimization_attempted'] = True
                    self.optimization_report[tool_name]['error'] = str(e)
        
        print("=== Optimization Complete ===")
        print(f"Optimized docstrings for {len(self.optimized_docstrings)} tools")
        
        # Return the optimization report
        return self.optimization_report
        
        
    
    def train_with_bootstrapped_examples(self, 
                                       train_examples: List[dspy.Example], 
                                       num_bootstrapped: int = 5,
                                       max_bootstrapped_demos: int = 3):
        """
        Train the tool selector using bootstrapped examples.
        
        Args:
            train_examples: Examples to use for training
            num_bootstrapped: Number of bootstrapped example sets to generate
            max_bootstrapped_demos: Maximum number of examples to include in each bootstrapped set
            
        Returns:
            Optimized predictor module
        """
        # Use BootstrapFewShot optimizer
        optimizer = dspy.teleprompt.BootstrapFewShot(
            metric=lambda example, pred, trace=None: 1.0 if pred.selected_tool == example.correct_tool else 0.0,
            num_bootstrapped=num_bootstrapped,
            max_bootstrapped_demos=max_bootstrapped_demos
        )
        
        # Split train examples into train and validation
        train_size = int(0.8 * len(train_examples))
        optimizer_train = train_examples[:train_size]
        optimizer_val = train_examples[train_size:]
        
        # Compile the predictor
        optimized_selector = optimizer.compile(
            student=self.tool_selector,
            trainset=optimizer_train,
            valset=optimizer_val
        )
        
        # Update the selector with the optimized version
        self.tool_selector = optimized_selector
        
        return self
    