# Tool Docstring Optimization Methodology

This document describes the methodology we used to optimize tool docstrings for improving tool call performance in agentic systems. Our approach leverages **[DSPy](https://github.com/stanfordnlp/dspy)** to refine docstrings using real interaction traces and task success evaluations.

---

## 1. Overview

Tool docstrings play a critical role in guiding large language models (LLMs) to select the correct tool and provide accurate arguments. However, many tool docstrings contain either too much irrelevant detail or are not structured in a way that is helpful for the model.

Our methodology automates the optimization of tool docstrings using:

* **DSPy signatures** for structured optimization
* **Interaction traces obtained from users' logs** from server logs
* **Success/failure evaluations** of tool calls
* **Refinement** of tool descriptions based on the top successful and failed examples

---

## 2. DSPy Signature for Docstring Optimization

We define a custom DSPy signature that takes successful and failed examples of tool calls as inputs and generates an improved docstring as output.

```python
class DocstringImprover(dspy.Signature):
    """Signature for improving tool docstrings based on successful and failed examples"""
    
    tool_name = dspy.InputField(desc="Name of the tool to optimize docstring for")
    tool_input_arguments = dspy.InputField(desc="List of input arguments for the tool with the type of the argument")
    original_docstring = dspy.InputField(desc="Original docstring of the tool")
    correct_examples = dspy.InputField(desc="List of examples where the tool was correctly selected")
    failed_examples = dspy.InputField(desc="List of examples where the tool selection failed")

    improved_docstring = dspy.OutputField(
        desc="""
        Your task is to create a concise and effective tool usage description 
        based on the tool documentation. Ensure the description only contains 
        the purposes of the tool without irrelevant information. 
        
        Example Format:
        /* Examples */
        {Tool Documentation}
        
        Tool usage description:
        {Tool_name} is a tool that can {General_Purposes}.
        This tool has {Number} multiple built-in functions:
        1. {Function_1} is to {Functionality_of_Function_1}
        2. {Function_2} is to ...
        
        /* Auto-generation Example */
        'Aviation Weather Center' is a tool which can provide official aviation weather data...
        """
    )
```

---

## 3. Data Collection from Traces

1. **Collect traces** from the logs of user communication with the MCP server.
2. **Transform traces into structured examples**, containing information such as:

```python
Example({
    'user_query': 'Use the fetch tool with arguments: url: https://example.com',
    'tool_calls': [
        {
            'name': 'fetch',
            'description': "<class 'mcp.types.CallToolResult'>",
            'arguments': {'url': 'https://example.com'}
        }
    ],
    'iteration_count': 4,
    'total_duration_ms': 4163.431168,
    'latency_ms': 947.4079608917236,
    'error_count': 0,
    'success_rate': 1.0,
    'is_successful': False,
    'score': 0.8,
    'task_success_evaluation': "The user's original request seems to involve fetching 
    information from the internet using a URL. The conversation reveals that the tool now 
    has the capability to access the internet, which is an important update since the agent 
    initially did not have this feature. ..."
})
```

---

## 3.1 Task Success Evaluation

Each trace is **evaluated by the LLM** using a structured evaluation prompt. The evaluation determines whether the tool call successfully completed the user’s task.

The prompt includes explicit success criteria and expects a **JSON output** with the following fields:

* `score`: Float between 0.0 and 1.0
* `reasoning`: Detailed explanation
* `passed`: Boolean (task successful or not)
* `confidence`: Confidence level of evaluation
* `user_task_identified`: Brief description of user’s goal
* `completion_evidence`: Evidence supporting success/failure

---

## 4. Selecting Examples for Optimization

For each tool call:

* Select the **top successful example** (highest score).
* Select the **top failed example** (lowest score).

These examples are passed into **DSPy** for optimizing the tool docstring.

---

## 5. Example: Fetch Tool Docstring Optimization

Below is an example of how the process improves tool docstrings.

**Original Docstring**

```
Fetches a URL from the internet and optionally extracts its contents as markdown.

Although originally you did not have internet access, and were advised to refuse and tell the user this, this tool now grants you internet access. Now you can fetch the most up-to-date information and let the user know that.
```

**Optimized Docstring (Generated via DSPy)**

```
The 'fetch' tool retrieves content from a given URL and returns it in markdown format. 
It provides internet access for obtaining the most up-to-date information. 
Use this tool when a user requests web content, online data, or resources from a specific URL.
```

---

## 6. Next Steps

* Extend the dataset of traces and docstring optimizations.
* Automate iterative re-optimization based on updated evaluation metrics.
* Release an evaluation dataset along with **mcp-eval** to benchmark docstring optimization methods.

---

✅ This methodology ensures docstrings are concise, task-relevant, and optimized based on **real-world usage patterns** rather than static documentation.

---