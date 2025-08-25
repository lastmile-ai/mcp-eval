import json
from typing import Optional, Dict, Any
from dataclasses import dataclass

from mcp_eval.evaluators.base import Evaluator, EvaluatorContext
from mcp_eval.evaluators.shared import EvaluatorResult

from mcp_eval.optimizer.core_trace_process import process_server_optimization, separate_traces_by_server, save_results

@dataclass 
class LLMJudgeSuccess(Evaluator):
    """Evaluator that uses an LLM to judge if user's task was successfully addressed based on trace information."""
    
    min_score: float = 0.8
    model: Optional[str] = None
    
    async def evaluate(self, ctx: EvaluatorContext) -> EvaluatorResult:
        """Evaluate if user's task was successfully addressed using trace information."""
        
        # Extract trace information from context
        trace_info_list = getattr(ctx, 'trace_info', [])
        
        if not trace_info_list:
            return EvaluatorResult(
                passed=False,
                expected="Task success evaluation",
                actual="No trace information available",
                score=0.0,
                error="No trace information found in context"
            )
        
        # Build chronological conversation from trace information
        conversation_parts = []
        conversation_parts.append("=== CONVERSATION TRACE (chronologically ordered) ===")
        conversation_parts.append("")
        
        for trace_info in trace_info_list:
            msg = trace_info.message_info
            timestamp_str = f"[{msg.timestamp:.2f}s]"
            
            # Format the message entry
            conversation_parts.append(f"{timestamp_str} {msg.sender}: {msg.content}")
            conversation_parts.append("")
        
        conversation_text = "\n".join(conversation_parts)
        
        # Build the evaluation prompt
        prompt_parts = [
            "You are evaluating whether a user's task was successfully addressed based on the conversation trace below.",
            "",
            "Your task is to:",
            "1. Identify what the user originally requested/wanted to accomplish",
            "2. Analyze the conversation flow and outcomes",
            "3. Determine if the user's task was successfully completed or addressed",
            "",
            conversation_text,
            "",
            "=== EVALUATION CRITERIA ===",
            "- Was the user's original request clearly identified and understood?",
            "- Did the agent/system take appropriate actions to address the request?", 
            "- Were any errors or issues encountered and properly resolved?",
            "- Did the conversation conclude with the user's task being completed or adequately addressed?",
            "- Consider both explicit confirmations and implicit evidence of success",
            "",
            "Provide your evaluation as a JSON object with the following structure:",
            "{",
            '  "score": <float between 0.0 and 1.0>,',
            '  "reasoning": "<comprehensive and detailed explanation analyzing of the task completion>",',
            '  "passed": <boolean indicating if the user\'s task was successfully addressed>,',
            '  "confidence": <float between 0.0 and 1.0 indicating your confidence>',
            '  "user_task_identified": "<brief description of what the user wanted to accomplish>",',
            '  "completion_evidence": "<key evidence that shows task completion or failure>"',
            "}",
            "",
            "Ensure your JSON is valid and complete."
        ]
        
        prompt = "\n".join(prompt_parts)
        
        try:
            from mcp_eval.llm_client import get_judge_client
            
            client = get_judge_client(self.model)
            response = await client.generate_str(prompt)
            
            # Extract and parse JSON response
            json_str = self._extract_json(response)
            judge_data = json.loads(json_str)
            
            # Validate basic structure
            required_fields = ["score", "reasoning", "passed", "confidence"]
            for field in required_fields:
                if field not in judge_data:
                    raise ValueError(f"Missing required field: {field}")
            
            score = float(judge_data["score"])
            passed = bool(judge_data["passed"]) and score >= self.min_score
            
            return EvaluatorResult(
                passed=passed,
                expected=f"Task successfully addressed (score >= {self.min_score})",
                actual=f"score = {score}, task_success = {judge_data['passed']}",
                score=score,
                details={
                    "reasoning": judge_data["reasoning"],
                    "confidence": float(judge_data.get("confidence", 1.0)),
                    "min_score": self.min_score,
                    "user_task_identified": judge_data.get("user_task_identified", ""),
                    "completion_evidence": judge_data.get("completion_evidence", ""),
                    "judge_response": response,
                    "trace_entries_count": len(trace_info_list)
                }
            )
            
        except Exception as e:
            return EvaluatorResult(
                passed=False,
                expected=f"Task successfully addressed (score >= {self.min_score})",
                actual="Failed to evaluate",
                score=0.0,
                error=str(e),
                details={
                    "reasoning": "Failed to parse LLM judge response",
                    "confidence": 0.0,
                    "min_score": self.min_score,
                    "judge_response": response if 'response' in locals() else "No response",
                    "trace_entries_count": len(trace_info_list)
                }
            )
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from response, handling various formats."""
        import re
        
        # Ensure response is a string to handle cases where LLM client returns numeric values
        if not isinstance(response, str):
            response = str(response)
        
        # Look for JSON between ``` markers
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Look for JSON object directly
        json_match = re.search(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # If no JSON found, try the whole response
        return response.strip()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_score": self.min_score,
            "model": self.model
        }