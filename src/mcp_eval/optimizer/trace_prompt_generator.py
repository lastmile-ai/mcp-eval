"""Class for processing TraceInformation and generating prompts for LLM evaluation."""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from ..metrics import TraceInformation


@dataclass
class TraceEvent:
    """Represents a single trace event with extracted information."""
    timestamp: float
    message: str
    span_name: str
    sender: str
    recipient: str
    message_type: str
    metadata: Dict[str, Any]
    
    def __lt__(self, other):
        """Enable sorting by timestamp."""
        return self.timestamp < other.timestamp


class TracePromptGenerator:
    """Processes TraceInformation objects and generates prompts for LLM evaluation."""
    
    def __init__(self):
      
        pass
    
    def extract_trace_events(self, trace_info_list: List[TraceInformation]) -> List[TraceEvent]:
        """Extract and sort trace events from TraceInformation objects.
        
        Args:
            trace_info_list: List of TraceInformation objects
            
        Returns:
            List of TraceEvent objects sorted by timestamp
        """
        events = []
        
        for trace_info in trace_info_list:
            msg_info = trace_info.message_info
            
            # Extract span name from original span or metadata
            span_name = "unknown"
            if trace_info.original_span:
                span_name = trace_info.original_span.name
            elif "span_name" in msg_info.metadata:
                span_name = msg_info.metadata["span_name"]
            
            # Create TraceEvent
            event = TraceEvent(
                timestamp=msg_info.timestamp,
                message=msg_info.content,
                span_name=span_name,
                sender=msg_info.sender,
                recipient=msg_info.recipient or "unknown",
                message_type=msg_info.message_type,
                metadata=msg_info.metadata
            )
            events.append(event)
        
        # Sort by timestamp
        events.sort()
        
        return events
    
    def generate_conversation_prompt(self, events: List[TraceEvent]) -> str:
        """Generate a conversational prompt from sorted trace events.
        
        Args:
            events: List of TraceEvent objects sorted by timestamp
            
        Returns:
            Formatted conversation prompt string
        """
        conversation_parts = []
        conversation_parts.append("## Conversation Timeline\n")
        
        for i, event in enumerate(events):
            # Format timestamp
            timestamp_str = f"[{event.timestamp:.3f}]"
            
            # Create event description
            event_desc = f"**{timestamp_str} {event.sender} ({event.message_type})**"
            if event.recipient != "unknown":
                event_desc += f" → {event.recipient}"
            
            # Add span context if meaningful
            if event.span_name and event.span_name != "unknown":
                event_desc += f" [{event.span_name}]"
            
            conversation_parts.append(event_desc)
            
            # Add message content
            if event.message:
                # Truncate very long messages
                message = event.message
                if len(message) > 500:
                    # summarize the message using llm
                    # create a class that includes an llm instance - open ai model - gpt-4o-mini 
                    # if the legnth of the message is greater than 500 characters 
                    # summarize the message using the llm and then append it to the conversation_parts
                    message = message[:500] + "... [truncated]"
                
                conversation_parts.append(f"{message}\n")
            
            # Add relevant metadata if present
            metadata_info = self._extract_relevant_metadata(event.metadata)
            if metadata_info:
                conversation_parts.append(f"*Metadata: {metadata_info}*\n")
            
            # Add separator between events
            if i < len(events) - 1:
                conversation_parts.append("---\n")
        
        return "\n".join(conversation_parts)
    
    def generate_evaluation_prompt(self, events: List[TraceEvent]) -> Dict[str, str]:
        """Generate structured prompts for LLM evaluation.
        """
        conversation = self.generate_conversation_prompt(events)
        
        return {
            "conversation": conversation,
        }
    
    def _extract_relevant_metadata(self, metadata: Dict[str, Any]) -> str:
        """Extract relevant metadata for display."""
        relevant_items = []
        
        # Look for error information
        if metadata.get("error.message"):
            relevant_items.append(f"Error: {metadata['error.message']}")
        
        # Look for tool information
        if metadata.get("mcp.tool.name"):
            relevant_items.append(f"Tool: {metadata['mcp.tool.name']}")
        
        # Look for model information
        if metadata.get("gen_ai.request.model"):
            relevant_items.append(f"Model: {metadata['gen_ai.request.model']}")
        
        # Look for token usage
        if metadata.get("gen_ai.usage.output_tokens"):
            relevant_items.append(f"Output tokens: {metadata['gen_ai.usage.output_tokens']}")
        
        return ", ".join(relevant_items)
    
    def _extract_user_query(self, events: List[TraceEvent]) -> str:
        """Extract the initial user query from events."""
        for event in events:
            if (event.message_type == "user_task" or 
                event.sender == "user" or
                "user" in event.message_type.lower()):
                return event.message
        
        # Fallback: return first meaningful message
        for event in events:
            if event.message and len(event.message.strip()) > 10:
                return event.message
        
        return "No user query found"
    
    def _extract_tool_usage_summary(self, events: List[TraceEvent]) -> str:
        """Extract tool usage summary from events."""
        tool_events = [e for e in events if e.message_type == "tool_interaction" or "tool" in e.span_name.lower()]
        
        if not tool_events:
            return "No tools were used"
        
        tool_summary = []
        tool_summary.append(f"Total tool interactions: {len(tool_events)}")
        
        # Group by tool name
        tool_counts = {}
        for event in tool_events:
            tool_name = event.metadata.get("mcp.tool.name", "unknown_tool")
            tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        
        tool_summary.append("Tools used:")
        for tool, count in tool_counts.items():
            tool_summary.append(f"  - {tool}: {count} times")
        
        return "\n".join(tool_summary)
    
    def _extract_final_outcomes(self, events: List[TraceEvent]) -> str:
        """Extract final outcomes and responses from events."""
        # Look for the last few meaningful events
        final_events = events[-3:] if len(events) > 3 else events
        
        outcomes = []
        for event in final_events:
            if event.message and len(event.message.strip()) > 20:
                outcome = f"[{event.sender}] {event.message[:200]}"
                if len(event.message) > 200:
                    outcome += "... [truncated]"
                outcomes.append(outcome)
        
        return "\n".join(outcomes) if outcomes else "No clear final outcomes identified"