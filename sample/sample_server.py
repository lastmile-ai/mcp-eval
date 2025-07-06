import asyncio
from datetime import datetime
import pytz
from mcp.server.fastmcp import FastMCP

# Create an MCP server instance using FastMCP
server = FastMCP(
    title="Sample Demo Server",
    description="A simple MCP server with time and summarization tools.",
)


@server.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """
    Gets the current time in the specified timezone.
    For example, 'America/New_York' or 'Europe/London'.
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return f"The current time in {timezone} is {current_time.strftime('%H:%M:%S')}."
    except pytz.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'."


@server.tool()
def summarize_text(text: str, length: int = 30) -> str:
    """
    Summarizes the provided text to a specified number of words.
    This is a naive implementation that just truncates the text.
    """
    words = text.split()
    if len(words) <= length:
        return text
    summary = " ".join(words[:length])
    return summary + "..."


if __name__ == "__main__":
    # This allows the server to be run directly for testing
    # and for the mcp-eval framework to run it via `stdio` transport.
    asyncio.run(server.run_stdio_async())
