import asyncio
from datetime import datetime
import pytz
from mcp.server.fastmcp import FastMCP

server = FastMCP(
    name="Sample Demo Server",
    instructions="A simple MCP server with time and summarization tools.",
)


@server.tool()
def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        return f"The current time in {timezone} is {current_time.strftime('%H:%M:%S')}."
    except pytz.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'."


@server.tool()
def summarize_text(text: str, length: int = 30) -> str:
    words = text.split()
    if len(words) <= length:
        return text
    summary = " ".join(words[:length])
    return summary + "..."


if __name__ == "__main__":
    asyncio.run(server.run_stdio_async())


