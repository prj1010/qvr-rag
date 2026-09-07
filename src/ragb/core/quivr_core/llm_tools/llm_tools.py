from typing import Dict, Any, Type, Union

from quivr_core.llm_tools.entity import ToolWrapper

from quivr_core.llm_tools.other_tools import (
    OtherTools,
)

try:
    from quivr_core.llm_tools.web_search_tools import WebSearchTools
except ImportError as exc:
    # Web search is an optional integration. Core RAG and non-network tools
    # must remain importable without langchain-community/Tavily installed.
    WebSearchTools = None
    _WEB_SEARCH_IMPORT_ERROR = exc
else:
    _WEB_SEARCH_IMPORT_ERROR = None

TOOLS_CATEGORIES = {OtherTools.name: OtherTools}
if WebSearchTools is not None:
    TOOLS_CATEGORIES[WebSearchTools.name] = WebSearchTools

# Register all ToolsList enums
TOOLS_LISTS = {tool.value: tool for tool in OtherTools.tools}
if WebSearchTools is not None:
    TOOLS_LISTS.update({tool.value: tool for tool in WebSearchTools.tools})


class LLMToolFactory:
    @staticmethod
    def create_tool(tool_name: str, config: Dict[str, Any]) -> Union[ToolWrapper, Type]:
        for category, tools_class in TOOLS_CATEGORIES.items():
            if tool_name in tools_class.tools:
                return tools_class.create_tool(tool_name, config)
            elif tool_name.lower() == category and tools_class.default_tool:
                return tools_class.create_tool(tools_class.default_tool, config)
        if (
            tool_name.lower() in {"tavily", "web search"}
            and _WEB_SEARCH_IMPORT_ERROR
        ):
            raise ImportError(
                "Tavily web search requires the optional langchain-community dependency."
            ) from _WEB_SEARCH_IMPORT_ERROR
        raise ValueError(f"Tool {tool_name} is not supported.")
