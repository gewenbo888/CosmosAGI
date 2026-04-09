"""Web search tool — uses DuckDuckGo for zero-config search."""

from __future__ import annotations

import logging
import urllib.parse
import urllib.request
import json

from cosmos_agi.tools.registry import register_tool

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo Instant Answer API.

    Falls back to a simple formatted query URL if the API doesn't return results.
    """
    try:
        encoded = urllib.parse.urlencode({"q": query, "format": "json", "no_html": "1"})
        url = f"https://api.duckduckgo.com/?{encoded}"

        req = urllib.request.Request(url, headers={"User-Agent": "CosmosAGI/0.1"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append(f"**{data.get('Heading', 'Result')}**\n{data['Abstract']}\nSource: {data.get('AbstractURL', '')}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text'][:200]}")

        if results:
            return "\n\n".join(results)

        return f"No instant results. Try searching: https://duckduckgo.com/?q={urllib.parse.quote(query)}"

    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return f"Search error: {e}. Try: https://duckduckgo.com/?q={urllib.parse.quote(query)}"


register_tool(
    "web_search",
    web_search,
    "Search the web for information using DuckDuckGo.",
    {"query": "str", "max_results": "int (default 5)"},
)
