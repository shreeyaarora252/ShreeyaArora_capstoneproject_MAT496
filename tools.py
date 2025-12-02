"""
External tools for the interview bot.
Demonstrates tool calling / MCP integration.
"""

from langchain_core.tools import tool
from duckduckgo_search import DDGS
import wikipedia
from typing import Optional


@tool
def web_search_interview_tips(query: str) -> str:
    """
    Search for interview preparation tips and resources using AI-powered recommendations.
    Uses LLM to provide contextual, personalized advice based on the query.
    
    Args:
        query: Search query about interview topics
    
    Returns:
        AI-generated resource recommendations with explanations
    """
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage
    import os
    
    # Use the same LLM instance
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    
    # Prompt the LLM to act as an interview preparation expert
    system_prompt = """You are an expert technical interview coach with deep knowledge of:
- Data Structures & Algorithms preparation (LeetCode, competitive programming)
- System Design interviews (scalability, distributed systems)
- Behavioral interviews (STAR method, leadership stories)

When asked about interview preparation, provide:
1. Specific, actionable resources (courses, books, platforms, YouTube channels)
2. A structured study plan or approach
3. Key topics to focus on
4. Pro tips based on your expertise

Be concise but comprehensive. Include actual URLs when mentioning well-known resources like:
- LeetCode (leetcode.com)
- System Design Primer (github.com/donnemartin/system-design-primer)
- NeetCode (neetcode.io)
- Grokking courses (educative.io)
- ByteByteGo (bytebytego.com)

Format your response clearly with headings and bullet points."""

    user_prompt = f"A candidate is asking: '{query}'\n\nProvide comprehensive interview preparation guidance with specific resources and actionable advice."
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        return response.content
        
    except Exception as e:
        # Fallback if LLM call fails
        return f"""I encountered an error fetching personalized recommendations. Here are some reliable starting points:

**System Design:**
• System Design Primer: https://github.com/donnemartin/system-design-primer
• Grokking the System Design Interview

**DSA:**
• LeetCode: https://leetcode.com/
• NeetCode: https://neetcode.io/

**Behavioral:**
• Use STAR method for structuring answers
• Prepare 5-7 stories covering different scenarios

Error details: {str(e)}"""


@tool
def get_algorithm_explanation(algorithm_name: str) -> str:
    """
    Get detailed explanation of a data structure or algorithm from Wikipedia.
    Useful when candidate needs clarification on concepts.
    
    Args:
        algorithm_name: Name of algorithm/data structure (e.g., "binary search tree", "dynamic programming")
    
    Returns:
        Summary explanation from Wikipedia
    """
    try:
        # Search for the page
        search_results = wikipedia.search(algorithm_name, results=1)
        
        if not search_results:
            return f"No information found for '{algorithm_name}'. Try being more specific."
        
        # Get page summary
        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=4, auto_suggest=False)
        
        return f"**{page_title}**\n\n{summary}\n\nFor more details, see: https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Multiple topics found for '{algorithm_name}'. Be more specific. Options: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{algorithm_name}'."
    except Exception as e:
        return f"Error retrieving information: {str(e)}"


@tool
def complexity_analyzer(code_description: str) -> str:
    """
    Analyze time and space complexity of an algorithm based on its description.
    Provides Big-O notation explanation.
    
    Args:
        code_description: Description of the algorithm/code (e.g., "nested loop through array, then binary search")
    
    Returns:
        Complexity analysis with explanation
    """
    # This is a simple heuristic-based analyzer
    # In production, you'd use actual code parsing
    
    description_lower = code_description.lower()
    
    time_complexity = "O(n)"
    space_complexity = "O(1)"
    explanation = []
    
    # Time complexity heuristics
    if "nested" in description_lower and "loop" in description_lower:
        time_complexity = "O(n²)"
        explanation.append("Nested loops typically indicate O(n²) time complexity")
    elif "recursion" in description_lower or "recursive" in description_lower:
        if "divide" in description_lower or "binary" in description_lower:
            time_complexity = "O(n log n) or O(log n)"
            explanation.append("Divide-and-conquer recursion often yields O(n log n) or O(log n)")
        else:
            time_complexity = "O(2^n) or O(n!)"
            explanation.append("Recursive solutions can be exponential without memoization")
    elif "binary search" in description_lower:
        time_complexity = "O(log n)"
        explanation.append("Binary search operates in O(log n) time")
    elif "sort" in description_lower:
        time_complexity = "O(n log n)"
        explanation.append("Efficient sorting algorithms like merge/quick sort are O(n log n)")
    elif "loop" in description_lower or "iterate" in description_lower:
        time_complexity = "O(n)"
        explanation.append("Single loop through data is O(n)")
    
    # Space complexity heuristics
    if "recursion" in description_lower or "recursive" in description_lower:
        space_complexity = "O(n) or O(log n)"
        explanation.append("Recursion uses stack space")
    elif "hash" in description_lower or "map" in description_lower or "dictionary" in description_lower:
        space_complexity = "O(n)"
        explanation.append("Hash maps/dictionaries require O(n) space")
    elif "array" in description_lower or "list" in description_lower:
        if "new" in description_lower or "create" in description_lower:
            space_complexity = "O(n)"
            explanation.append("Creating new data structures requires O(n) space")
    
    result = f"**Complexity Analysis**\n\n"
    result += f"**Time Complexity:** {time_complexity}\n"
    result += f"**Space Complexity:** {space_complexity}\n\n"
    result += "**Reasoning:**\n"
    for exp in explanation:
        result += f"• {exp}\n"
    
    result += "\n*Note: This is a heuristic analysis. Exact complexity depends on implementation details.*"
    
    return result


# Export tools list
AVAILABLE_TOOLS = [
    web_search_interview_tips,
    get_algorithm_explanation,
    complexity_analyzer
]
