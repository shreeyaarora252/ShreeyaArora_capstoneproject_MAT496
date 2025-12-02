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
    Search the web for interview preparation tips, best practices, and resources.
    Useful when candidate asks for study resources or latest interview trends.
    
    Args:
        query: Search query about interview topics (e.g., "best way to prepare for system design interviews")
    
    Returns:
        Summary of top search results or curated recommendations
    """
    
    # Curated resources by topic (fallback strategy)
    curated_resources = {
        "system design": """**📚 System Design Resources**

**1. System Design Primer (GitHub)**
Comprehensive guide covering scalability, load balancing, caching, databases, and more.
🔗 https://github.com/donnemartin/system-design-primer

**2. Grokking the System Design Interview**
Popular course covering real-world system design problems with detailed solutions.
🔗 https://www.educative.io/courses/grokking-the-system-design-interview

**3. ByteByteGo (Alex Xu)**
System Design Interview books (Volume 1 & 2) - industry favorites
🔗 https://bytebytego.com/

**4. YouTube Channels:**
• Gaurav Sen - Excellent system design videos
• Tech Dummies Narendra L - Clear explanations
• Exponent - Mock interviews

**5. Practice Platforms:**
• Pramp - Free mock interviews
• interviewing.io - Practice with engineers""",
        
        "dsa": """**📚 DSA (Data Structures & Algorithms) Resources**

**1. LeetCode**
Premium platform with 2000+ problems, company-specific questions
🔗 https://leetcode.com/

**2. NeetCode**
Curated list of 150 essential problems with video explanations
🔗 https://neetcode.io/

**3. AlgoExpert**
Structured curriculum with 160+ questions + video explanations
🔗 https://www.algoexpert.io/

**4. Books:**
• Cracking the Coding Interview (CTCI) - Gayle Laakmann McDowell
• Elements of Programming Interviews

**5. Free Resources:**
• GeeksforGeeks - Comprehensive tutorials
• HackerRank - Practice problems
• Codeforces - Competitive programming""",
        
        "behavioral": """**📚 Behavioral Interview Resources**

**1. STAR Method Framework**
Situation, Task, Action, Result - structure your answers effectively

**2. Common Questions to Prepare:**
• Tell me about yourself
• Biggest challenge you've faced
• Conflict with a teammate
• Leadership experience
• Failure and what you learned

**3. Resources:**
• Cracking the Coding Interview - Behavioral section
• Glassdoor - Company-specific questions
• Blind - Real interview experiences

**4. Mock Interview Platforms:**
• Pramp - Free peer practice
• interviewing.io - Practice with engineers
• Big Interview - AI feedback"""
    }
    
    # Try web search first with enhanced query
    try:
        enhanced_query = query
        if "interview" not in query.lower():
            enhanced_query = f"{query} interview preparation"
        
        print(f"🔍 Searching web: '{enhanced_query}'")
        
        ddgs = DDGS()
        results = ddgs.text(enhanced_query, max_results=5)
        
        if results:
            # Filter out irrelevant results
            filtered_results = []
            for result in results:
                title_lower = result['title'].lower()
                href = result['href'].lower()
                
                # Skip login/signup pages and unrelated domains
                skip_keywords = ["log in", "sign up", "create account", "login", "signin"]
                skip_domains = ["/login", "/signin", "/signup", "learn.lboro"]
                
                if any(word in title_lower for word in skip_keywords):
                    continue
                if any(domain in href for domain in skip_domains):
                    continue
                
                # Prefer educational/tech content
                if any(domain in href for domain in ["github", "educative", "leetcode", "medium", "dev.to", "youtube", "geeksforgeeks"]):
                    filtered_results.insert(0, result)  # Prioritize
                else:
                    filtered_results.append(result)
                
                if len(filtered_results) >= 3:
                    break
            
            if filtered_results:
                summary = f"**📚 Resources for: '{query}'**\n\n"
                for i, result in enumerate(filtered_results[:3], 1):
                    summary += f"**{i}. {result['title']}**\n"
                    summary += f"{result['body'][:180]}...\n"
                    summary += f"🔗 {result['href']}\n\n"
                return summary
    
    except Exception as e:
        print(f"⚠️ Web search failed: {e}")
    
    # Fallback to curated resources
    print("📖 Using curated resources")
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["system design", "scalability", "architecture"]):
        return curated_resources["system design"]
    elif any(word in query_lower for word in ["dsa", "algorithm", "data structure", "leetcode", "coding"]):
        return curated_resources["dsa"]
    elif any(word in query_lower for word in ["behavioral", "leadership", "conflict", "star method"]):
        return curated_resources["behavioral"]
    else:
        # General response
        return """**📚 General Interview Preparation Resources**

        **Technical Interviews:**
        • System Design Primer: https://github.com/donnemartin/system-design-primer
        • LeetCode: https://leetcode.com/
        • NeetCode: https://neetcode.io/

        **Practice Platforms:**
        • Pramp (free mock interviews)
        • interviewing.io
        • Blind (company reviews & experiences)

        **Books:**
        • Cracking the Coding Interview
        • System Design Interview (Alex Xu)
        • Designing Data-Intensive Applications

        Ask me for specific topic resources (e.g., "system design resources" or "DSA practice")!"""


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
