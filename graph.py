"""
LangGraph nodes for interview practice bot.
Each node handles a specific part of the interview flow.
"""

import json
import random
import os
from typing import Literal
from dotenv import load_dotenv
from schemas import QuestionEvaluation, TopicSelection, HintResponse, SessionSummary
from vector_store import search_questions, search_by_weak_topics, get_question_by_id

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from state import InterviewState
from prompts import (
    INTERVIEWER_SYSTEM_PROMPT,
    QUESTION_SELECTOR_PROMPT,
    ANSWER_EVALUATOR_PROMPT,
    HINT_PROVIDER_PROMPT,
    TOPIC_SELECTION_PROMPT,
    SESSION_SUMMARY_PROMPT
)
from tools import AVAILABLE_TOOLS



# Load environment variables FIRST
load_dotenv()

# Verify API key is loaded
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found! Please:\n"
        "1. Create .env file in project root\n"
        "2. Add: GROQ_API_KEY=your_key_here\n"
        "3. Make sure .env is in same directory as main.py"
    )

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=groq_api_key  # Explicitly pass the key
)
llm_with_structured_output = llm
# Load question bank
with open('data/questions.json', 'r') as f:
    QUESTION_BANK = json.load(f)


def topic_selector_node(state: InterviewState) -> InterviewState:
    """
    Initial node: Greets user and helps select interview topic.
    """
    # If topic already selected, skip
    if state.get("topic_category"):
        return state
    
    # Generate welcome message
    messages = [
        SystemMessage(content="You are a friendly technical interviewer."),
        HumanMessage(content=TOPIC_SELECTION_PROMPT)
    ]
    
    response = llm.invoke(messages)
    
    # Add to conversation history
    state["messages"].append(AIMessage(content=response.content))
    
    return state


def topic_parser_node(state: InterviewState) -> InterviewState:
    """
    Parses user's topic selection from their response.
    Sets topic_category and current_topic in state.
    """
    # Get last user message
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower()
            break
    
    if not last_user_msg:
        return state
    
    # Simple keyword matching (can be enhanced with LLM)
    if any(word in last_user_msg for word in ["dsa", "algorithm", "data structure", "array", "tree", "graph"]):
        state["topic_category"] = "dsa"
        # Extract specific topic if mentioned
        topics = ["arrays", "strings", "linked_lists", "trees", "dynamic_programming", "graphs"]
        for topic in topics:
            if topic.replace("_", " ") in last_user_msg or topic in last_user_msg:
                state["current_topic"] = topic
                break
        if not state.get("current_topic"):
            state["current_topic"] = "arrays"  # default
            
    elif any(word in last_user_msg for word in ["system design", "design", "scalability", "architecture"]):
        state["topic_category"] = "system_design"
        topics = ["scalability", "caching", "databases", "microservices"]
        for topic in topics:
            if topic in last_user_msg:
                state["current_topic"] = topic
                break
        if not state.get("current_topic"):
            state["current_topic"] = "scalability"
            
    elif any(word in last_user_msg for word in ["behavioral", "leadership", "conflict"]):
        state["topic_category"] = "behavioral"
        state["current_topic"] = "leadership"
    
    return state


def question_generator_node(state: InterviewState) -> InterviewState:
    """
    Selects and presents an appropriate question using RAG.
    Uses semantic search over vector store instead of random selection.
    """
    topic_category = state.get("topic_category", "dsa")
    current_topic = state.get("current_topic")
    difficulty = state.get("difficulty_level", "medium")
    weak_topics = state.get("weak_topics", [])
    questions_attempted = state.get("questions_attempted", 0)
    
    selected_question_data = None
    
    # Strategy 1: If weak topics exist, prioritize them (RAG-based)
    if weak_topics and questions_attempted > 0:
        print(f"🔍 Searching for questions on weak topics: {weak_topics}")
        results = search_by_weak_topics(weak_topics, difficulty=difficulty, n_results=3)
        if results:
            # Pick highest similarity question
            selected_question_id = results[0]['id']
            selected_question_data = get_question_by_id(selected_question_id)
            print(f"✓ Found question via RAG: {selected_question_id} (similarity: {results[0]['similarity_score']:.2f})")
    
    # Strategy 2: Semantic search based on topic and category
    if not selected_question_data and current_topic:
        query = f"{current_topic} {topic_category} interview question"
        print(f"🔍 Semantic search: '{query}'")
        results = search_questions(
            query=query,
            category=topic_category,
            difficulty=difficulty,
            n_results=3
        )
        if results:
            selected_question_id = results[0]['id']
            selected_question_data = get_question_by_id(selected_question_id)
            print(f"✓ Found question via semantic search: {selected_question_id}")
    
    # Strategy 3: Fallback to category-based search
    if not selected_question_data:
        results = search_questions(
            query=f"{topic_category} interview questions",
            category=topic_category,
            difficulty=difficulty,
            n_results=3
        )
        if results:
            selected_question_id = results[0]['id']
            selected_question_data = get_question_by_id(selected_question_id)
    
    # Strategy 4: Final fallback to JSON random (shouldn't happen)
    if not selected_question_data:
        print("⚠️ Falling back to random selection")
        available_questions = QUESTION_BANK.get(topic_category, [])
        if available_questions:
            import random
            selected_question_data = random.choice(available_questions)
    
    if not selected_question_data:
        state["messages"].append(AIMessage(content="Sorry, I couldn't find a suitable question. Please try a different topic."))
        return state
    
    # Update state
    state["current_question"] = selected_question_data["question"]
    state["current_question_id"] = selected_question_data["id"]
    state["current_topic"] = selected_question_data["topic"]
    state["hints_used"] = 0
    state["waiting_for_answer"] = True
    
    # Format question message
    question_msg = f"**Question ({selected_question_data['difficulty'].title()} - {selected_question_data['topic'].replace('_', ' ').title()}):**\n\n{selected_question_data['question']}\n\nTake your time. You can ask for hints if needed!"
    
    state["messages"].append(AIMessage(content=question_msg))
    
    return state


def answer_evaluator_node(state: InterviewState) -> InterviewState:
    """
    Evaluates user's answer and provides structured feedback.
    Uses Pydantic schema for consistent output.
    """
    question = state.get("current_question")
    user_answer = state.get("user_answer")
    
    if not user_answer:
        # Get last user message
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_answer = msg.content
                break
    
    # Get question context
    question_id = state.get("current_question_id")
    topic_category = state.get("topic_category", "dsa")
    questions = QUESTION_BANK.get(topic_category, [])
    question_data = next((q for q in questions if q["id"] == question_id), None)
    
    expected_concepts = ""
    if question_data and "expected_concepts" in question_data:
        expected_concepts = f"\n\nExpected concepts: {', '.join(question_data['expected_concepts'])}"
    
    # Prepare evaluation prompt
    eval_prompt = f"""Evaluate this technical interview answer.

Question: {question}
{expected_concepts}

Candidate's Answer: {user_answer}

Provide a thorough evaluation covering:
1. Score (0-100) based on correctness and approach
2. What they did well (specific strengths)
3. What was missing or incorrect (weaknesses)
4. Time/space complexity analysis (if DSA question)
5. Specific improvements they should make
6. Topics to study next
7. Overall encouraging feedback

Be constructive and specific. Acknowledge partial credit where appropriate."""
    
    # Use structured output with Pydantic schema
    structured_llm = llm.with_structured_output(QuestionEvaluation)
    
    try:
        evaluation = structured_llm.invoke([
            SystemMessage(content="You are an experienced technical interviewer providing detailed feedback."),
            HumanMessage(content=eval_prompt)
        ])
        
        # Format feedback message from structured output
        feedback_parts = [
            f"**Score: {evaluation.score}/100**\n",
            "**✅ Strengths:**"
        ]
        for strength in evaluation.strengths:
            feedback_parts.append(f"  • {strength}")
        
        if evaluation.weaknesses:
            feedback_parts.append("\n**⚠️ Areas to Improve:**")
            for weakness in evaluation.weaknesses:
                feedback_parts.append(f"  • {weakness}")
        
        if evaluation.time_complexity or evaluation.space_complexity:
            feedback_parts.append("\n**⏱️ Complexity Analysis:**")
            if evaluation.time_complexity:
                feedback_parts.append(f"  • Time: {evaluation.time_complexity}")
            if evaluation.space_complexity:
                feedback_parts.append(f"  • Space: {evaluation.space_complexity}")
        
        if evaluation.suggested_improvements:
            feedback_parts.append("\n**💡 Suggested Improvements:**")
            for improvement in evaluation.suggested_improvements:
                feedback_parts.append(f"  • {improvement}")
        
        if evaluation.follow_up_topics:
            feedback_parts.append("\n**📚 Topics to Study Next:**")
            for topic in evaluation.follow_up_topics:
                feedback_parts.append(f"  • {topic}")
        
        feedback_parts.append(f"\n{evaluation.overall_feedback}")
        
        feedback = "\n".join(feedback_parts)
        
        # Update state
        state["feedback"] = feedback
        state["answer_score"] = evaluation.score
        state["questions_attempted"] = state.get("questions_attempted", 0) + 1
        
        if evaluation.score >= 70:
            state["correct_answers"] = state.get("correct_answers", 0) + 1
        else:
            # Add follow-up topics to weak areas
            current_topic = state.get("current_topic")
            weak_topics = state.get("weak_topics", [])
            if current_topic and current_topic not in weak_topics:
                weak_topics.append(current_topic)
            # Add follow-up topics from evaluation
            for topic in evaluation.follow_up_topics:
                if topic not in weak_topics:
                    weak_topics.append(topic)
            state["weak_topics"] = weak_topics
        
        state["waiting_for_answer"] = False
        state["messages"].append(AIMessage(content=feedback))
        
        # Ask if they want another question
        follow_up = "\n\nWould you like to try another question? (yes/no)"
        state["messages"].append(AIMessage(content=follow_up))
        
    except Exception as e:
        # Fallback to non-structured if parsing fails
        print(f"Structured output failed: {e}")
        state["messages"].append(AIMessage(content="I had trouble evaluating that. Could you try rephrasing your answer?"))
    
    return state


def hint_provider_node(state: InterviewState) -> InterviewState:
    """
    Provides progressive hints when user is stuck.
    """
    hints_used = state.get("hints_used", 0)
    max_hints = state.get("max_hints", 3)
    question_id = state.get("current_question_id")
    
    if hints_used >= max_hints:
        msg = "You've used all available hints. Try your best to answer!"
        state["messages"].append(AIMessage(content=msg))
        return state
    
    # Find question in bank
    topic_category = state.get("topic_category", "dsa")
    questions = QUESTION_BANK.get(topic_category, [])
    question_data = next((q for q in questions if q["id"] == question_id), None)
    
    if question_data and "hints" in question_data:
        hint_text = question_data["hints"][hints_used]
        state["hints_used"] = hints_used + 1
        
        hint_msg = f"**Hint {hints_used + 1}/{max_hints}:** {hint_text}"
        state["messages"].append(AIMessage(content=hint_msg))
    else:
        # Fallback: generate hint with LLM
        hint_prompt = HINT_PROVIDER_PROMPT.format(
            question=state.get("current_question"),
            hints_used=hints_used,
            max_hints=max_hints,
            current_hint_number=hints_used + 1
        )
        
        messages = [
            SystemMessage(content="You are a helpful interviewer providing hints."),
            HumanMessage(content=hint_prompt)
        ]
        
        response = llm.invoke(messages)
        state["hints_used"] = hints_used + 1
        state["messages"].append(AIMessage(content=response.content))
    
    return state


def session_summary_node(state: InterviewState) -> InterviewState:
    """
    Generates structured final session summary.
    """
    questions_attempted = state.get("questions_attempted", 0)
    correct_answers = state.get("correct_answers", 0)
    weak_topics = state.get("weak_topics", [])
    
    summary_prompt = f"""Generate a session summary for this interview practice session.

Session Statistics:
- Questions attempted: {questions_attempted}
- Correct/Good answers: {correct_answers}
- Topics covered: {state.get("current_topic", "Various")}
- Identified weak areas: {", ".join(weak_topics) if weak_topics else "None"}

Provide:
1. Overall performance rating
2. Strongest areas demonstrated
3. Areas needing improvement
4. Specific recommended resources (courses, problem sets, articles)
5. Motivational closing message

Be specific and encouraging."""
    
    structured_llm = llm.with_structured_output(SessionSummary)
    
    try:
        summary = structured_llm.invoke([
            SystemMessage(content="You are a supportive technical interviewer providing final session feedback."),
            HumanMessage(content=summary_prompt)
        ])
        
        # Format summary message
        summary_parts = [
            "🎉 **Session Complete!**\n",
            f"**Performance Rating: {summary.performance_rating.replace('_', ' ').title()}**",
            f"Questions Attempted: {summary.questions_attempted}",
            f"Strong Answers: {summary.correct_answers}\n"
        ]
        
        if summary.strongest_areas:
            summary_parts.append("**💪 Your Strengths:**")
            for area in summary.strongest_areas:
                summary_parts.append(f"  ✓ {area}")
            summary_parts.append("")
        
        if summary.areas_to_improve:
            summary_parts.append("**📈 Focus Areas:**")
            for area in summary.areas_to_improve:
                summary_parts.append(f"  → {area}")
            summary_parts.append("")
        
        if summary.recommended_resources:
            summary_parts.append("**📚 Recommended Next Steps:**")
            for resource in summary.recommended_resources:
                summary_parts.append(f"  • {resource}")
            summary_parts.append("")
        
        summary_parts.append(summary.motivational_message)
        
        summary_text = "\n".join(summary_parts)
        
        state["messages"].append(AIMessage(content=summary_text))
        state["session_active"] = False
        
    except Exception as e:
        print(f"Structured summary failed: {e}")
        fallback = f"Great session! You attempted {questions_attempted} questions. Keep practicing!"
        state["messages"].append(AIMessage(content=fallback))
        state["session_active"] = False
    
    return state


# Routing functions
def should_continue_interview(state: InterviewState) -> Literal["continue", "end"]:
    """
    Decides whether to continue with more questions or end session.
    """
    # Check last user message
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower()
            break
    
    if last_user_msg and any(word in last_user_msg for word in ["no", "stop", "end", "quit", "exit"]):
        return "end"
    
    return "continue"


def route_user_input(state: InterviewState) -> Literal["answer", "hint", "new_question"]:
    """
    Routes based on user's intent: answering, asking hint, or requesting new question.
    """
    # Get last user message
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content.lower()
            break
    
    if not last_user_msg:
        return "answer"
    
    # Check for hint request
    if any(word in last_user_msg for word in ["hint", "help", "stuck", "clue"]):
        return "hint"
    
    # Check if waiting for answer
    if state.get("waiting_for_answer"):
        return "answer"
    
    # Default: new question
    return "new_question"

def tool_calling_node(state: InterviewState) -> InterviewState:
    """
    Handles tool calls when user asks for external information.
    Demonstrates MCP-like tool integration.
    """
    # Get last user message
    last_user_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user_msg = msg.content
            break
    
    if not last_user_msg:
        return state
    
    msg_lower = last_user_msg.lower()
    
    # Direct tool routing based on keywords
    tool_to_call = None
    tool_args = {}
    
    # Detect which tool to use
    if any(word in msg_lower for word in ["resource", "tip", "prepare", "best way", "recommend", "guide", "tutorial", "learn"]):
        tool_to_call = "web_search_interview_tips"
        # Clean up the query
        query = last_user_msg.replace("I want to", "").replace("i want to", "").strip()
        tool_args = {"query": query}
        
    elif any(word in msg_lower for word in ["explain", "what is", "tell me about", "definition of"]):
        # Extract concept name (simple heuristic)
        concept = last_user_msg
        for phrase in ["explain ", "what is ", "tell me about ", "definition of ", "i want to learn about ", "learn about "]:
            if phrase in msg_lower:
                idx = msg_lower.index(phrase) + len(phrase)
                concept = last_user_msg[idx:].strip().rstrip("?").rstrip(".")
                break
        tool_to_call = "get_algorithm_explanation"
        tool_args = {"algorithm_name": concept}
        
    elif "complexity" in msg_lower:
        tool_to_call = "complexity_analyzer"
        tool_args = {"code_description": last_user_msg}
    
    # Execute tool if identified
    if tool_to_call:
        print(f"🔧 Calling tool: {tool_to_call} with args: {tool_args}")
        
        tool_func = None
        for t in AVAILABLE_TOOLS:
            if t.name == tool_to_call:
                tool_func = t
                break
        
        if tool_func:
            try:
                result = tool_func.invoke(tool_args)
                print(f"✓ Tool executed successfully")
                state["messages"].append(AIMessage(content=result))
            except Exception as e:
                print(f"❌ Tool error: {e}")
                state["messages"].append(AIMessage(content=f"I encountered an error fetching that information. Let's continue with the interview."))
        else:
            state["messages"].append(AIMessage(content="I couldn't find the right tool for that request."))
    else:
        # No specific tool detected, use LLM to respond conversationally
        messages = [
            SystemMessage(content="You are a helpful technical interviewer."),
            HumanMessage(content=last_user_msg)
        ]
        response = llm.invoke(messages)
        state["messages"].append(AIMessage(content=response.content))
    
    return state


"""
Assemble all nodes into LangGraph workflow.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

def create_interview_graph():
    """
    Creates and compiles the interview practice bot graph.
    """
    # Initialize graph with our state
    workflow = StateGraph(InterviewState)
    
    # Add all nodes
    workflow.add_node("topic_selector", topic_selector_node)
    workflow.add_node("topic_parser", topic_parser_node)
    workflow.add_node("question_generator", question_generator_node)
    workflow.add_node("hint_provider", hint_provider_node)
    workflow.add_node("answer_evaluator", answer_evaluator_node)
    workflow.add_node("session_summary", session_summary_node)
    
    # Start: topic selection
    workflow.set_entry_point("topic_selector")
    
    # After topic selector, END - wait for user to respond
    workflow.add_edge("topic_selector", END)
    
    # When user responds after topic selection, we'll manually route to topic_parser
    # This is handled in main.py by checking if topic_category exists
    
    # After parsing topic, generate first question
    workflow.add_edge("topic_parser", "question_generator")
    
    # After question, END - wait for user response
    workflow.add_edge("question_generator", END)
    
    # After hint, END - wait for user to answer
    workflow.add_edge("hint_provider", END)
    
    # After evaluation, END - wait for continue/stop decision
    workflow.add_edge("answer_evaluator", END)
    
    # Session summary is terminal
    workflow.add_edge("session_summary", END)
    
    # Compile with memory for conversation persistence
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph
