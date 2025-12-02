"""
Main entry point for Interview Practice Bot.
Provides CLI interface for interaction.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from graph import create_interview_graph
from state import InterviewState
from vector_store import initialize_vector_store

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress huggingface warning

# Load environment variables
load_dotenv()

def print_separator():
    """Print a visual separator."""
    print("\n" + "="*70 + "\n")

def print_ai_message(content):
    """Format and print AI messages."""
    print(f"🤖 Interviewer: {content}")

def get_user_input():
    """Get user input with prompt."""
    return input("\n👤 You: ").strip()

def main():
    """
    Main interaction loop.
    """
    print_separator()
    print("🎯 Welcome to Technical Interview Practice Bot!")
    print("📚 Powered by LangGraph + Groq")
    print_separator()
    
    # Initialize vector store (RAG setup)
    print("🔧 Setting up RAG system...")
    initialize_vector_store()
    print_separator()

    # Create graph
    graph = create_interview_graph()
    
    # Initialize state with empty messages list
    state = {
        "messages": [],
        "difficulty_level": "medium",
        "questions_attempted": 0,
        "correct_answers": 0,
        "weak_topics": [],
        "hints_used": 0,
        "max_hints": 3,
        "waiting_for_answer": False,
        "session_active": True
    }
    
    # Configuration for graph execution
    config = {"configurable": {"thread_id": "interview_session_1"}}
    
    # Start conversation - invoke topic_selector
    result = graph.invoke(state, config)
    
    # Print initial greeting
    if result["messages"]:
        last_ai_msg = result["messages"][-1]
        print_ai_message(last_ai_msg.content)
    
    # Main interaction loop
    while result.get("session_active", True):
        try:
            # Get user input
            user_input = get_user_input()
            
            if not user_input:
                print("⚠️  Please provide input.")
                continue
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print_separator()
                print("👋 Thanks for practicing! Good luck with your interviews!")
                print_separator()
                break
            
            # Add user message to state
            result["messages"].append(HumanMessage(content=user_input))
            
            # Determine which node to invoke based on state
            
            # FIRST: Check for tool/resource requests (regardless of state)
            tool_keywords = [
                "resource", "tip", "how to prepare", "best way", "recommend",
                "explain", "what is", "tell me about", "complexity of",
                "help me understand", "guide", "tutorial", "learn"
            ]
            
            if any(keyword in user_input.lower() for keyword in tool_keywords):
                print("🔧 Detected tool request...")
                from graph import tool_calling_node
                result = tool_calling_node(result)
                
            elif not result.get("topic_category"):
                # User just selected topic, parse it
                from graph import topic_parser_node, question_generator_node
                result = topic_parser_node(result)
                result = question_generator_node(result)
                
            elif result.get("waiting_for_answer"):
                # User is responding to a question
                # Check if asking for hint
                if any(word in user_input.lower() for word in ["hint", "help", "stuck", "clue"]):
                    from graph import hint_provider_node
                    result = hint_provider_node(result)
                else:
                    # Evaluate answer
                    from graph import answer_evaluator_node
                    result = answer_evaluator_node(result)
                    
            else:
                # User responding to "want another question?"
                if any(word in user_input.lower() for word in ["yes", "yeah", "sure", "ok", "another"]):
                    from graph import question_generator_node
                    result = question_generator_node(result)
                elif any(word in user_input.lower() for word in ["no", "stop", "end", "quit"]):
                    from graph import session_summary_node
                    result = session_summary_node(result)
                else:
                    # Assume they want another question
                    from graph import question_generator_node
                    result = question_generator_node(result)
            
            # Print new AI messages
            ai_messages_to_print = []
            for msg in reversed(result["messages"]):
                if hasattr(msg, 'content'):
                    if msg.__class__.__name__ == "HumanMessage" and msg.content == user_input:
                        break
                    if msg.__class__.__name__ == "AIMessage":
                        ai_messages_to_print.insert(0, msg.content)
            
            for content in ai_messages_to_print:
                print_separator()
                print_ai_message(content)
            
            # Check if session ended
            if not result.get("session_active", True):
                print_separator()
                print("✅ Session completed! Great work!")
                print_separator()
                break
                
        except KeyboardInterrupt:
            print_separator()
            print("\n⚠️  Session interrupted. Goodbye!")
            print_separator()
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Please try again or type 'exit' to quit.")
    
    print("\n📊 Session Stats:")
    print(f"   Questions Attempted: {result.get('questions_attempted', 0)}")
    print(f"   Correct Answers: {result.get('correct_answers', 0)}")
    if result.get('weak_topics'):
        print(f"   Topics to Review: {', '.join(result['weak_topics'])}")
    print_separator()

if __name__ == "__main__":
    main()
