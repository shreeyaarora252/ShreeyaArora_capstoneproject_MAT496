"""
Main entry point for Interview Practice Bot.
Provides CLI interface for interaction.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from graph import create_interview_graph
from vector_store import initialize_vector_store

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Suppress huggingface warning

# Load environment variables
load_dotenv()


def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 70 + "\n")


def print_ai_message(content: str):
    """Format and print AI messages."""
    print(f"🤖 Interviewer: {content}")


def get_user_input() -> str:
    """Get user input with prompt."""
    return input("\n👤 You: ").strip()


def determine_user_intent(user_input: str, state: dict) -> str:
    """
    Use LLM to determine user intent instead of keyword matching.
    Returns EXACTLY one of:
      - select_topic
      - request_resources
      - request_explanation
      - request_hint
      - answer_question
      - continue
      - end_session
    """
    router_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    context = f"""Analyze the user's intent based on the conversation state.

Current State:
- Topic selected: {state.get('topic_category', 'None')}
- Waiting for answer to question: {state.get('waiting_for_answer', False)}
- Has current question: {bool(state.get('current_question'))}

User's message: "{user_input}"

Possible intents (choose EXACTLY ONE, output only the label):
- select_topic: User is choosing what to practice (DSA, system design, behavioral, etc.)
- request_resources: User asking for how to prepare, what to study, practice resources, books, videos, websites, etc.
- request_explanation: User asking to explain a concept, algorithm, system design idea, complexity, etc.
- request_hint: User is stuck on the current question and wants a hint.
- answer_question: User is giving an answer to the current interview question (DSA / system design / behavioral).
- continue: User wants another interview question or to keep going.
- end_session: User wants to stop the session.

Your job:
- Look at the user's message and the state.
- Return ONLY one of these labels: select_topic, request_resources, request_explanation, request_hint, answer_question, continue, end_session.
"""

    try:
        intent_response = router_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an intent classifier. Respond with ONLY one of: "
                        "select_topic, request_resources, request_explanation, "
                        "request_hint, answer_question, continue, end_session."
                    )
                ),
                HumanMessage(content=context),
            ]
        )
        intent = intent_response.content.strip().lower()
        print(f"🧠 AI detected intent: {intent}")
        return intent
    except Exception as e:
        print(f"⚠️ Intent detection error: {e}")
        # Simple fallback if router fails
        if state.get("waiting_for_answer"):
            return "answer_question"
        if not state.get("topic_category"):
            return "select_topic"
        return "continue"


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
        "session_active": True,
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

            # Check for explicit exit commands
            if user_input.lower() in ["exit", "quit", "bye"]:
                print_separator()
                print("👋 Thanks for practicing! Good luck with your interviews!")
                print_separator()
                break

            # Add user message to state
            result["messages"].append(HumanMessage(content=user_input))

            # Use LLM to determine user intent
            intent = determine_user_intent(user_input, result)

            # Route based on LLM-determined intent
            if intent in ("request_resources", "request_explanation"):
                from graph import tool_calling_node

                result = tool_calling_node(result)

            elif intent == "select_topic":
                from graph import topic_parser_node, question_generator_node

                result = topic_parser_node(result)

                # Only generate question if topic was successfully parsed
                if result.get("topic_category"):
                    result = question_generator_node(result)
                else:
                    clarification_msg = (
                        "I didn't quite catch that. Please specify which category you'd like to practice:\n\n"
                        "• **DSA** (Data Structures & Algorithms)\n"
                        "• **System Design**\n"
                        "• **Behavioral**\n\n"
                        "Which would you prefer?"
                    )
                    result["messages"].append(AIMessage(content=clarification_msg))

            elif intent == "request_hint":
                from graph import hint_provider_node

                result = hint_provider_node(result)

            elif intent == "answer_question":
                from graph import answer_evaluator_node

                result = answer_evaluator_node(result)

            elif intent == "continue":
                from graph import question_generator_node

                result = question_generator_node(result)

            elif intent == "end_session":
                from graph import session_summary_node

                result = session_summary_node(result)

            else:
                # Fallback: general conversational response
                print("🤔 Intent unclear, responding conversationally...")
                conv_llm = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                )
                conv_response = conv_llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are a helpful technical interview coach. "
                                "Respond naturally to the user's message."
                            )
                        ),
                        *result["messages"][-3:],  # Last 3 messages for context
                    ]
                )
                result["messages"].append(AIMessage(content=conv_response.content))

            # Print new AI messages
            ai_messages_to_print = []
            for msg in reversed(result["messages"]):
                if hasattr(msg, "content"):
                    if (
                        msg.__class__.__name__ == "HumanMessage"
                        and msg.content == user_input
                    ):
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
    if result.get("weak_topics"):
        print(f"   Topics to Review: {', '.join(result['weak_topics'])}")
    print_separator()


if __name__ == "__main__":
    main()
