from typing import TypedDict, List, Optional, Literal
from langgraph.graph import MessagesState

class InterviewState(MessagesState):
    """
    State schema for interview practice bot.
    Inherits MessagesState to track conversation history automatically.
    """
    # Topic tracking
    current_topic: Optional[str] = None  # e.g., "arrays", "system_design", "dynamic_programming"
    topic_category: Optional[Literal["dsa", "system_design", "behavioral"]] = None
    
    # Question management
    current_question: Optional[str] = None
    current_question_id: Optional[str] = None
    difficulty_level: str = "medium"  # easy, medium, hard
    
    # User interaction
    user_answer: Optional[str] = None
    hints_used: int = 0
    max_hints: int = 3
    
    # Evaluation & feedback
    answer_score: Optional[int] = None  # 0-100
    feedback: Optional[str] = None
    areas_to_improve: List[str] = []
    
    # Session tracking
    questions_attempted: int = 0
    correct_answers: int = 0
    weak_topics: List[str] = []  # Topics user struggles with
    
    # Flow control
    waiting_for_answer: bool = False
    session_active: bool = True
