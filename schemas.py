"""
Pydantic schemas for structured LLM outputs.
Ensures consistent, parseable responses.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class QuestionEvaluation(BaseModel):
    """
    Structured evaluation of a candidate's answer.
    """
    score: int = Field(
        description="Score from 0-100 based on correctness and completeness",
        ge=0,
        le=100
    )
    
    strengths: List[str] = Field(
        description="What the candidate did well (specific points)",
        min_items=1
    )
    
    weaknesses: List[str] = Field(
        description="Areas where the answer was incorrect or incomplete",
        min_items=0
    )
    
    time_complexity: Optional[str] = Field(
        default=None,
        description="Time complexity analysis (for DSA questions only, e.g., 'O(n)', 'O(n log n)')"
    )
    
    space_complexity: Optional[str] = Field(
        default=None,
        description="Space complexity analysis (for DSA questions only)"
    )
    
    suggested_improvements: List[str] = Field(
        description="Specific actionable improvements",
        min_items=1
    )
    
    follow_up_topics: List[str] = Field(
        description="Topics the candidate should study next based on this answer",
        min_items=0,
        max_items=3
    )
    
    overall_feedback: str = Field(
        description="Encouraging summary feedback message"
    )


class TopicSelection(BaseModel):
    """
    Parsed topic selection from user input.
    """
    topic_category: Literal["dsa", "system_design", "behavioral"] = Field(
        description="The main category selected by the user"
    )
    
    specific_topic: Optional[str] = Field(
        default=None,
        description="Specific subtopic if mentioned (e.g., 'arrays', 'caching', 'leadership')"
    )
    
    difficulty_preference: Optional[Literal["easy", "medium", "hard"]] = Field(
        default="medium",
        description="Difficulty level if user specified one"
    )
    
    reasoning: str = Field(
        description="Brief explanation of why this topic was selected"
    )


class HintResponse(BaseModel):
    """
    Structured hint provision.
    """
    hint_text: str = Field(
        description="The actual hint to provide to the candidate"
    )
    
    hint_level: Literal["gentle", "moderate", "strong"] = Field(
        description="How much the hint reveals about the solution"
    )
    
    encouragement: str = Field(
        description="Brief encouraging message"
    )


class SessionSummary(BaseModel):
    """
    Structured session performance summary.
    """
    questions_attempted: int = Field(ge=0)
    correct_answers: int = Field(ge=0)
    
    performance_rating: Literal["excellent", "good", "needs_improvement"] = Field(
        description="Overall performance assessment"
    )
    
    strongest_areas: List[str] = Field(
        description="Topics/concepts where candidate performed well",
        max_items=3
    )
    
    areas_to_improve: List[str] = Field(
        description="Topics/concepts that need more practice",
        max_items=3
    )
    
    recommended_resources: List[str] = Field(
        description="Specific resources to study (e.g., 'Practice more tree traversal problems on LeetCode')",
        max_items=5
    )
    
    motivational_message: str = Field(
        description="Encouraging closing message"
    )
