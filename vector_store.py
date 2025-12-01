"""
Vector store setup for semantic search over questions.
Uses ChromaDB for persistent storage and retrieval.
"""

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

# Initialize embedding model (lightweight, runs locally)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Collection names
COLLECTION_NAME = "interview_questions"


def initialize_vector_store():
    """
    Initialize vector database with questions from JSON.
    Embeds questions and stores them for semantic retrieval.
    """
    # Load questions
    with open('data/questions.json', 'r') as f:
        question_bank = json.load(f)
    
    # Get or create collection
    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
        print(f"✓ Loaded existing collection with {collection.count()} questions")
        return collection
    except:
        collection = chroma_client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "Technical interview questions with semantic search"}
        )
        print("✓ Created new collection")
    
    # Prepare documents for embedding
    documents = []
    metadatas = []
    ids = []
    
    for category, questions in question_bank.items():
        for q in questions:
            # Create rich text for embedding (question + hints + concepts)
            doc_text = f"{q['question']} "
            if 'hints' in q:
                doc_text += " ".join(q['hints']) + " "
            if 'expected_concepts' in q:
                doc_text += " ".join(q['expected_concepts'])
            
            documents.append(doc_text)
            metadatas.append({
                "id": q["id"],
                "topic": q["topic"],
                "difficulty": q["difficulty"],
                "category": category,
                "question": q["question"]
            })
            ids.append(q["id"])
    
    # Generate embeddings
    print(f"Embedding {len(documents)} questions...")
    embeddings = embedding_model.encode(documents, show_progress_bar=True)
    
    # Add to collection
    collection.add(
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"✓ Added {len(documents)} questions to vector store")
    return collection


def search_questions(
    query: str,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    topic: Optional[str] = None,
    n_results: int = 5
) -> List[Dict]:
    """
    Semantic search for relevant questions.
    
    Args:
        query: Natural language query (e.g., "questions about arrays and hashmaps")
        category: Filter by category (dsa, system_design, behavioral)
        difficulty: Filter by difficulty (easy, medium, hard)
        topic: Filter by specific topic
        n_results: Number of results to return
    
    Returns:
        List of question dictionaries with similarity scores
    """
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    # Build where filter with proper ChromaDB syntax
    where_conditions = []
    if category:
        where_conditions.append({"category": category})
    if difficulty:
        where_conditions.append({"difficulty": difficulty})
    if topic:
        where_conditions.append({"topic": topic})
    
    # Construct proper where filter
    where_filter = None
    if len(where_conditions) == 1:
        where_filter = where_conditions[0]
    elif len(where_conditions) > 1:
        where_filter = {"$and": where_conditions}
    
    # Embed query
    query_embedding = embedding_model.encode([query])[0]
    
    # Search
    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where_filter
        )
    except Exception as e:
        print(f"⚠️ Search filter error: {e}")
        # Fallback: search without filters
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
    
    # Format results
    questions = []
    if results and results['ids'] and len(results['ids'][0]) > 0:
        for i in range(len(results['ids'][0])):
            questions.append({
                "id": results['ids'][0][i],
                "question": results['metadatas'][0][i]['question'],
                "topic": results['metadatas'][0][i]['topic'],
                "difficulty": results['metadatas'][0][i]['difficulty'],
                "category": results['metadatas'][0][i]['category'],
                "similarity_score": 1 - results['distances'][0][i]  # Convert distance to similarity
            })
    
    return questions


def search_by_weak_topics(weak_topics: List[str], difficulty: str = "medium", n_results: int = 3) -> List[Dict]:
    """
    Retrieve questions targeting weak areas.
    Prioritizes topics user struggles with.
    """
    if not weak_topics:
        return []
    
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    # Create query from weak topics
    query = f"practice problems on {' '.join(weak_topics)}"
    query_embedding = embedding_model.encode([query])[0]
    
    # Search with topic filter using $or operator
    all_results = []
    
    # Build OR filter for weak topics
    topic_conditions = [{"topic": topic} for topic in weak_topics[:3]]
    
    where_filter = None
    if len(topic_conditions) == 1:
        where_filter = topic_conditions[0]
    elif len(topic_conditions) > 1:
        where_filter = {"$or": topic_conditions}
    
    # Add difficulty filter if needed
    if where_filter and difficulty:
        where_filter = {"$and": [where_filter, {"difficulty": difficulty}]}
    elif difficulty:
        where_filter = {"difficulty": difficulty}
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 2,  # Get more to deduplicate
            where=where_filter
        )
        
        # Format results
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                all_results.append({
                    "id": results['ids'][0][i],
                    "question": results['metadatas'][0][i]['question'],
                    "topic": results['metadatas'][0][i]['topic'],
                    "difficulty": results['metadatas'][0][i]['difficulty'],
                    "category": results['metadatas'][0][i]['category'],
                    "similarity_score": 1 - results['distances'][0][i]
                })
    except Exception as e:
        print(f"⚠️ Weak topic search error: {e}")
        # Fallback: simple search without filters
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        if results and results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                all_results.append({
                    "id": results['ids'][0][i],
                    "question": results['metadatas'][0][i]['question'],
                    "topic": results['metadatas'][0][i]['topic'],
                    "difficulty": results['metadatas'][0][i]['difficulty'],
                    "category": results['metadatas'][0][i]['category'],
                    "similarity_score": 1 - results['distances'][0][i]
                })
    
    # Remove duplicates and sort by similarity
    seen_ids = set()
    unique_results = []
    for r in all_results:
        if r['id'] not in seen_ids:
            seen_ids.add(r['id'])
            unique_results.append(r)
    
    unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return unique_results[:n_results]


def get_question_by_id(question_id: str) -> Optional[Dict]:
    """
    Retrieve full question details by ID.
    """
    with open('data/questions.json', 'r') as f:
        question_bank = json.load(f)
    
    for category, questions in question_bank.items():
        for q in questions:
            if q['id'] == question_id:
                return q
    return None
