from typing import List, Dict, Union
import numpy as np

# Attempt to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class ContextEmbedder:
    """
    Transforms user queries and conversation history into dense, continuous vectors.
    
    Why use continuous embeddings instead of discrete bucketing (e.g., categorizing 
    queries into 'math', 'coding', 'chat')?
    1. Expressiveness: Embeddings capture semantic nuance that hard buckets miss.
    2. Scalability: As query diversity grows, managing discrete classification buckets becomes brittle.
    3. Bandit Compatibility: Contextual bandits (like LinUCB and Neural Thompson Sampling) 
       naturally operate on continuous feature vectors to learn complex, non-linear decision boundaries.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', mock_mode: bool = False):
        """
        Initializes the embedder. 
        'all-MiniLM-L6-v2' is chosen as the default because it's a lightweight model 
        perfect for real-time routing logic where embedding latency must be extremely 
        low (single-digit milliseconds), otherwise the router itself becomes a bottleneck.
        """
        self.mock_mode = mock_mode
        self.embedding_dim = 384  # Output dimension for all-MiniLM-L6-v2
        
        if not self.mock_mode:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers is not installed. Use mock_mode=True or pip install sentence-transformers")
            # For a production router, this would be loaded on CPU/GPU depending on load.
            # Small models like MiniLM are often fast enough on CPU for low-concurrency routing.
            self.model = SentenceTransformer(model_name)
        
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Formats a multi-turn conversation into a single string for embedding.
        
        Why this approach?
        Instead of just embedding the final query, we include the context. 
        A query like "Can you explain it more simply?" is meaningless without the 
        history. We concatenate recent turns to capture the full context state.
        """
        if not messages:
            return ""
            
        formatted_turns = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            formatted_turns.append(f"{role}: {content}")
            
        # Join with newlines to provide clear separation between turns
        return "\n".join(formatted_turns)

    def get_embedding(self, query: Union[str, List[Dict[str, str]]]) -> np.ndarray:
        """
        Given a single string query or a list of message dicts (conversation history),
        returns a 1D numpy array representing the context vector.
        """
        if isinstance(query, list):
            text_to_embed = self._format_conversation(query)
        else:
            text_to_embed = query
            
        if not text_to_embed.strip():
            # Return a zero vector for empty queries to avoid downstream errors in the bandit
            return np.zeros(self.embedding_dim)
            
        if self.mock_mode:
            # Generate a deterministic pseudo-random vector based on the string hash.
            # This ensures that identical queries get the same mock embedding, allowing 
            # for basic testing of bandit convergence patterns even in mock mode.
            np.random.seed(abs(hash(text_to_embed)) % (2**32))
            vector = np.random.randn(self.embedding_dim)
            # Normalize it as sentence-transformers typically produce normalized embeddings
            return vector / np.linalg.norm(vector)
            
        # Real embedding generation
        embedding = self.model.encode(text_to_embed, convert_to_numpy=True, normalize_embeddings=True)
        return embedding
