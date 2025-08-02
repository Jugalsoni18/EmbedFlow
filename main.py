import os
import numpy as np
from neo4j import GraphDatabase
from typing import List, Dict, Any
import json

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()


class EmbeddingGraphDB:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 api_key: str, use_gemini: bool = True):
        """
        Initializes the graph database connection and configures the embedding model.
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.api_key = api_key
        self.use_gemini = use_gemini

        if use_gemini:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.genai = genai
                self.embedding_model = "models/embedding-001"
            except ImportError:
                raise ImportError("Google GenerativeAI not installed. Run: pip install google-generativeai")
        else:
            raise ValueError("Only Gemini API is supported in this script currently.")

        self._create_vector_index()

    def _create_vector_index(self):
        """Create a vector index for embeddings in Neo4j."""
        with self.driver.session() as session:
            try:
                session.run("DROP INDEX text_embeddings IF EXISTS")
                session.run("""
                    CREATE VECTOR INDEX text_embeddings IF NOT EXISTS
                    FOR (n:TextNode) ON (n.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 768,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
                print("Vector index created successfully")
            except Exception as e:
                print(f"Could not create vector index: {e}")
                print("Proceeding without vector index. Similarity search will fallback to manual calculation.")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using Gemini API."""
        result = self.genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def store_text_with_embedding(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Store a text node with embedding and metadata."""
        embedding = self.generate_embedding(text)
        if metadata is None:
            metadata = {}
        
        with self.driver.session() as session:
            query = """
                CREATE (n:TextNode {
                    text: $text,
                    embedding: $embedding,
                    created_at: datetime()
                })
            """
            params = {"text": text, "embedding": embedding}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    query += f" SET n.{key} = ${key}"
                    params[key] = value
            query += " RETURN elementId(n) as node_id"

            result = session.run(query, **params)
            node_id = result.single()["node_id"]
            print(f"Stored node ID: {node_id}")
            return node_id

    def create_similarity_relationship(self, node_id1: str, node_id2: str, threshold: float = 0.8):
        """Create similarity relationship between two nodes if similarity exceeds threshold."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n1:TextNode), (n2:TextNode)
                WHERE elementId(n1) = $node_id1 AND elementId(n2) = $node_id2
                RETURN n1.embedding as emb1, n2.embedding as emb2
            """, node_id1=node_id1, node_id2=node_id2)

            record = result.single()
            if record:
                emb1 = np.array(record['emb1'])
                emb2 = np.array(record['emb2'])
                dot = np.dot(emb1, emb2)
                norm1 = np.linalg.norm(emb1)
                norm2 = np.linalg.norm(emb2)
                if norm1 > 0 and norm2 > 0:
                    similarity = dot / (norm1 * norm2)
                    if similarity > threshold:
                        session.run("""
                            MATCH (n1:TextNode), (n2:TextNode)
                            WHERE elementId(n1) = $node_id1 AND elementId(n2) = $node_id2
                            MERGE (n1)-[r:SIMILAR_TO {similarity: $similarity}]->(n2)
                        """, node_id1=node_id1, node_id2=node_id2, similarity=float(similarity))
                        print(f"Created SIMILAR_TO with similarity: {similarity:.3f}")

    def similarity_search(self, query_text: str, limit: int = 5) -> List[Dict]:
        """Perform similarity search with fallback to manual calculation."""
        query_embedding = self.generate_embedding(query_text)
        with self.driver.session() as session:
            try:
                results = session.run("""
                    CALL db.index.vector.queryNodes('text_embeddings', $limit, $query_embedding)
                    YIELD node, score
                    RETURN node.text as text,
                           node.category as category,
                           node.index as index,
                           node.created_at as created_at,
                           score,
                           elementId(node) as node_id
                    ORDER BY score DESC
                """, query_embedding=query_embedding, limit=limit)
                return [dict(r) for r in results]
            except Exception:
                print("Vector index query failed, falling back to manual search")
                return self._manual_similarity_search(query_embedding, limit)

    def _manual_similarity_search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """Fallback manual similarity search using cosine similarity."""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (n:TextNode)
                RETURN n.text as text, n.category as category, n.index as index,
                       n.created_at as created_at, n.embedding as embedding, elementId(n) as node_id
            """)
            query_emb = np.array(query_embedding)
            scored_nodes = []
            for record in results:
                node_emb = np.array(record['embedding'])
                dot = np.dot(query_emb, node_emb)
                norm_q = np.linalg.norm(query_emb)
                norm_n = np.linalg.norm(node_emb)
                if norm_q > 0 and norm_n > 0:
                    similarity = dot / (norm_q * norm_n)
                    scored_nodes.append({
                        'text': record['text'],
                        'category': record['category'],
                        'index': record['index'],
                        'created_at': record['created_at'],
                        'node_id': record['node_id'],
                        'score': float(similarity)
                    })
            scored_nodes.sort(key=lambda x: x['score'], reverse=True)
            return scored_nodes[:limit]

    def get_all_stored_data(self) -> List[Dict]:
        """Retrieve all stored TextNode nodes."""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (n:TextNode)
                RETURN n.text as text, n.category as category, n.index as index,
                       n.created_at as created_at, elementId(n) as node_id
                ORDER BY n.created_at DESC
            """)
            return [dict(r) for r in results]

    def get_similarity_relationships(self) -> List[Dict]:
        """Retrieve all similarity relationships."""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (n1:TextNode)-[r:SIMILAR_TO]->(n2:TextNode)
                RETURN n1.text as text1, n2.text as text2, r.similarity as similarity
                ORDER BY r.similarity DESC
            """)
            return [dict(r) for r in results]

    def close(self):
        """Close the database connection."""
        self.driver.close()


def main():
    """Main function to demonstrate the embedding graph database."""
    # Fetch configuration from environment variables
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")

    print("Starting Embedding Graph Database demo...")

    db = EmbeddingGraphDB(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GEMINI_API_KEY)

    sample_texts = [
        "Python is a high-level programming language with dynamic semantics.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn.",
        "Graph databases store data in nodes and relationships, making them ideal for connected data.",
        "Natural language processing helps computers understand and interpret human language.",
        "Deep learning uses neural networks with multiple layers to solve complex problems."
    ]

    print("\nStoring sample texts with embeddings...")
    node_ids = []
    for i, text in enumerate(sample_texts):
        metadata = {"category": "tech", "index": i}
        node_id = db.store_text_with_embedding(text, metadata)
        node_ids.append(node_id)

    print("\nCreating similarity relationships...")
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            db.create_similarity_relationship(node_ids[i], node_ids[j], threshold=0.7)

    print("\nStored data:")
    for item in db.get_all_stored_data():
        category = item.get('category', 'N/A')
        index = item.get('index', 'N/A')
        print(f"- {item['text'][:50]}... (ID: {item['node_id']}) [Category: {category}, Index: {index}]")

    print("\nSimilarity relationships:")
    for rel in db.get_similarity_relationships():
        print(f"- Similarity {rel['similarity']:.3f}: '{rel['text1'][:30]}...' ↔ '{rel['text2'][:30]}...'")

    print("\nSimilarity search results:")
    queries = [
        "programming languages and coding",
        "artificial intelligence and learning algorithms",
        "database systems and data storage"
    ]
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = db.similarity_search(query, limit=3)
        for i, res in enumerate(results, 1):
            print(f"  {i}. Score: {res['score']:.3f} | {res['text']}")
            category = res.get('category', 'N/A')
            index = res.get('index', 'N/A')
            if category != 'N/A' or index != 'N/A':
                print(f"     Category: {category}, Index: {index}")

    print("\nDemo completed successfully.")
    db.close()


if __name__ == '__main__':
    main()
