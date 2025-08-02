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
                        print(f"🔗 Created SIMILAR_TO with similarity: {similarity:.3f}")

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

def get_user_input():
    """Get texts from user input."""
    print("\nEnter your texts for embedding and analysis:")
    print("You can choose from the following options:")
    print("1. Use predefined sample texts (tech-related)")
    print("2. Enter a specific number of custom texts")
    print("3. Enter texts one by one until you press Enter on empty line")
    
    while True:
        choice = input("\nChoose option (1, 2, or 3): ").strip()
        
        if choice == "1":

            sample_texts = [
                "Python is a high-level programming language with dynamic semantics.",
                "Machine learning is a subset of artificial intelligence that enables computers to learn.",
                "Graph databases store data in nodes and relationships, making them ideal for connected data.",
                "Natural language processing helps computers understand and interpret human language.",
                "Deep learning uses neural networks with multiple layers to solve complex problems."
            ]
            print(f"Using {len(sample_texts)} predefined sample texts")
            return sample_texts
            
        elif choice == "2":
            # Enter specific number of texts
            try:
                num_texts = int(input("How many texts do you want to enter? "))
                if num_texts <= 0:
                    print(" Please enter a positive number")
                    continue
                    
                texts = []
                for i in range(num_texts):
                    while True:
                        text = input(f"Enter text #{i+1}: ").strip()
                        if text:
                            texts.append(text)
                            break
                        else:
                            print(" Text cannot be empty. Please try again.")
                
                print(f"Collected {len(texts)} texts from user input")
                return texts
                
            except ValueError:
                print(" Please enter a valid number")
                continue
                
        elif choice == "3":
            # Enter texts until empty line
            print("\nEnter your texts, one per line.")
            print("Press Enter on an empty line when you're finished.")
            
            texts = []
            while True:
                text = input(f"Text #{len(texts)+1} (or press Enter to finish): ").strip()
                if text == "":
                    if len(texts) == 0:
                        print(" You must enter at least one text. Please try again.")
                        continue
                    break
                texts.append(text)
            
            print(f"Collected {len(texts)} texts from user input")
            return texts
            
        else:
            print(" Invalid choice. Please enter 1, 2, or 3.")

def get_user_queries():
    """Get search queries from user input."""
    print("\n Now enter search queries to test similarity search:")
    print("You can:")
    print("1. Use predefined queries")
    print("2. Enter your own custom queries")
    
    while True:
        choice = input("\nChoose option (1 or 2): ").strip()
        
        if choice == "1":
            queries = [
                "programming languages and coding",
                "artificial intelligence and learning algorithms",
                "database systems and data storage"
            ]
            print(f"Using {len(queries)} predefined queries")
            return queries
            
        elif choice == "2":
            print("\nEnter your search queries, one per line.")
            print("Press Enter on an empty line when you're finished.")
            
            queries = []
            while True:
                query = input(f"Query #{len(queries)+1} (or press Enter to finish): ").strip()
                if query == "":
                    if len(queries) == 0:
                        print(" You must enter at least one query. Please try again.")
                        continue
                    break
                queries.append(query)
            
            print(f"Collected {len(queries)} queries from user input")
            return queries
            
        else:
            print(" Invalid choice. Please enter 1 or 2.")

def get_user_metadata():
    """Get metadata from user input."""
    print("\nDo you want to add custom metadata to your texts?")
    choice = input("Enter 'y' for yes or 'n' for no: ").strip().lower()
    
    if choice == 'y' or choice == 'yes':
        print("\nEnter metadata key-value pairs.")
        print("Examples: category=tech, source=blog, priority=high")
        print("Press Enter on empty line when finished.")
        
        metadata_template = {}
        while True:
            entry = input("Enter metadata (key=value) or press Enter to finish: ").strip()
            if entry == "":
                break
            
            if "=" in entry:
                key, value = entry.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                
                metadata_template[key] = value
                print(f"Added metadata: {key} = {value}")
            else:
                print("Invalid format. Please use 'key=value'")
        
        return metadata_template
    else:
        return {"category": "user_input"}

def main():
    """Main function with user input integration."""
    # Fetch configuration from environment variables
    NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set in environment variables.")

    print("Starting Interactive Embedding Graph Database...")
    
    # Initialize database connection
    db = EmbeddingGraphDB(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GEMINI_API_KEY)

    # Get user input for texts
    user_texts = get_user_input()
    
    # Get metadata template from user
    metadata_template = get_user_metadata()

    # Store texts with embeddings
    print(f"\nStoring {len(user_texts)} texts with embeddings...")
    node_ids = []
    for i, text in enumerate(user_texts):
        # Create metadata for this text
        metadata = metadata_template.copy()
        metadata["index"] = i
        
        node_id = db.store_text_with_embedding(text, metadata)
        node_ids.append(node_id)

    # Create similarity relationships
    print(f"\nCreating similarity relationships...")
    threshold = float(input("Enter similarity threshold (0.0-1.0, default 0.7): ") or "0.7")
    
    for i in range(len(node_ids)):
        for j in range(i+1, len(node_ids)):
            db.create_similarity_relationship(node_ids[i], node_ids[j], threshold=threshold)

    # Display stored data
    print(f"\nAll stored data:")
    for item in db.get_all_stored_data():
        category = item.get('category', 'N/A')
        index = item.get('index', 'N/A')
        print(f"- {item['text'][:50]}... (ID: {item['node_id']}) [Category: {category}, Index: {index}]")

    # Display similarity relationships
    print(f"\nSimilarity relationships:")
    relationships = db.get_similarity_relationships()
    if relationships:
        for rel in relationships:
            print(f"- Similarity {rel['similarity']:.3f}: '{rel['text1'][:30]}...' ↔ '{rel['text2'][:30]}...'")
    else:
        print("No similarity relationships found (all similarities below threshold)")

    # Get user queries and perform searches
    user_queries = get_user_queries()
    
    print(f"\nSimilarity search results:")
    search_limit = int(input("How many results per query (default 3)? ") or "3")
    
    for query in user_queries:
        print(f"\nQuery: '{query}'")
        results = db.similarity_search(query, limit=search_limit)
        
        if results:
            for i, res in enumerate(results, 1):
                print(f"  {i}. Score: {res['score']:.3f} | {res['text']}")
                category = res.get('category', 'N/A')
                index = res.get('index', 'N/A')
                if category != 'N/A' or index != 'N/A':
                    print(f"     Category: {category}, Index: {index}")
        else:
            print("  No results found")

    print(f"\nInteractive demo completed successfully!")
    db.close()

if __name__ == '__main__':
    main()
