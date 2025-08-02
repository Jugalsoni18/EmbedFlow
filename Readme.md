# Text Embeddings with Neo4j Graph Database

A Python application that generates text embeddings using Google's Gemini API and stores them in a Neo4j graph database with semantic similarity relationships.

## Features

- **Text Embedding Generation**: Uses Gemini's models/embedding-001 to create 768-dimensional vector embeddings
- **Graph Database Storage**: Stores embeddings and metadata in Neo4j with proper indexing
- **Similarity Analysis**: Automatically creates relationships between semantically similar texts
- **Semantic Search**: Performs similarity queries with fallback to manual cosine similarity calculation
- **Environment Configuration**: Secure credential management using .env files

## 📋 Prerequisites

- Python 3.6+
- Neo4j database (local installation or Neo4j Aura)
- Google Gemini API key

## 🛠️ Installation

1. Clone or download the project files

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Neo4j**:
   
   **Local Installation**
   - Download and install Neo4j Desktop
   - Create a new database
   - Start the database service
   
   **Neo4j Aura (Cloud)**
   - Visit Neo4j Aura
   - Create a free account and database
   - Note your connection URI and credentials

4. **Create environment configuration**:
   
   Create a `.env` file in your project directory:
   ```text
   # Neo4j Configuration
   NEO4J_URI=neo4j://127.0.0.1:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_neo4j_password

   # Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## 🔑 API Setup

### Google Gemini API
1. Visit Google AI Studio
2. Sign in with your Google account
3. Create a new API key
4. Add the key to your `.env` file as `GEMINI_API_KEY`

## 📦 Dependencies

```text
neo4j>=5.0.0
google-generativeai>=0.3.0
numpy>=1.21.0
python-dotenv>=1.0.0
```

## 🎯 Usage

### Main Scripts

This project includes two main scripts with different purposes:

- **main.py**: Uses pre-trained data to demonstrate core functionality out of the box
- **main2.py**: Allows users to input their own data to test or train on custom scenarios

### Basic Usage

For demonstration with sample data:
```bash
python main.py
```

For custom data input:
```bash
python main2.py
```

The scripts will:
1. Connect to Neo4j and create vector indexes
2. Generate embeddings for texts (sample or custom)
3. Store texts with metadata in the graph database
4. Create similarity relationships between related texts
5. Perform semantic search queries
6. Display results and relationships

### Sample Output

```text
Starting Embedding Graph Database demo...
✅ Vector index created successfully

Storing sample texts with embeddings...
🗃️ Stored node ID: 4:abc123...
🗃️ Stored node ID: 4:def456...

Creating similarity relationships...
🔗 Created SIMILAR_TO with similarity: 0.902
🔗 Created SIMILAR_TO with similarity: 0.833

Similarity search results:
Query: 'programming languages and coding'

Score: 0.876 | Python is a high-level programming language...
Category: tech, Index: 0
Score: 0.834 | Machine learning is a subset of artificial...
Category: tech, Index: 1
```

## 🔍 Viewing the Graph

1. **Open Neo4j Browser**: Navigate to http://localhost:7474 where you can visualize and interact with your graph database
2. **Login**: Use your Neo4j credentials
3. **View all nodes**:
   ```cypher
   MATCH (n) RETURN n;
   ```
4. **View similarity relationships**:
   ```cypher
   MATCH (n:TextNode)-[r:SIMILAR_TO]->(m:TextNode)
   RETURN n, r, m;
   ```

## 🏗️ Architecture

### Core Components

- **EmbeddingGraphDB**: Main class handling database operations and embedding generation
- **Vector Indexing**: Neo4j vector indexes for efficient similarity search
- **Relationship Creation**: Automatic similarity relationship generation
- **Fallback Search**: Manual cosine similarity when vector indexes aren't available

### Data Model

```text
TextNode {
  text: String,
  embedding: List[Float],
  category: String,
  index: Integer,
  created_at: DateTime
}

SIMILAR_TO {
  similarity: Float
}
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NEO4J_URI | Neo4j connection URI | neo4j://127.0.0.1:7687 |
| NEO4J_USER | Neo4j username | neo4j |
| NEO4J_PASSWORD | Neo4j password | Required |
| GEMINI_API_KEY | Google Gemini API key | Required |

### Similarity Threshold

Modify the similarity threshold in the main function:
```python
db.create_similarity_relationship(node_ids[i], node_ids[j], threshold=0.7)
```

## 🔧 Customization

### Adding Custom Texts

Replace the `sample_texts` array with your own content:
```python
sample_texts = [
    "Your custom text here",
    "Another piece of content",
    # Add more texts...
]
```

### Custom Metadata

Add additional metadata fields:
```python
metadata = {
    "category": "your_category",
    "source": "document_name",
    "author": "author_name",
    "timestamp": datetime.now()
}
```

### Search Queries

Customize search queries:
```python
queries = [
    "your search term",
    "another search query",
    # Add more queries...
]
```

## 🐛 Troubleshooting

### Common Issues

**Connection Errors**
- Ensure Neo4j is running and accessible
- Verify credentials in `.env` file
- Check network connectivity

**API Errors**
- Verify Gemini API key is valid and active
- Check API quotas and rate limits
- Ensure internet connectivity

**Vector Index Issues**
- The script automatically falls back to manual similarity calculation
- For optimal performance, use Neo4j 5.11+ with vector index support

### Debug Mode

Enable verbose logging by adding debug prints:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Considerations

- **Batch Processing**: For large datasets, consider batching operations
- **Index Optimization**: Vector indexes significantly improve query performance
- **Memory Usage**: Large embedding collections may require memory optimization
- **API Limits**: Be aware of Gemini API rate limits for production use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🔗 Related Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Google Gemini API](https://ai.google.dev/)
- [Neo4j Vector Indexes](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)