"""
============================================================================
LESSON: Data Splitting and Data Chunking for Vector Databases in Python
============================================================================

Today's Agenda:
1. Understanding Data Splitting (train/test, etc.)
2. Data Chunking for Vector Databases
3. Working with Vector Databases (ChromaDB)
4. Complete End-to-End Example

Let's begin!
============================================================================
"""

# ============================================================================
# PART 1: DATA SPLITTING
# ============================================================================

print("\n" + "="*70)
print("PART 1: DATA SPLITTING")
print("="*70)

"""
Data splitting is the process of dividing your dataset into subsets:
- Training set: Used to train models
- Validation set: Used to tune hyperparameters
- Test set: Used for final evaluation

For vector databases, we also split:
- Indexing data: Documents we want to search over
- Query data: Questions/queries we want to find answers for
"""

# Example 1: Basic Train/Test Split
print("\n--- Example 1: Basic Train/Test Split ---")

from sklearn.model_selection import train_test_split
import numpy as np

# Sample data: documents and labels
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Vector databases store embeddings",
    "Natural language processing analyzes text",
    "Deep learning uses neural networks",
    "Data science combines statistics and coding",
    "Artificial intelligence mimics human thinking",
    "Computer vision processes images",
    "Robotics combines hardware and software",
    "Quantum computing uses quantum mechanics"
]

labels = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  # Example labels

# Split data: 80% train, 20% test
train_docs, test_docs, train_labels, test_labels = train_test_split(
    documents, 
    labels, 
    test_size=0.2, 
    random_state=42,
    stratify=labels  # Maintains class distribution
)

print(f"Total documents: {len(documents)}")
print(f"Training set: {len(train_docs)} documents")
print(f"Test set: {len(test_docs)} documents")
print(f"\nTraining documents: {train_docs[:3]}")
print(f"Test documents: {test_docs}")

# ============================================================================
# PART 2: DATA CHUNKING FOR VECTOR DATABASES
# ============================================================================

print("\n" + "="*70)
print("PART 2: DATA CHUNKING FOR VECTOR DATABASES")
print("="*70)

"""
Why chunk data?
- Vector databases have token/embedding limits
- Smaller chunks improve search precision
- Better context matching
- Manageable embedding dimensions

Common chunking strategies:
1. Character-based chunking
2. Token-based chunking  
3. Sentence-based chunking
4. Semantic chunking (by meaning)
"""

# Example 2: Simple Character-Based Chunking
print("\n--- Example 2: Simple Character-Based Chunking ---")

def simple_chunk_text(text, chunk_size=100, overlap=20):
    """
    Split text into chunks with overlap.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap to maintain context
    
    return chunks

# Sample long text
long_text = """
Vector databases are specialized databases designed to store and query high-dimensional 
vectors (embeddings). They are essential for applications like semantic search, 
recommendation systems, and AI-powered applications. Unlike traditional databases that 
store structured data, vector databases excel at similarity searches, allowing you to 
find items that are semantically similar to a query even if they don't contain exact 
keyword matches. This makes them perfect for natural language processing, computer 
vision, and other machine learning applications where understanding meaning and context 
is crucial.
"""

chunks = simple_chunk_text(long_text, chunk_size=150, overlap=30)
print(f"Original text length: {len(long_text)} characters")
print(f"Number of chunks: {len(chunks)}")
print("\nChunks:")
for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars):")
    print(chunk[:100] + "..." if len(chunk) > 100 else chunk)

# Example 3: Sentence-Based Chunking (Better for NLP)
print("\n--- Example 3: Sentence-Based Chunking ---")

import re

def sentence_chunk_text(text, sentences_per_chunk=3):
    """
    Split text into chunks by sentences.
    
    Args:
        text: Input text to chunk
        sentences_per_chunk: Number of sentences per chunk
    """
    # Split text into sentences (simple regex-based)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i+sentences_per_chunk])
        chunks.append(chunk)
    
    return chunks

sentence_chunks = sentence_chunk_text(long_text, sentences_per_chunk=2)
print(f"Number of sentence-based chunks: {len(sentence_chunks)}")
for i, chunk in enumerate(sentence_chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)

# ============================================================================
# PART 3: ADVANCED CHUNKING WITH LANGCHAIN
# ============================================================================

print("\n" + "="*70)
print("PART 3: ADVANCED CHUNKING WITH LANGCHAIN")
print("="*70)

print("\nNote: To use LangChain chunking, install with: pip install langchain")
print("We'll demonstrate the concept here:\n")

def token_based_chunk(text, chunk_size=100, chunk_overlap=20):
    """
    Token-based chunking (simplified version).
    In production, use LangChain's TokenTextSplitter or tiktoken.
    """
    words = text.split()
    chunks = []
    
    i = 0
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunks.append(' '.join(chunk_words))
        i += chunk_size - chunk_overlap
    
    return chunks

token_chunks = token_based_chunk(long_text, chunk_size=50, chunk_overlap=10)
print(f"Token-based chunks: {len(token_chunks)}")
print(f"First chunk: {token_chunks[0]}")

# ============================================================================
# PART 4: VECTOR DATABASE INTEGRATION
# ============================================================================

print("\n" + "="*70)
print("PART 4: VECTOR DATABASE WITH CHROMADB")
print("="*70)

print("\nNote: Install ChromaDB with: pip install chromadb sentence-transformers")

def demonstrate_vector_db_workflow():
    """
    Demonstrates the complete workflow:
    1. Load documents
    2. Chunk documents
    3. Generate embeddings
    4. Store in vector database
    5. Query the database
    """
    
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        print("\n--- Setting up Vector Database ---")
        
        # Initialize ChromaDB client
        client = chromadb.Client()
        
        # Create or get a collection
        collection = client.create_collection(
            name="tutorial_collection",
            metadata={"description": "Tutorial on data chunking"}
        )
        
        print("✓ Created ChromaDB collection")
        
        # Load embedding model
        print("\n--- Loading Embedding Model ---")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model
        print("✓ Loaded embedding model")
        
        # Prepare documents
        print("\n--- Preparing Documents ---")
        sample_documents = [
            "Python is a versatile programming language used for data science.",
            "Machine learning algorithms learn patterns from data.",
            "Vector databases enable semantic search capabilities.",
            "Natural language processing helps computers understand text.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        # Chunk the documents
        all_chunks = []
        chunk_metadata = []
        
        for doc_id, doc in enumerate(sample_documents):
            chunks = sentence_chunk_text(doc, sentences_per_chunk=1)
            all_chunks.extend(chunks)
            chunk_metadata.extend([
                {"source_doc": doc_id, "chunk_index": i} 
                for i in range(len(chunks))
            ])
        
        print(f"✓ Created {len(all_chunks)} chunks from {len(sample_documents)} documents")
        
        # Generate embeddings
        print("\n--- Generating Embeddings ---")
        embeddings = model.encode(all_chunks).tolist()
        print(f"✓ Generated {len(embeddings)} embeddings (dimension: {len(embeddings[0])})")
        
        # Add to vector database
        print("\n--- Storing in Vector Database ---")
        collection.add(
            embeddings=embeddings,
            documents=all_chunks,
            ids=[f"chunk_{i}" for i in range(len(all_chunks))],
            metadatas=chunk_metadata
        )
        print("✓ Stored all chunks in vector database")
        
        # Query the database
        print("\n--- Querying Vector Database ---")
        query = "What is machine learning?"
        query_embedding = model.encode([query]).tolist()[0]
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        print(f"\nQuery: '{query}'")
        print("\nTop 3 Similar Results:")
        for i, (doc, metadata) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0]
        ), 1):
            print(f"\n{i}. {doc}")
            print(f"   Source document: {metadata['source_doc']}")
        
        print("\n✓ Successfully queried vector database!")
        
        return True
        
    except ImportError as e:
        print(f"\n⚠ Missing dependencies. Install with:")
        print("  pip install chromadb sentence-transformers")
        print(f"\nError: {e}")
        return False

# Run the demonstration
demonstrate_vector_db_workflow()

# ============================================================================
# PART 5: COMPLETE WORKFLOW EXAMPLE
# ============================================================================

print("\n" + "="*70)
print("PART 5: COMPLETE WORKFLOW - SPLIT, CHUNK, AND INDEX")
print("="*70)

def complete_workflow_example():
    """
    Complete workflow demonstrating:
    1. Splitting data into train/test
    2. Chunking documents
    3. Preparing for vector database
    """
    
    # Step 1: Load your dataset
    print("\n--- Step 1: Load Dataset ---")
    all_documents = documents * 2  # Simulate larger dataset
    print(f"Total documents: {len(all_documents)}")
    
    # Step 2: Split into indexing and query sets
    print("\n--- Step 2: Split Data ---")
    indexing_docs, query_docs = train_test_split(
        all_documents,
        test_size=0.3,
        random_state=42
    )
    print(f"Documents for indexing: {len(indexing_docs)}")
    print(f"Documents for queries: {len(query_docs)}")
    
    # Step 3: Chunk the indexing documents
    print("\n--- Step 3: Chunk Documents ---")
    chunked_docs = []
    doc_metadata = []
    
    for doc_id, doc in enumerate(indexing_docs):
        chunks = sentence_chunk_text(doc, sentences_per_chunk=1)
        chunked_docs.extend(chunks)
        doc_metadata.extend([
            {"original_doc_id": doc_id, "chunk_num": i}
            for i in range(len(chunks))
        ])
    
    print(f"Created {len(chunked_docs)} chunks from {len(indexing_docs)} documents")
    print(f"Average chunks per document: {len(chunked_docs)/len(indexing_docs):.2f}")
    
    # Step 4: Prepare for vector database
    print("\n--- Step 4: Prepare for Vector DB ---")
    print("Ready to:")
    print("  1. Generate embeddings for chunks")
    print("  2. Store in vector database")
    print("  3. Query using query_docs")
    
    return chunked_docs, query_docs, doc_metadata

complete_workflow_example()

# ============================================================================
# BEST PRACTICES AND TIPS
# ============================================================================

print("\n" + "="*70)
print("BEST PRACTICES AND TIPS")
print("="*70)

tips = """
1. CHUNKING STRATEGIES:
   - Character-based: Fast but may break sentences
   - Sentence-based: Better for natural language, maintains context
   - Token-based: Good balance, respects word boundaries
   - Semantic chunking: Best quality but more complex

2. CHUNK SIZE:
   - Too small: May lose context, too many chunks
   - Too large: May include irrelevant information
   - Recommended: 100-500 tokens for most use cases
   - Overlap: 10-20% helps maintain context between chunks

3. DATA SPLITTING:
   - Use stratified splits for imbalanced data
   - Keep test set completely separate (don't peek!)
   - Typical splits: 80/10/10 or 70/15/15 (train/val/test)

4. VECTOR DATABASES:
   - ChromaDB: Easy to use, good for prototyping
   - Pinecone: Managed service, production-ready
   - FAISS: Facebook's library, very fast
   - Weaviate: GraphQL interface, great features

5. EMBEDDING MODELS:
   - all-MiniLM-L6-v2: Fast, good balance (384 dim)
   - all-mpnet-base-v2: Better quality (768 dim)
   - BGE models: State-of-the-art for many tasks
"""

print(tips)

print("\n" + "="*70)
print("LESSON COMPLETE! 🎓")
print("="*70)
print("\nNext steps:")
print("1. Install dependencies: pip install chromadb sentence-transformers")
print("2. Try the examples with your own data")
print("3. Experiment with different chunk sizes and strategies")
print("4. Build your own vector search application!")
print("\n")

