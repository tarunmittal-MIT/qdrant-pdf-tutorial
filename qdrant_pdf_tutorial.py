"""
============================================================================
LESSON: Data Splitting and Chunking with Qdrant Vector Database
Using Azure AZ-104 Learning PDF
============================================================================

Today's Agenda:
1. Download and extract text from Azure AZ-104 PDF
2. Data splitting strategies
3. Advanced chunking techniques for PDFs
4. Storing chunks in Qdrant vector database
5. Querying and retrieving relevant information

Let's begin!
============================================================================
"""

import os
import re
import uuid
from pathlib import Path

# ============================================================================
# PART 1: SETUP AND DEPENDENCIES
# ============================================================================

print("\n" + "="*70)
print("PART 1: SETUP AND INSTALLATION")
print("="*70)

print("""
Required packages:
  pip install qdrant-client pymupdf sentence-transformers requests tqdm

Note: Qdrant can run in two modes:
  1. Local mode (using QdrantClient in-memory or local storage)
  2. Cloud mode (using Qdrant Cloud or Docker container)

For this tutorial, we'll use QdrantClient in local mode.
""")

# ============================================================================
# PART 2: DOWNLOAD AZ-104 PDF
# ============================================================================

print("\n" + "="*70)
print("PART 2: DOWNLOADING AZURE AZ-104 LEARNING PDF")
print("="*70)

def download_az104_pdf(pdf_path="az104-learning-guide.pdf"):
    """
    Download Azure AZ-104 learning PDF from Microsoft website.
    
    Note: You may need to manually download from:
    https://learn.microsoft.com/en-us/credentials/certifications/exams/az-104/
    or search for "AZ-104 study guide PDF" on Microsoft Learn
    """
    import requests
    
    # Common URLs where the PDF might be available
    # Note: These are example URLs - actual download may require authentication
    possible_urls = [
        "https://aka.ms/AZ-104StudyGuide",  # Common Microsoft Learn shortcut
        "https://learn.microsoft.com/en-us/training/paths/azure-administrator",
    ]
    
    if os.path.exists(pdf_path):
        print(f"✓ PDF already exists at: {pdf_path}")
        return pdf_path
    
    print("\n⚠ PDF not found. Please download manually:")
    print("   1. Visit: https://learn.microsoft.com/en-us/credentials/certifications/exams/az-104/")
    print("   2. Look for study guide or exam preparation materials")
    print("   3. Download the PDF and save it as 'az104-learning-guide.pdf'")
    print(f"   4. Or place it in the current directory: {os.getcwd()}")
    
    return None

# Check for PDF
pdf_path = download_az104_pdf()

# ============================================================================
# PART 3: PDF TEXT EXTRACTION
# ============================================================================

print("\n" + "="*70)
print("PART 3: EXTRACTING TEXT FROM PDF")
print("="*70)

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF using PyMuPDF (fitz).
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary with text, page info, and metadata
    """
    try:
        import fitz  # PyMuPDF
        
        if not os.path.exists(pdf_path):
            print(f"⚠ PDF not found: {pdf_path}")
            print("Using sample text for demonstration...")
            return {
                'full_text': """
                Azure Administrator (AZ-104) Certification Guide
                
                Module 1: Manage Azure Identities and Governance
                Azure Active Directory (Azure AD) is Microsoft's cloud-based identity and 
                access management service. It helps organizations manage users, groups, and 
                applications. Key features include single sign-on, multi-factor authentication, 
                and conditional access policies.
                
                Module 2: Implement and Manage Storage
                Azure Storage provides scalable cloud storage for data objects. It includes 
                Blob Storage for unstructured data, File Storage for file shares, Queue Storage 
                for messaging, and Table Storage for NoSQL data. Storage accounts can be 
                configured with different performance tiers and redundancy options.
                
                Module 3: Deploy and Manage Azure Compute Resources
                Azure Virtual Machines (VMs) provide on-demand, scalable computing resources. 
                Virtual Machine Scale Sets allow you to create and manage a group of load-balanced 
                VMs. Azure Container Instances offer serverless container execution, while 
                Azure Kubernetes Service (AKS) provides managed Kubernetes orchestration.
                
                Module 4: Implement and Manage Virtual Networking
                Azure Virtual Network (VNet) enables Azure resources to communicate with each 
                other, the internet, and on-premises networks. Key components include subnets, 
                Network Security Groups (NSGs), VPN Gateways, and ExpressRoute for dedicated 
                connectivity.
                """,
                'pages': [{'page_num': 1, 'text': 'Sample text...'}],
                'total_pages': 1
            }
        
        doc = fitz.open(pdf_path)
        full_text = ""
        pages_data = []
        
        print(f"\n--- Processing PDF: {pdf_path} ---")
        print(f"Total pages: {len(doc)}")
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            full_text += text + "\n\n"
            pages_data.append({
                'page_num': page_num,
                'text': text,
                'char_count': len(text)
            })
            
            if page_num % 10 == 0:
                print(f"  Processed {page_num}/{len(doc)} pages...")
        
        doc.close()
        
        print(f"✓ Extracted {len(full_text)} characters from {len(pages_data)} pages")
        
        return {
            'full_text': full_text,
            'pages': pages_data,
            'total_pages': len(pages_data)
        }
        
    except ImportError:
        print("⚠ PyMuPDF not installed. Install with: pip install pymupdf")
        return None
    except Exception as e:
        print(f"⚠ Error extracting PDF: {e}")
        return None

# Extract text from PDF (or use sample if not available)
pdf_data = extract_text_from_pdf(pdf_path if pdf_path else "az104-learning-guide.pdf")

if pdf_data:
    print(f"\nSample text (first 200 chars):")
    print(pdf_data['full_text'][:200] + "...")

# ============================================================================
# PART 4: DATA SPLITTING STRATEGIES
# ============================================================================

print("\n" + "="*70)
print("PART 4: DATA SPLITTING STRATEGIES")
print("="*70)

def split_data_for_vector_db(text, split_ratio=0.8):
    """
    Split data into indexing set and query/validation set.
    
    Args:
        text: Full text content
        split_ratio: Ratio of data for indexing (rest for queries)
        
    Returns:
        Tuple of (indexing_text, query_text)
    """
    from sklearn.model_selection import train_test_split
    
    # Split by paragraphs or sections
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    indexing_paragraphs, query_paragraphs = train_test_split(
        paragraphs,
        test_size=1 - split_ratio,
        random_state=42
    )
    
    indexing_text = '\n\n'.join(indexing_paragraphs)
    query_text = '\n\n'.join(query_paragraphs)
    
    return indexing_text, query_text

if pdf_data:
    indexing_text, query_text = split_data_for_vector_db(pdf_data['full_text'])
    print(f"\nIndexing set: {len(indexing_text)} characters")
    print(f"Query set: {len(query_text)} characters")
    print(f"Split ratio: {len(indexing_text)/(len(indexing_text)+len(query_text))*100:.1f}%")

# ============================================================================
# PART 5: ADVANCED CHUNKING STRATEGIES
# ============================================================================

print("\n" + "="*70)
print("PART 5: CHUNKING STRATEGIES FOR PDF DOCUMENTS")
print("="*70)

def sentence_chunk_text(text, sentences_per_chunk=3, chunk_overlap_sentences=1):
    """
    Split text into chunks by sentences with overlap.
    Best for natural language documents like PDFs.
    
    Args:
        text: Input text
        sentences_per_chunk: Number of sentences per chunk
        chunk_overlap_sentences: Number of overlapping sentences
        
    Returns:
        List of text chunks
    """
    # Split by sentences (handles multiple sentence endings)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
    
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i:i+sentences_per_chunk]
        if chunk_sentences:
            chunk = ' '.join(chunk_sentences)
            chunks.append(chunk)
        i += sentences_per_chunk - chunk_overlap_sentences
    
    return chunks

def token_based_chunk(text, max_tokens=500, overlap_tokens=50):
    """
    Split text into chunks based on approximate token count.
    More accurate for embedding models with token limits.
    
    Args:
        text: Input text
        max_tokens: Maximum tokens per chunk (approx 1 token = 0.75 words)
        overlap_tokens: Overlapping tokens between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        # Approximate: 1 token ≈ 0.75 words, so adjust accordingly
        chunk_size = int(max_tokens * 0.75)
        chunk_words = words[i:i+chunk_size]
        if chunk_words:
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
        i += chunk_size - int(overlap_tokens * 0.75)
    
    return chunks

def paragraph_chunk_text(text, max_chars=1000, overlap_chars=100):
    """
    Split text into chunks by paragraphs with character limits.
    Good for preserving document structure.
    
    Args:
        text: Input text
        max_chars: Maximum characters per chunk
        overlap_chars: Overlapping characters
        
    Returns:
        List of text chunks with metadata
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para in paragraphs:
        para_length = len(para)
        
        if current_length + para_length > max_chars and current_chunk:
            # Save current chunk
            chunks.append({
                'text': '\n\n'.join(current_chunk),
                'char_count': current_length,
                'para_count': len(current_chunk)
            })
            # Start new chunk with overlap
            if overlap_chars > 0 and current_chunk:
                overlap_text = '\n\n'.join(current_chunk[-1:])
                current_chunk = [overlap_text[-overlap_chars:]]
                current_length = len(current_chunk[0])
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(para)
        current_length += para_length + 2  # +2 for \n\n
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            'text': '\n\n'.join(current_chunk),
            'char_count': current_length,
            'para_count': len(current_chunk)
        })
    
    return chunks

# Demonstrate chunking strategies
if pdf_data:
    print("\n--- Strategy 1: Sentence-Based Chunking ---")
    sentence_chunks = sentence_chunk_text(
        pdf_data['full_text'], 
        sentences_per_chunk=3, 
        chunk_overlap_sentences=1
    )
    print(f"Created {len(sentence_chunks)} chunks")
    print(f"Sample chunk: {sentence_chunks[0][:150]}...")
    
    print("\n--- Strategy 2: Token-Based Chunking ---")
    token_chunks = token_based_chunk(pdf_data['full_text'], max_tokens=300, overlap_tokens=30)
    print(f"Created {len(token_chunks)} chunks")
    print(f"Sample chunk: {token_chunks[0][:150]}...")
    
    print("\n--- Strategy 3: Paragraph-Based Chunking ---")
    para_chunks = paragraph_chunk_text(pdf_data['full_text'], max_chars=800, overlap_chars=100)
    print(f"Created {len(para_chunks)} chunks")
    print(f"Average chunk size: {sum(c['char_count'] for c in para_chunks)/len(para_chunks):.0f} chars")
    print(f"Sample chunk: {para_chunks[0]['text'][:150]}...")

# ============================================================================
# PART 6: QDRANT VECTOR DATABASE INTEGRATION
# ============================================================================

print("\n" + "="*70)
print("PART 6: STORING CHUNKS IN QDRANT")
print("="*70)

def setup_qdrant_collection(collection_name="az104_chunks", vector_size=384):
    """
    Initialize Qdrant client and create collection.
    
    Args:
        collection_name: Name of the collection
        vector_size: Size of embedding vectors (384 for all-MiniLM-L6-v2)
        
    Returns:
        QdrantClient instance and collection info
    """
    try:
        from qdrant_client import QdrantClient, models
        
        # Initialize Qdrant client (local mode)
        # For production, use: QdrantClient(url="https://your-cluster.qdrant.io", api_key="...")
        client = QdrantClient(location=":memory:")  # In-memory for tutorial
        # For persistent storage: QdrantClient(path="./qdrant_db")
        
        print(f"✓ Connected to Qdrant")
        
        # Check if collection exists
        try:
            collection_info = client.get_collection(collection_name)
            print(f"✓ Collection '{collection_name}' already exists")
            # Optionally delete and recreate for fresh start
            # client.delete_collection(collection_name)
        except:
            # Create new collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE  # Cosine similarity
                )
            )
            print(f"✓ Created collection '{collection_name}'")
        
        return client, collection_name
        
    except ImportError:
        print("⚠ qdrant-client not installed. Install with: pip install qdrant-client")
        return None, None
    except Exception as e:
        print(f"⚠ Error setting up Qdrant: {e}")
        return None, None

def store_chunks_in_qdrant(client, collection_name, chunks, embeddings, metadata=None):
    """
    Store text chunks and their embeddings in Qdrant.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        chunks: List of text chunks
        embeddings: List of embedding vectors
        metadata: Optional list of metadata dictionaries
    """
    try:
        from qdrant_client import models
        
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_metadata = metadata[idx] if metadata and idx < len(metadata) else {}
            point_metadata['text_length'] = len(chunk)
            point_metadata['chunk_id'] = idx
            
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                    payload={
                        'text': chunk,
                        **point_metadata
                    }
                )
            )
        
        # Batch upsert
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        print(f"✓ Stored {len(points)} chunks in Qdrant")
        return True
        
    except Exception as e:
        print(f"⚠ Error storing in Qdrant: {e}")
        return False

def query_qdrant(client, collection_name, query_text, embedding_model, top_k=5):
    """
    Query Qdrant vector database with a text query.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        query_text: Search query text
        embedding_model: SentenceTransformer model for generating embeddings
        top_k: Number of results to return
        
    Returns:
        List of search results
    """
    try:
        # Generate embedding for query
        query_embedding = embedding_model.encode(query_text).tolist()
        
        # Search in Qdrant
        results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        return results
        
    except Exception as e:
        print(f"⚠ Error querying Qdrant: {e}")
        return []

# ============================================================================
# PART 7: COMPLETE WORKFLOW EXAMPLE
# ============================================================================

print("\n" + "="*70)
print("PART 7: COMPLETE END-TO-END WORKFLOW")
print("="*70)

def complete_workflow_demo():
    """
    Complete workflow:
    1. Extract PDF text
    2. Split into indexing/query sets
    3. Chunk the indexing text
    4. Generate embeddings
    5. Store in Qdrant
    6. Query the database
    """
    
    # Check if we have PDF data
    if not pdf_data:
        print("⚠ No PDF data available. Using sample text.")
        sample_text = """
        Azure Administrator Certification covers managing Azure identities, storage, compute, 
        and networking resources. Key topics include Azure Active Directory, Virtual Machines, 
        Storage Accounts, Virtual Networks, and monitoring solutions.
        """
        pdf_data_local = {'full_text': sample_text}
    else:
        pdf_data_local = pdf_data
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("\n--- Step 1: Loading Embedding Model ---")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"✓ Loaded model: all-MiniLM-L6-v2 (dimension: {model.get_sentence_embedding_dimension()})")
        
        # Step 2: Split data
        print("\n--- Step 2: Splitting Data ---")
        indexing_text, query_text = split_data_for_vector_db(pdf_data_local['full_text'])
        print(f"✓ Split data: {len(indexing_text)} chars for indexing, {len(query_text)} chars for queries")
        
        # Step 3: Chunk indexing text
        print("\n--- Step 3: Chunking Documents ---")
        chunks = sentence_chunk_text(indexing_text, sentences_per_chunk=3, chunk_overlap_sentences=1)
        print(f"✓ Created {len(chunks)} chunks")
        
        # Step 4: Generate embeddings
        print("\n--- Step 4: Generating Embeddings ---")
        embeddings = model.encode(chunks, show_progress_bar=True)
        print(f"✓ Generated {len(embeddings)} embeddings")
        
        # Step 5: Setup Qdrant
        print("\n--- Step 5: Setting up Qdrant ---")
        client, collection_name = setup_qdrant_collection(
            collection_name="az104_chunks",
            vector_size=model.get_sentence_embedding_dimension()
        )
        
        if not client:
            print("⚠ Skipping Qdrant storage due to setup error")
            return False
        
        # Step 6: Store in Qdrant
        print("\n--- Step 6: Storing Chunks in Qdrant ---")
        metadata = [{'source': 'az104_pdf', 'chunk_index': i} for i in range(len(chunks))]
        store_chunks_in_qdrant(client, collection_name, chunks, embeddings, metadata)
        
        # Step 7: Query examples
        print("\n--- Step 7: Querying Vector Database ---")
        query_examples = [
            "What is Azure Active Directory?",
            "How do I manage Azure storage?",
            "Explain Azure Virtual Machines",
        ]
        
        for query in query_examples:
            print(f"\n🔍 Query: '{query}'")
            results = query_qdrant(client, collection_name, query, model, top_k=3)
            
            if results:
                print(f"   Top {len(results)} results:")
                for i, result in enumerate(results, 1):
                    score = result.score
                    text_preview = result.payload['text'][:150]
                    print(f"   {i}. [Score: {score:.3f}] {text_preview}...")
            else:
                print("   No results found")
        
        print("\n" + "="*70)
        print("✓ WORKFLOW COMPLETE!")
        print("="*70)
        
        return True
        
    except ImportError as e:
        print(f"\n⚠ Missing dependency: {e}")
        print("Install with: pip install sentence-transformers qdrant-client")
        return False
    except Exception as e:
        print(f"\n⚠ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

# Run the complete workflow
complete_workflow_demo()

# ============================================================================
# PART 8: BEST PRACTICES AND TIPS
# ============================================================================

print("\n" + "="*70)
print("BEST PRACTICES FOR PDF CHUNKING WITH QDRANT")
print("="*70)

best_practices = """
📚 CHUNKING STRATEGIES FOR PDFS:

1. SENTENCE-BASED CHUNKING (Recommended for PDFs):
   ✓ Preserves context and meaning
   ✓ Better semantic understanding
   ✓ Ideal for documents with natural language
   ✓ Use 2-5 sentences per chunk with 1 sentence overlap

2. PARAGRAPH-BASED CHUNKING:
   ✓ Maintains document structure
   ✓ Good for technical documents
   ✓ Respects natural breaks in content
   ✓ Use 500-1000 characters per chunk

3. TOKEN-BASED CHUNKING:
   ✓ Respects model token limits
   ✓ Good for embedding models with constraints
   ✓ More precise size control
   ✓ Use 256-512 tokens per chunk

🔧 QDRANT CONFIGURATION:

1. DISTANCE METRICS:
   - COSINE: Best for semantic similarity (recommended)
   - EUCLIDEAN: Good for L2 distance
   - DOT: For normalized vectors

2. VECTOR SIZE:
   - all-MiniLM-L6-v2: 384 dimensions (fast, good quality)
   - all-mpnet-base-v2: 768 dimensions (better quality)
   - BGE models: 768+ dimensions (state-of-the-art)

3. STORAGE OPTIONS:
   - In-memory: Fast, but data lost on restart
   - Local storage: Persistent, good for development
   - Qdrant Cloud: Production-ready, scalable

📊 DATA SPLITTING:

1. INDEXING VS QUERY SET:
   - 80-90% for indexing (chunks to search over)
   - 10-20% for queries (test queries)

2. VALIDATION:
   - Keep test queries separate
   - Don't peek at test set!
   - Use for evaluation metrics

🎯 OPTIMIZATION TIPS:

1. CHUNK SIZE:
   - Too small: May lose context
   - Too large: May include irrelevant info
   - Sweet spot: 200-500 tokens

2. OVERLAP:
   - 10-20% overlap maintains context
   - Prevents information loss at boundaries
   - Improves retrieval quality

3. METADATA:
   - Store page numbers, sections, source info
   - Enables filtered searches
   - Helps with result interpretation

4. BATCH PROCESSING:
   - Process chunks in batches
   - Use progress bars for large documents
   - Monitor memory usage
"""

print(best_practices)

print("\n" + "="*70)
print("LESSON COMPLETE! 🎓")
print("="*70)
print("\nNext Steps:")
print("1. Download Azure AZ-104 PDF from Microsoft Learn")
print("2. Install dependencies: pip install -r requirements.txt")
print("3. Run this script: python qdrant_pdf_tutorial.py")
print("4. Experiment with different chunking strategies")
print("5. Build your own Q&A system with the PDF!")
print("\n")

