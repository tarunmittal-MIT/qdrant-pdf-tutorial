# Data Splitting and Chunking Tutorial with Qdrant

Welcome! This tutorial teaches you how to work with data splitting and chunking for vector databases using Python and Qdrant, with the Azure AZ-104 learning PDF as our example document.

## 📚 Learning Objectives

By the end of this tutorial, you will:
- Understand data splitting strategies for vector databases
- Learn multiple chunking techniques (sentence, token, paragraph-based)
- Extract text from PDF documents
- Store and query data in Qdrant vector database
- Build a complete end-to-end pipeline

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Azure AZ-104 PDF

The Azure AZ-104 learning PDF can be downloaded from Microsoft Learn:

**Official Resources:**
- [AZ-104 Exam Page](https://learn.microsoft.com/en-us/credentials/certifications/exams/az-104/)
- [Azure Administrator Learning Path](https://learn.microsoft.com/en-us/training/paths/azure-administrator/)

**Option 1: Manual Download**
1. Visit the Microsoft Learn pages above
2. Look for study guides or exam preparation materials
3. Download the PDF and save it as `az104-learning-guide.pdf` in this directory

**Option 2: Use Helper Script**
```bash
python download_pdf_helper.py
```

**Note:** The tutorial will work even without the PDF - it includes sample text for demonstration.

### 3. Run the Tutorial

```bash
python qdrant_pdf_tutorial.py
```

## 📁 Files Overview

- **`qdrant_pdf_tutorial.py`** - Main tutorial script with all lessons and examples
- **`download_pdf_helper.py`** - Helper script to locate and download the AZ-104 PDF
- **`requirements.txt`** - All required Python packages
- **`vector_db_tutorial.py`** - Original tutorial (uses ChromaDB for reference)

## 📖 Tutorial Structure

### Part 1: Setup and Installation
- Installing required packages
- Understanding Qdrant setup options

### Part 2: PDF Download
- Downloading Azure AZ-104 PDF
- Understanding PDF sources

### Part 3: PDF Text Extraction
- Extracting text using PyMuPDF
- Handling page-by-page extraction

### Part 4: Data Splitting
- Splitting data into indexing and query sets
- Understanding train/test splits for vector databases

### Part 5: Chunking Strategies
- **Sentence-based chunking**: Best for natural language
- **Token-based chunking**: Respects model limits
- **Paragraph-based chunking**: Maintains document structure

### Part 6: Qdrant Integration
- Setting up Qdrant collections
- Storing chunks with embeddings
- Querying the vector database

### Part 7: Complete Workflow
- End-to-end pipeline demonstration
- Real query examples

### Part 8: Best Practices
- Chunking recommendations
- Qdrant configuration tips
- Optimization strategies

## 🔧 Key Technologies

- **Qdrant**: High-performance vector database
- **PyMuPDF (fitz)**: PDF text extraction
- **Sentence Transformers**: Generate embeddings
- **scikit-learn**: Data splitting utilities

## 💡 Key Concepts

### Data Splitting
Dividing your dataset into:
- **Indexing set**: Documents to search over (80-90%)
- **Query set**: Test queries for evaluation (10-20%)

### Chunking Strategies

1. **Sentence-Based** (Recommended)
   - Preserves context and meaning
   - 2-5 sentences per chunk
   - 1 sentence overlap

2. **Token-Based**
   - Respects embedding model limits
   - 256-512 tokens per chunk
   - More precise size control

3. **Paragraph-Based**
   - Maintains document structure
   - 500-1000 characters per chunk
   - Good for technical documents

### Qdrant Configuration

- **Distance Metric**: Cosine similarity (best for semantic search)
- **Vector Size**: 384 (all-MiniLM-L6-v2) or 768 (all-mpnet-base-v2)
- **Storage**: In-memory (tutorial), local storage, or Qdrant Cloud

## 🎯 Practice Exercises

1. **Experiment with Chunk Sizes**
   - Try different chunk sizes and observe search quality
   - Compare sentence vs. token vs. paragraph chunking

2. **Build a Q&A System**
   - Store the entire AZ-104 PDF in Qdrant
   - Create a function to answer questions about Azure

3. **Optimize Retrieval**
   - Add metadata (page numbers, sections)
   - Use filters for more precise searches

4. **Evaluate Performance**
   - Test with different embedding models
   - Measure query response times

## 📝 Notes

- The tutorial uses Qdrant in-memory mode by default (data is not persisted)
- For production, consider using local storage or Qdrant Cloud
- Sample text is included if the PDF is not available
- All examples are fully functional and can be run independently

## 🔗 Additional Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Sentence Transformers](https://www.sbert.net/)
- [Microsoft Learn](https://learn.microsoft.com/)
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)

## ❓ Troubleshooting

**Issue: Import errors**
- Solution: Run `pip install -r requirements.txt`

**Issue: PDF not found**
- Solution: Download PDF manually or use sample text mode

**Issue: Qdrant connection errors**
- Solution: The tutorial uses in-memory mode, no external setup needed

**Issue: Memory errors with large PDFs**
- Solution: Process PDF in batches or reduce chunk size

---

Happy Learning! 🎓

