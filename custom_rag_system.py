from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.llms.cohere import Cohere
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader, DocxReader
from qdrant_client import QdrantClient
import cohere
import os

# Your API keys
COHERE_API_KEY = "Iftf4J9ZlMqANexAhp5CTk644Z7i2Lq0aqfmeqxC"
QDRANT_URL = "https://39910d80-039b-4577-b490-72e3a8219ba5.europe-west3-0.gcp.cloud.qdrant.io/"
QDRANT_API_KEY = "Xt7g1TmFWmhVGAucembRPi73mzxV4kNXgoZmAmd4PsP54fF50xKpFg"

class RAGSystem:
    def __init__(self, collection_name="test_collection"):
        # Initialize Cohere embedding
        self.embed_model = CohereEmbedding(
            api_key=COHERE_API_KEY,
            model_name="embed-english-v3.0"
        )
        
        # Initialize LLM
        self.llm = Cohere(
            api_key=COHERE_API_KEY,
            model="command",
            temperature=0.7,
            max_tokens=512
        )
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=20
        )
        
        # Initialize Qdrant
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        # Create vector store
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=collection_name
        )
        
        # Configure settings
        Settings.chunk_size = 512
        Settings.chunk_overlap = 20
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.node_parser = self.node_parser
        
        # Initialize index
        self.index = None
        
        print("RAG system initialized successfully!")

    def load_documents(self, directory_path):
        """Load documents from a directory."""
        documents = []
        
        # Initialize readers
        pdf_reader = PDFReader()
        docx_reader = DocxReader()
        
        print(f"\nLoading documents from {directory_path}...")
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            try:
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        documents.append(Document(text=text))
                        print(f"Loaded text file: {filename}")
                
                elif filename.endswith('.pdf'):
                    docs = pdf_reader.load_data(file_path)
                    documents.extend(docs)
                    print(f"Loaded PDF file: {filename}")
                
                elif filename.endswith('.docx'):
                    docs = docx_reader.load_data(file_path)
                    documents.extend(docs)
                    print(f"Loaded DOCX file: {filename}")
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
        
        if documents:
            self.index = VectorStoreIndex.from_documents(
                documents,
                vector_store=self.vector_store
            )
            print(f"\nSuccessfully loaded {len(documents)} documents")
        else:
            print("\nNo documents were loaded!")
            
        return len(documents)

    def query(self, question: str) -> str:
        """Query the knowledge base."""
        if not self.index:
            return "No documents have been loaded yet!"
            
        query_engine = self.index.as_query_engine(
            similarity_top_k=2,
            streaming=False
        )
        response = query_engine.query(question)
        return str(response)

def main():
    # Initialize the RAG system
    rag = RAGSystem()
    
    # Get directory path from user
    while True:
        dir_path = input("\nEnter the path to your documents directory: ").strip()
        if os.path.isdir(dir_path):
            break
        print("Invalid directory path! Please try again.")
    
    # Load documents
    num_docs = rag.load_documents(dir_path)
    
    if num_docs == 0:
        print("No documents were loaded. Please add some documents and try again.")
        return
    
    print("\nRAG system is ready! Type your questions (or 'quit' to exit)")
    print("-" * 50)
    
    while True:
        question = input("\nEnter your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the RAG system!")
            break
            
        if not question:
            print("Please enter a valid question!")
            continue
            
        try:
            answer = rag.query(question)
            print("\nAnswer:", answer)
            print("-" * 50)
        except Exception as e:
            print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
