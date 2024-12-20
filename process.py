from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from transformers import pipeline
import pypdf
import pandas as pd
import re
import numpy as np

class RAGPipeline:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=f"sentence-transformers/{embedding_model}")
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF file."""
        pdf_reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
    def extract_gdp_table(self, text):
        """Extract GDP table data specifically."""
        # Look for the GDP table section
        table_start = text.find("Table of Yearly U.S. GDP")
        if table_start == -1:
            return None
        
        # Extract the relevant text chunk
        table_text = text[table_start:table_start + 1000]  # Adjust size as needed
        
        # Extract years and values
        years = []
        all_industries = []
        manufacturing = []
        finance = []
        arts = []
        other = []
        
        # Parse the table data using regex
        year_pattern = r'(\d{4})'
        value_pattern = r'(\d+(?:,\d+)*)'
        
        # Find all years
        years = re.findall(year_pattern, table_text)
        
        # Find all numeric values
        values = re.findall(value_pattern, table_text)
        
        # Convert to DataFrame
        data = []
        current_row = []
        for value in values:
            value = int(value.replace(',', ''))
            current_row.append(value)
            if len(current_row) == 5:  # We have 5 columns
                data.append(current_row)
                current_row = []
        
        if data:
            df = pd.DataFrame(data, columns=['All Industries', 'Manufacturing', 'Finance', 'Arts', 'Other'])
            df['Year'] = years[:len(df)]
            return df
        return None

    def process_pdf(self, pdf_path):
        """Process PDF file and store embeddings."""
        # Extract text
        text = self.extract_pdf_text(pdf_path)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(chunks, self.embeddings)
        else:
            self.vector_store.add_texts(chunks)
        
        # Extract and store GDP table
        self.gdp_table = self.extract_gdp_table(text)
        
        return len(chunks)

    def query(self, user_query, k=3):
        """Query the vector store and return relevant chunks."""
        if self.vector_store is None:
            raise ValueError("No documents have been processed yet.")
        
        # Check if query is about GDP
        if 'gdp' in user_query.lower():
            # Try to extract year from query
            year_match = re.search(r'\b(20\d{2})\b', user_query)
            if year_match and self.gdp_table is not None:
                year = int(year_match.group(1))
                gdp_data = self.gdp_table[self.gdp_table['Year'] == year]
                if not gdp_data.empty:
                    return [(f"GDP data for {year}:\n{gdp_data.to_string()}", 1.0)]
        
        results = self.vector_store.similarity_search_with_score(user_query, k=k)
        return results

def main():
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Process PDF
    pdf_path = "E:/rahultrail/source.pdf"
    num_chunks = rag.process_pdf(pdf_path)
    print(f"Processed {num_chunks} chunks from PDF")
    
    # Example GDP query
    gdp_query = "What types of graphs are discussed"
    results = rag.query(gdp_query)
    
    # Print results
    print("\nQuery Results:")
    for doc, score in results:
        print(f"Score: {score}")
        print(f"Content: {doc}\n")
    
    # Example of other queries
    general_query = "What types of graphs are discussed in the document?"
    results = rag.query(general_query)
    
    print("\nGeneral Query Results:")
    for doc, score in results:
        print(f"Score: {score}")
        print(f"Content: {doc.page_content}\n")

if __name__ == "__main__":
    main()