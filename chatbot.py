# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# class DocumentChatbot:
#     def __init__(self):
#         # Initialize the embedding model
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # Initialize the LLM for response generation
#         self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
#         self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        
#         # Store document chunks and their embeddings
#         self.document_chunks = []
#         self.chunk_embeddings = None
        
#     def preprocess_document(self, text: str, chunk_size: int = 200):
#         """Split document into chunks and compute embeddings"""
#         # Simple chunk creation by sentences
#         sentences = text.split('.')
#         chunks = []
#         current_chunk = ""
        
#         for sentence in sentences:
#             if len(current_chunk) + len(sentence) < chunk_size:
#                 current_chunk += sentence + "."
#             else:
#                 if current_chunk:
#                     chunks.append(current_chunk)
#                 current_chunk = sentence + "."
        
#         if current_chunk:
#             chunks.append(current_chunk)
            
#         self.document_chunks = chunks
#         self.chunk_embeddings = self.embedding_model.encode(chunks)
        
#     def get_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
#         """Retrieve most relevant document chunks for the query"""
#         query_embedding = self.embedding_model.encode([query])
        
#         # Calculate similarity scores
#         similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
#         # Get top-k chunk indices
#         top_indices = np.argsort(similarities)[-top_k:]
        
#         # Return relevant chunks
#         return [self.document_chunks[i] for i in top_indices]
    
#     def generate_response(self, query: str, context: List[str]) -> str:
#         """Generate response based on query and retrieved context"""
#         # Combine context and query into a prompt

#         combined_context = ' '.join(context)
#         if len(combined_context) > 1000:
#             combined_context=combined_context[:1000]+"..."  # arbitrary threshold
#         prompt = f"""Context: {' '.join(context)}
        
# Question: {query}

# Answer: """
#         # Tokenizer configuration
#         tokenizer_kwargs = {
#             "return_tensors": "pt",
#             "max_length": 512,
#             "truncation": True,
#             "padding": True
#         }
        
#         # Generate response using the model
#         inputs = self.tokenizer(prompt, **tokenizer_kwargs)
        
#         outputs = self.model.generate(
#             inputs.input_ids,
#             max_length=512,
#             min_length=30,
#             num_beams=4,
#             temperature=0.7,
#             top_p=0.9,
#             truncation=True,
#             no_repeat_ngram_size=3
#         )
        
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         # Extract only the answer part
#         try:
#             answer = response.split("Answer: ")[1].strip()
#         except IndexError:
#             answer = response.strip()
            
#         return answer
    
#     def chat(self, query: str) -> str:
#         """Main chat function that combines retrieval and generation"""
#         if not self.document_chunks or self.chunk_embeddings is None:
#             return "Please upload and process a document first."
            
#         try:
#             # Get relevant context
#             relevant_chunks = self.get_relevant_context(query)
            
#             # Generate response
#             response = self.generate_response(query, relevant_chunks)
            
#             return response
            
#         except Exception as e:
#             return f"An error occurred: {str(e)}"

#fixed code 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DocumentChatbot:
    def __init__(self):
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize the LLM for response generation
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        self.model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        
        # Store document chunks and their embeddings
        self.document_chunks = []
        self.chunk_embeddings = None
        
    def preprocess_document(self, text: str, chunk_size: int = 200):
        """Split document into chunks and compute embeddings"""
        # Simple chunk creation by sentences
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk)
            
        self.document_chunks = chunks
        self.chunk_embeddings = self.embedding_model.encode(chunks)
        
    def get_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve most relevant document chunks for the query"""
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # Get top-k chunk indices
        top_indices = np.argsort(similarities)[-top_k:]
        
        # Return relevant chunks
        return [self.document_chunks[i] for i in top_indices]
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response based on query and retrieved context"""
        # Combine context and query into a prompt
        combined_context = ' '.join(context)
        if len(combined_context) > 1000:
            combined_context = combined_context[:1000] + "..."  # arbitrary threshold
        prompt = f"""Context: {combined_context}
        
Question: {query}

Answer: """
        # Tokenizer configuration
        tokenizer_kwargs = {
            "return_tensors": "pt",
            "max_length": 512,
            "truncation": True,
            "padding": True
        }
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, **tokenizer_kwargs)
        
        # Generate response using the model
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=512,
            min_length=10,
            num_beams=1,
            temperature=1.0,
            top_p=0.8,
            no_repeat_ngram_size=3
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part
        try:
            answer = response.split("Answer: ")[1].strip()
        except IndexError:
            answer = response.strip()
            
        return answer
    
    def chat(self, query: str) -> str:
        """Main chat function that combines retrieval and generation"""
        if not self.document_chunks or self.chunk_embeddings is None:
            return "Please upload and process a document first."
            
        try:
            # Get relevant context
            relevant_chunks = self.get_relevant_context(query)
            
            # Generate response
            response = self.generate_response(query, relevant_chunks)
            
            return response
            
        except Exception as e:
            return f"An error occurred: {str(e)}"
#using faiss

