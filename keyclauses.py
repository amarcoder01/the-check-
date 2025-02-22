from transformers import pipeline
import re
import spacy
import numpy as np

class LegalClauseExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  
        )

        # Define key legal terms and clause categories
        self.clause_categories = [
            "payment terms",
            "termination",
            "confidentiality",
            "liability",
            "intellectual property",
            "warranty",
            "indemnification",
            "force majeure",
            "governing law",
            "dispute resolution"
        ]

        # Common legal terms patterns
        self.legal_terms_patterns = [
            r"shall|must|will",
            r"party|parties",
            r"agree|agreement",
            r"term|period",
            r"terminate|termination",
            r"confidential|confidentiality",
            r"liability|indemnify|indemnification",
            r"intellectual property|IP",
            r"warrant|warranty|warranties",
            r"law|jurisdiction|governing",
            r"dispute|arbitration|mediation"
        ]

    def extract_key_terms(self, text):
        """Extract important legal terms from the text."""
        doc = self.nlp(text)
        key_terms = []

        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Extract terms matching legal patterns
        for pattern in self.legal_terms_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group()
                if len(term) > 3:  
                    key_terms.append(term)

        return list(set(key_terms))  # Remove duplicates

    def extract_clauses(self, text):
        """Extract clauses from text using zero-shot classification."""
        # Split text into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Remove empty sentences and very short ones
        sentences = [sent for sent in sentences if len(sent) > 10]

        clauses = {}
        
        for category in self.clause_categories:
            relevant_sentences = []
            
            # Process each sentence individually instead of batches
            for sentence in sentences:
                try:
                    result = self.classifier(
                        sentence,
                        candidate_labels=[category],
                        multi_label=False
                    )
                    
                    # Check if the confidence score is high enough
                    if result['scores'][0] > 0.6:  # Lowered threshold slightly
                        relevant_sentences.append(sentence)
                        
                except Exception as e:
                    print(f"Error processing sentence for category {category}: {str(e)}")
                    continue
            
            if relevant_sentences:
                clauses[category] = relevant_sentences

        return clauses

    def analyze_document(self, text):
        """Perform complete analysis of the legal document."""
        try:
            key_terms = self.extract_key_terms(text)
            key_clauses = self.extract_clauses(text)
            return {
                'key_terms': key_terms,
                'key_clauses': key_clauses
            }
        except Exception as e:
            print(f"Error in analyze_document: {str(e)}")
            return {
                'key_terms': [],
                'key_clauses': {}
            }

def format_analysis_results(results):
    """Format the analysis results for display."""
    formatted_text = "📑 Document Analysis Results\n\n"
    
    # Format key terms
    formatted_text += "🔑 Key Legal Terms:\n"
    formatted_text += "-------------------\n"
    if results['key_terms']:
        for term in sorted(set(results['key_terms'])):  # Sort and remove duplicates
            formatted_text += f"• {term}\n"
    else:
        formatted_text += "No key terms found.\n"
    
    # Format clauses by category
    formatted_text += "\n📋 Key Clauses by Category:\n"
    formatted_text += "-------------------------\n"
    if results['key_clauses']:
        for category in sorted(results['key_clauses'].keys()):  # Sort categories
            formatted_text += f"\n🔸 {category.title()}:\n"
            for clause in results['key_clauses'][category]:
                formatted_text += f"  • {clause.strip()}\n"
    else:
        formatted_text += "\nNo key clauses found.\n"
    
    return formatted_text

def analyze_legal_document(text):
    """Main function to analyze a legal document."""
    try:
        extractor = LegalClauseExtractor()
        results = extractor.analyze_document(text)
        return format_analysis_results(results)
    except Exception as e:
        return f"Error analyzing document: {str(e)}\n\nPlease try again with a different document or contact support."