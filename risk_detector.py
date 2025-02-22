from transformers import pipeline
import spacy
import networkx as nx
from typing import Dict, List, Tuple
import logging

class LegalRiskDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        
        self.risk_categories = [
            "high_risk",
            "medium_risk",
            "low_risk",
            "neutral"
        ]
        
        self.risk_patterns = {
            "ambiguous_terms": [
                r"reasonable",
                r"substantial",
                r"material",
                r"appropriate",
                r"as necessary",
                r"subject to",
                r"deemed"
            ],
            "hidden_obligations": [
                r"shall",
                r"must",
                r"required to",
                r"obligated",
                r"responsible for"
            ],
            "conditional_clauses": [
                r"provided that",
                r"subject to",
                r"contingent upon",
                r"in the event",
                r"if and only if"
            ]
        }

    def analyze_dependencies(self, text: str) -> List[Dict]:
        """Analyze clause dependencies using SpaCy's dependency parser."""
        doc = self.nlp(text)
        dependencies = []
        
        for sent in doc.sents:
            clause_deps = []
            root = [token for token in sent if token.dep_ == "ROOT"][0]
            
            # Build dependency tree
            for token in sent:
                if token.dep_ in ["advcl", "ccomp", "xcomp"]:
                    clause_deps.append({
                        "main_clause": root.text,
                        "dependent_clause": token.text,
                        "dependency_type": token.dep_
                    })
            
            if clause_deps:
                dependencies.extend(clause_deps)
        
        return dependencies

    def classify_risk_level(self, text: str) -> Dict:
        """Classify the risk level of legal text."""
        classification = self.classifier(
            sequences=text,
            candidate_labels=self.risk_categories,
            multi_label=True
        )
        
        # Get the highest scoring risk category
        max_score_idx = classification['scores'].index(max(classification['scores']))
        risk_level = classification['labels'][max_score_idx]
        confidence = classification['scores'][max_score_idx]
        
        return {
            "risk_level": risk_level,
            "confidence": confidence,
            "text": text
        }

    def detect_ambiguities(self, text: str) -> List[Dict]:
        """Detect ambiguous terms and phrases."""
        doc = self.nlp(text)
        ambiguities = []
        
        for pattern_type, patterns in self.risk_patterns.items():
            for pattern in patterns:
                matches = [sent.text for sent in doc.sents 
                          if any(term in sent.text.lower() 
                          for term in [pattern.lower()])]
                
                if matches:
                    ambiguities.append({
                        "type": pattern_type,
                        "pattern": pattern,
                        "contexts": matches
                    })
        
        return ambiguities

    def analyze_risks(self, text: str) -> Dict:
        """Perform comprehensive risk analysis of legal text."""
        try:
            dependencies = self.analyze_dependencies(text)
            risk_classification = self.classify_risk_level(text)
            ambiguities = self.detect_ambiguities(text)
            
            return {
                "risk_classification": risk_classification,
                "dependencies": dependencies,
                "ambiguities": ambiguities,
                "status": "success"
            }
        except Exception as e:
            logging.error(f"Error in risk analysis: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

def format_risk_analysis(analysis_results: Dict) -> str:
    """Format risk analysis results for display."""
    output = "🚨 Risk Analysis Report\n\n"
    
    if analysis_results["status"] == "error":
        return f"Error in risk analysis: {analysis_results['error']}"
    
    # Risk Classification
    risk_class = analysis_results["risk_classification"]
    output += f"Risk Level: {risk_class['risk_level']}\n"
    output += f"Confidence: {risk_class['confidence']:.2%}\n\n"
    
    # Dependencies
    output += "🔄 Clause Dependencies:\n"
    output += "----------------------\n"
    if analysis_results["dependencies"]:
        for dep in analysis_results["dependencies"]:
            output += f"• Main Clause: {dep['main_clause']}\n"
            output += f"  Dependent on: {dep['dependent_clause']}\n"
            output += f"  Type: {dep['dependency_type']}\n\n"
    else:
        output += "No significant dependencies found.\n\n"
    
    # Ambiguities
    output += "⚠️ Potential Ambiguities:\n"
    output += "------------------------\n"
    if analysis_results["ambiguities"]:
        for amb in analysis_results["ambiguities"]:
            output += f"\n{amb['type'].replace('_', ' ').title()}:\n"
            output += f"Pattern: '{amb['pattern']}'\n"
            output += "Contexts:\n"
            for context in amb["contexts"]:
                output += f"  • {context}\n"
    else:
        output += "No significant ambiguities detected.\n"
    
    return output