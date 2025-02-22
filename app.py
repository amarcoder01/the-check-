
#with risk detection and analysis
import streamlit as st
import spacy
import networkx as nx
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import torch
import base64
import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import ssl
import logging
import feedparser

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import sqlite3
import json
import schedule
import time
from typing import Dict, List, Tuple
from chatbot import DocumentChatbot
from regulatory_tracker import RegulatoryTracker
import matplotlib.pyplot as plt
import io

def create_simple_visualizations(analysis_results, risk_analysis):
    """Create simple visualizations for document analysis"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#F0F2F6')
    
    # 1. Simple bar chart of clause categories
    if analysis_results and 'key_clauses' in analysis_results and analysis_results['key_clauses']:
        categories = list(analysis_results['key_clauses'].keys())
        counts = [len(clauses) for clauses in analysis_results['key_clauses'].values()]
        
        ax1.bar(range(len(categories)), counts, color='skyblue')
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories, rotation=45, ha='right')
    else:
        ax1.text(0.5, 0.5, 'No clause data available', 
                horizontalalignment='center', verticalalignment='center')
    ax1.set_title('Clauses by Category')
    ax1.set_ylabel('Number of Clauses')
    
    # 2. Simple pie chart of risk distribution
    risk_counts = {'High Risk': 0, 'Medium Risk': 0, 'Low Risk': 0}
    
    if analysis_results and 'key_clauses' in analysis_results:
        for category, clauses in analysis_results.get('key_clauses', {}).items():
            if any('urgent' in clause.lower() or 'critical' in clause.lower() for clause in clauses):
                risk_counts['High Risk'] += 1
            elif any('important' in clause.lower() or 'should' in clause.lower() for clause in clauses):
                risk_counts['Medium Risk'] += 1
            else:
                risk_counts['Low Risk'] += 1
    
        if sum(risk_counts.values()) > 0:
            colors = ['#ff9999', '#ffcc99', '#99ff99']
            ax2.pie(risk_counts.values(), labels=risk_counts.keys(), colors=colors, autopct='%1.1f%%')
        else:
            ax2.text(0.5, 0.5, 'No risk data available', 
                    horizontalalignment='center', verticalalignment='center')
    else:
        ax2.text(0.5, 0.5, 'No risk data available', 
                horizontalalignment='center', verticalalignment='center')
    ax2.set_title('Risk Distribution')
    
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf
    
    # 2. Simple pie chart of risk distribution
    risk_counts = {'High Risk': 0, 'Medium Risk': 0, 'Low Risk': 0}
    for category, clauses in analysis_results.get('key_clauses', {}).items():
        if any('urgent' in clause.lower() or 'critical' in clause.lower() for clause in clauses):
            risk_counts['High Risk'] += 1
        elif any('important' in clause.lower() or 'should' in clause.lower() for clause in clauses):
            risk_counts['Medium Risk'] += 1
        else:
            risk_counts['Low Risk'] += 1
    
    colors = ['#ff9999', '#ffcc99', '#99ff99']
    ax2.pie(risk_counts.values(), labels=risk_counts.keys(), colors=colors, autopct='%1.1f%%')
    ax2.set_title('Risk Distribution')
    
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf
# Email Sender Class
class EmailSender:
    @staticmethod
    def send_email(sender_email, sender_password, recipient_email, subject, body):
        try:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
            logger.info(f"Attempting to connect to SMTP server for: {sender_email}")
            
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = recipient_email
            msg["Subject"] = subject
            
            msg.attach(MIMEText(body, "plain"))
            
            context = ssl.create_default_context()
            
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                logger.info("Connected to SMTP server, attempting login...")
                server.login(sender_email, sender_password)
                logger.info("Login successful, sending message...")
                server.send_message(msg)
                
            return "Email sent successfully!"
            
        except smtplib.SMTPAuthenticationError as e:
            error_msg = str(e)
            if "Application-specific password required" in error_msg:
                return ("Authentication failed: You need to use an App Password. "
                        "Please go to Google Account > Security > App Passwords to generate one.")
            elif "Username and Password not accepted" in error_msg:
                return ("Authentication failed: Username and Password not accepted. "
                        "If using Gmail, please ensure:\n"
                        "1. You've enabled 2-Step Verification\n"
                        "2. You're using an App Password (not your regular password)\n"
                        "3. The email address is correct")
            else:
                logger.error(f"Authentication error: {error_msg}")
                return f"Authentication failed: {error_msg}"
            
        except smtplib.SMTPException as e:
            logger.error(f"SMTP error: {str(e)}")
            return f"SMTP error occurred: {str(e)}"
            
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"Error sending email: {str(e)}"

# Key Clauses Extractor Class
class LegalClauseExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  
        )

        self.clause_categories = [
            "payment terms", "termination", "confidentiality",
            "liability", "intellectual property", "warranty",
            "indemnification", "force majeure", "governing law",
            "dispute resolution"
        ]

        self.legal_terms_patterns = [
            r"shall|must|will", r"party|parties", r"agree|agreement",
            r"term|period", r"terminate|termination",
            r"confidential|confidentiality",
            r"liability|indemnify|indemnification",
            r"intellectual property|IP", r"warrant|warranty|warranties",
            r"law|jurisdiction|governing", r"dispute|arbitration|mediation"
        ]

    def extract_key_terms(self, text):
        doc = self.nlp(text)
        key_terms = []
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        for pattern in self.legal_terms_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group()
                if len(term) > 3:  
                    key_terms.append(term)

        return list(set(key_terms))

    def extract_clauses(self, text):
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        sentences = [sent for sent in sentences if len(sent) > 10]

        clauses = {}
        batch_size = 10  
        
        for category in self.clause_categories:
            relevant_sentences = []
            
            if not sentences:
                continue
                
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                try:
                    classification = self.classifier(
                        sequences=batch,
                        candidate_labels=[category],
                        multi_label=False
                    )
                    
                    if isinstance(classification, dict):
                        scores = classification['scores']
                        if not isinstance(scores, list):
                            scores = [scores]
                        for sent, score in zip(batch, scores):
                            if score > 0.7:
                                relevant_sentences.append(sent)
                    else:
                        for result, sent in zip(classification, batch):
                            if result['scores'][0] > 0.7:
                                relevant_sentences.append(sent)
                                
                except Exception as e:
                    print(f"Error processing batch for category {category}: {str(e)}")
                    continue
            
            if relevant_sentences:
                clauses[category] = relevant_sentences

        return clauses

    def analyze_document(self, text):
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

# Risk Detector Class
class LegalRiskDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1
        )
        
        self.risk_categories = ["high_risk", "medium_risk", "low_risk", "neutral"]
        
        self.risk_patterns = {
            "ambiguous_terms": [
                r"reasonable", r"substantial", r"material", r"appropriate",
                r"as necessary", r"subject to", r"deemed"
            ],
            "hidden_obligations": [
                r"shall", r"must", r"required to", r"obligated",
                r"responsible for"
            ],
            "conditional_clauses": [
                r"provided that", r"subject to", r"contingent upon",
                r"in the event", r"if and only if"
            ]
        }

    def analyze_risks(self, text):
        try:
            doc = self.nlp(text)
            
            # Risk Classification
            classification = self.classifier(
                sequences=text,
                candidate_labels=self.risk_categories,
                multi_label=True
            )
            
            risk_level = classification['labels'][0]
            confidence = classification['scores'][0]
            
            # Dependency Analysis
            dependencies = []
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ["advcl", "ccomp", "xcomp"]:
                        dependencies.append({
                            "main_clause": token.head.text,
                            "dependent_clause": token.text,
                            "dependency_type": token.dep_
                        })
            
            # Ambiguity Detection
            ambiguities = []
            for pattern_type, patterns in self.risk_patterns.items():
                for pattern in patterns:
                    matches = [sent.text for sent in doc.sents 
                             if pattern.lower() in sent.text.lower()]
                    if matches:
                        ambiguities.append({
                            "type": pattern_type,
                            "pattern": pattern,
                            "contexts": matches
                        })
            
            return {
                "risk_classification": {
                    "risk_level": risk_level,
                    "confidence": confidence
                },
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


class RegulatoryTracker:
    def __init__(self, db_path="regulatory_updates.db"):
        self.db_path = db_path
        self.setup_database()
        # Add feed URLs for regulatory updates
        self.feed_urls = [
            #"https://www.sec.gov/newsroom/press-releases"
            "https://gdpr-info.eu/",
            "https://www.sec.gov/news/pressreleases.rss",
            "https://www.ftc.gov/news-events/press-releases/feed",
            "https://www.justice.gov/feeds/opa/justice-news.xml"
        ]
        
    def setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS regulatory_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    publication_date DATETIME NOT NULL,
                    affected_terms TEXT,
                    impact_level TEXT,
                    processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    notification_sent BOOLEAN DEFAULT FALSE
                )
            """)
            
    def fetch_and_store_updates(self):
        """Fetch updates from RSS feeds and store them in the database"""
        try:
            for feed_url in self.feed_urls:
                feed = feedparser.parse(feed_url)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for entry in feed.entries[:5]:  # Get latest 5 entries
                        # Check if entry already exists
                        cursor.execute("""
                            SELECT id FROM regulatory_updates 
                            WHERE title = ? AND source = ?
                        """, (entry.title, feed.feed.title))
                        
                        if cursor.fetchone() is None:
                            # Simple impact level determination based on keywords
                            content = entry.summary if hasattr(entry, 'summary') else entry.title
                            impact_level = self._determine_impact_level(content)
                            
                            # Convert publication date to datetime
                            pub_date = datetime.strptime(
                                entry.published, '%a, %d %b %Y %H:%M:%S %z'
                            ) if hasattr(entry, 'published') else datetime.now()
                            
                            cursor.execute("""
                                INSERT INTO regulatory_updates 
                                (source, title, content, publication_date, impact_level)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                feed.feed.title,
                                entry.title,
                                content,
                                pub_date,
                                impact_level
                            ))
            
            return True
        except Exception as e:
            logging.error(f"Error fetching updates: {str(e)}")
            return False
    
    def _determine_impact_level(self, content):
        """Simple impact level determination based on keywords"""
        high_impact_keywords = ['urgent', 'critical', 'immediate', 'mandatory']
        medium_impact_keywords = ['important', 'should', 'recommended']
        
        content_lower = content.lower()
        
        if any(keyword in content_lower for keyword in high_impact_keywords):
            return 'HIGH'
        elif any(keyword in content_lower for keyword in medium_impact_keywords):
            return 'MEDIUM'
        return 'LOW'

    def get_recent_updates(self):
        """Get recent updates from the database"""
        try:
            # First fetch new updates
            self.fetch_and_store_updates()
            
            # Then retrieve from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                updates = cursor.execute("""
                    SELECT source, title, content, publication_date, impact_level
                    FROM regulatory_updates
                    ORDER BY publication_date DESC
                    LIMIT 5
                """).fetchall()
                
                if not updates:
                    # Add some sample data if no updates found
                    sample_updates = [
                        ("Sample Source", "Sample Update 1", "This is a sample regulatory update.", 
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "MEDIUM"),
                        ("Sample Source", "Sample Update 2", "Another sample regulatory update.", 
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "LOW")
                    ]
                    
                    for update in sample_updates:
                        cursor.execute("""
                            INSERT INTO regulatory_updates 
                            (source, title, content, publication_date, impact_level)
                            VALUES (?, ?, ?, ?, ?)
                        """, update)
                    
                    conn.commit()
                    updates = sample_updates
                
                return updates
                
        except Exception as e:
            logging.error(f"Error fetching updates: {str(e)}")
            return []
# Streamlit Application
def initialize_session_state():
    if 'filepath' not in st.session_state:
        st.session_state['filepath'] = None
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None
    if 'summary' not in st.session_state:
        st.session_state['summary'] = None
    if 'analysis' not in st.session_state:
        st.session_state['analysis'] = None
    if 'risk_analysis' not in st.session_state:
        st.session_state['risk_analysis'] = None
    if 'regulatory_updates' not in st.session_state:
        st.session_state['regulatory_updates'] = None
    if 'email_sent' not in st.session_state:
        st.session_state['email_sent'] = False
    if 'recipient_email' not in st.session_state:
        st.session_state['recipient_email'] = ''
    if 'sender_email' not in st.session_state:
        st.session_state['sender_email'] = ''
    if 'chatbot' not in st.session_state:
        st.session_state['chatbot'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    

def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def displayPDF(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(pages)
    return " ".join([doc.page_content for doc in texts])

def llm_pipeline(file_path):
    model_name = "LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    text = file_preprocessing(file_path)
    
    inputs = tokenizer.encode("summarize: " + text, 
                            return_tensors="pt", 
                            max_length=1024, 
                            truncation=True)
    
    summary_ids = model.generate(inputs, 
                               max_length=150, 
                               min_length=40, 
                               length_penalty=2.0, 
                               num_beams=4, 
                               early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def format_risk_analysis(analysis_results):
    output = "🚨 Risk Analysis Report\n\n"
    
    if analysis_results["status"] == "error":
        return f"Error in risk analysis: {analysis_results['error']}"
    
    risk_class = analysis_results["risk_classification"]
    output += f"Risk Level: {risk_class['risk_level']}\n"
    output += f"Confidence: {risk_class['confidence']:.2%}\n\n"
    
    output += "🔄 Dependencies:\n"
    output += "--------------\n"
    for dep in analysis_results["dependencies"]:
        output += f"• Main: {dep['main_clause']}\n"
        output += f"  Dependent: {dep['dependent_clause']}\n"
        output += f"  Type: {dep['dependency_type']}\n\n"
    
    output += "⚠️ Ambiguities:\n"
    output += "-------------\n"
    for amb in analysis_results["ambiguities"]:
        output += f"\n{amb['type'].title()}:\n"
        output += f"Pattern: '{amb['pattern']}'\n"
        output += "Contexts:\n"
        for context in amb["contexts"]:
            output += f"  • {context}\n"
    
    return output

def main():
    st.markdown("""
    <style>
        .stApp {
            background-color: #281811;  # Darker brown
        }
        /* Optional: Make the sidebar match */
        .css-1d391kg {
            background-color: #1E110C;  # Slightly darker for contrast
        }
    </style>
""", unsafe_allow_html=True)
    # Add custom styling
    st.title("⚖️ Legal Documents Analyzer")  
    
    initialize_session_state()
    
    # Initialize components
    clause_extractor = LegalClauseExtractor()
    risk_detector = LegalRiskDetector()
    regulatory_tracker = RegulatoryTracker()
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    uploaded_file = st.file_uploader("Upload your legal document PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file
        st.session_state['filepath'] = os.path.join("data", uploaded_file.name)  # Changed
        filepath = os.path.join("data", uploaded_file.name)
        with open(st.session_state['filepath'], "wb") as temp_file:
            temp_file.write(uploaded_file.read())
    #chatbot
    if st.session_state['chatbot'] is None and st.session_state['filepath'] is not None:
            try:

                st.session_state['chatbot'] = DocumentChatbot()
                input_text = file_preprocessing(st.session_state['filepath'])
                st.session_state['chatbot'].preprocess_document(input_text)
            except Exception as e:
                st.error(f"Error initializing chatbot: {str(e)}")

    if st.session_state['uploaded_file'] and st.session_state['filepath']:
        filepath = os.path.join("data", st.session_state['uploaded_file'].name)
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            st.info("Uploaded File")
            displayPDF(st.session_state['filepath'])
            
        with col2:
            tab1, tab2, tab3, tab4,tab5 = st.tabs([
                "📝 Summary", 
                "🔍 Key Clauses",
                "⚠️ Risk Analysis",
                "📊 Regulatory Updates",
                "💬 Chat"
            ])
            
            with tab1:
                if st.button("Generate Summary"):
                    with st.spinner("Processing Summarization..."):
                        try:
                            summary = llm_pipeline(st.session_state['filepath'])
                            st.session_state['summary'] = summary
                            st.success("Summarization Complete!")
                            st.text_area("Summary", summary, height=300)
                        except Exception as e:
                            st.error(f"Error during summarization: {str(e)}")
            
            with tab2:
                if st.button("Extract Key Clauses"):
                    with st.spinner("Analyzing key clauses..."):
                        try:
                            input_text = file_preprocessing(filepath)
                            analysis_results = clause_extractor.analyze_document(input_text)
                            
                            st.subheader("📑 Key Legal Terms")
                            if analysis_results['key_terms']:
                                st.write(", ".join(analysis_results['key_terms']))
                            else:
                                st.write("No key terms found.")
                            
                            st.subheader("📋 Key Clauses by Category")
                            for category, clauses in analysis_results['key_clauses'].items():
                                with st.expander(f"{category.title()}"):
                                    for clause in clauses:
                                        st.write(f"• {clause}")
                            
                            st.session_state['analysis'] = analysis_results
                            st.success("Analysis Complete!")
                        except Exception as e:
                            st.error(f"Error during clause analysis: {str(e)}")
            
            with tab3:
                if st.button("Analyze Risks"):
                    with st.spinner("Analyzing potential risks..."):
                        try:
                            input_text = file_preprocessing(filepath)
                            risk_analysis = risk_detector.analyze_risks(input_text)
                            formatted_analysis = format_risk_analysis(risk_analysis)
                            
                            st.session_state['risk_analysis'] = formatted_analysis
                            st.markdown(formatted_analysis)
                            if st.session_state.get('analysis'):
                                    st.subheader("📊 Document Analysis Visualization")
                                    visualization = create_simple_visualizations(
                                        st.session_state['analysis'],
                                        risk_analysis
                                    )
                                    st.image(visualization)
                            else:
                                st.warning("Please run 'Extract Key Clauses' first to generate visualizations")
                            
                            # Add visualization
                            st.subheader("📊 Document Analysis Visualization")
                            visualization = create_simple_visualizations(
                                st.session_state.get('analysis', {}),
                                risk_analysis
                            )
                            st.image(visualization)
                            
                            # Risk level indicator
                            risk_level = risk_analysis["risk_classification"]["risk_level"]
                            risk_color = {
                                "high_risk": "red",
                                "medium_risk": "yellow",
                                "low_risk": "green",
                                "neutral": "blue"
                            }.get(risk_level, "gray")
                            
                            st.markdown(f"""
                                <div style='padding: 10px; background-color: {risk_color}; 
                                color: black; border-radius: 5px; text-align: center;'>
                                    Risk Level: {risk_level.replace('_', ' ').title()}
                                </div>
                                """, unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"Error during risk analysis: {str(e)}")
            
            with tab4:
                if st.button("Check Regulatory Updates"):
                    with st.spinner("Fetching recent regulatory updates..."):
                        try:
                            updates = regulatory_tracker.get_recent_updates()
                            if updates:
                                st.session_state['regulatory_updates'] = updates
                                for update in updates:
                                    with st.expander(f"{update[0]} - {update[1]}"):
                                        st.write(f"**Date:** {update[3]}")
                                        st.write(f"**Impact Level:** {update[4]}")
                                        st.write(f"**Content:**\n{update[2]}")
                            else:
                                st.info("No recent regulatory updates found.")
                        except Exception as e:
                            st.error(f"Error fetching regulatory updates: {str(e)}")
            
            with tab5:
                st.subheader("Chat with your Document")

                # Display chat history
                for message in st.session_state['chat_history']:
                    role = message["role"]
                    content = message["content"]
                    
                    if role == "user":
                        st.write(f"You: {content}")
                    else:
                        st.write(f"Assistant: {content}")
                
                # Chat input
                user_question = st.text_input("Ask a question about your document:", key="chat_input")
                
                if st.button("Send", key="send_button"):
                    if user_question:
                        # Add user message to chat history
                        st.session_state['chat_history'].append({
                            "role": "user",
                            "content": user_question
                        })
                        
                        # Get chatbot response
                        response = st.session_state['chatbot'].chat(user_question)
                        
                        # Add assistant response to chat history
                        st.session_state['chat_history'].append({
                            "role": "assistant",
                            "content": response
                        })
                        
                        # Clear input
                        st.rerun()
                
                if st.button("Clear Chat History"):
                    st.session_state['chat_history'] = []
                    st.rerun()

    

            

        # Email functionality
        if (st.session_state.get('summary') or 
            st.session_state.get('analysis') or 
            st.session_state.get('risk_analysis')):
            
            st.subheader("Send Results via Email")
            
            with st.form(key='email_form'):
                recipient_email = st.text_input(
                    "Recipient Email",
                    value=st.session_state['recipient_email']
                )
                sender_email = st.text_input(
                    "Your Email",
                    value=st.session_state['sender_email']
                )
                sender_password = st.text_input("Your Email Password", type="password")
                
                include_summary = st.checkbox("Include Summary", value=True)
                include_analysis = st.checkbox("Include Key Clauses Analysis", value=True)
                include_risks = st.checkbox("Include Risk Analysis", value=True)
                include_updates = st.checkbox("Include Regulatory Updates", value=True)
                
                submit_button = st.form_submit_button("Send Email")
                
                if submit_button:
                    if not all([recipient_email, sender_email, sender_password]):
                        st.error("Please fill in all email fields.")
                    elif not is_valid_email(recipient_email) or not is_valid_email(sender_email):
                        st.error("Please enter valid email addresses.")
                    else:
                        st.session_state['recipient_email'] = recipient_email
                        st.session_state['sender_email'] = sender_email
                        
                        email_content = []
                        if include_summary and st.session_state.get('summary'):
                            email_content.append("DOCUMENT SUMMARY:\n\n" + st.session_state['summary'])
                        
                        if include_analysis and st.session_state.get('analysis'):
                            email_content.append("\nKEY CLAUSES ANALYSIS:\n" + 
                                              str(st.session_state['analysis']))
                        
                        if include_risks and st.session_state.get('risk_analysis'):
                            email_content.append("\nRISK ANALYSIS:\n" + 
                                              st.session_state['risk_analysis'])
                        
                        if include_updates and st.session_state.get('regulatory_updates'):
                            updates_text = "\nREGULATORY UPDATES:\n"
                            for update in st.session_state['regulatory_updates']:
                                updates_text += f"\n{update[0]} - {update[1]}\n"
                                updates_text += f"Date: {update[3]}\n"
                                updates_text += f"Impact: {update[4]}\n"
                            email_content.append(updates_text)
                        
                        full_content = "\n\n".join(email_content)
                        
                        with st.spinner("Sending email..."):
                            try:
                                email_sender = EmailSender()
                                result = email_sender.send_email(
                                    sender_email,
                                    sender_password,
                                    recipient_email,
                                    "Legal Document Analysis Results",
                                    full_content
                                )
                                if "successfully" in result:
                                    st.success(result)
                                    st.session_state['email_sent'] = True
                                else:
                                    st.error(result)
                            except Exception as e:
                                st.error(f"Error sending email: {str(e)}")

if __name__ == "__main__":
    main()




      
