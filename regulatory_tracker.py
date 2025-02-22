
# import feedparser
# import requests
# from bs4 import BeautifulSoup
# from datetime import datetime
# from transformers import pipeline
# import sqlite3
# import json
# import logging
# from typing import List, Dict
# import schedule
# import time
# import spacy

# class RegulatoryTracker:
#     def __init__(self, db_path: str = "regulatory_updates.db"):
#         # Initialize logger
#         self.logger = logging.getLogger(__name__)
#         self.logger.setLevel(logging.INFO)
        
#         # Add handler if none exists
#         if not self.logger.handlers:
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#             handler.setFormatter(formatter)
#             self.logger.addHandler(handler)
            
#         self.db_path = db_path
#         self.sources = {
#             "gdpr": "https://edpb.europa.eu/news/news_en.rss",
#             "ccpa": "https://oag.ca.gov/privacy/rss",
#             # Add more regulatory sources as needed
#         }
#         self.setup_database()
        
#     def setup_database(self):
#         """Initialize the database schema."""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 conn.execute("""
#                     CREATE TABLE IF NOT EXISTS regulatory_updates (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         source TEXT NOT NULL,
#                         title TEXT NOT NULL,
#                         content TEXT NOT NULL,
#                         publication_date DATETIME NOT NULL,
#                         affected_terms TEXT,
#                         impact_level TEXT,
#                         processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
#                         notification_sent BOOLEAN DEFAULT FALSE
#                     )
#                 """)
                
#                 conn.execute("""
#                     CREATE TABLE IF NOT EXISTS affected_contracts (
#                         update_id INTEGER,
#                         contract_id TEXT,
#                         impact_description TEXT,
#                         status TEXT DEFAULT 'pending',
#                         FOREIGN KEY (update_id) REFERENCES regulatory_updates(id)
#                     )
#                 """)
#             self.logger.info("Database setup completed successfully")
#         except Exception as e:
#             self.logger.error(f"Error setting up database: {str(e)}")
#             raise

#     def fetch_updates(self) -> List[Dict]:
#         """Fetch updates from all configured sources."""
#         updates = []
        
#         for source_name, feed_url in self.sources.items():
#             try:
#                 self.logger.info(f"Fetching updates from {source_name}")
#                 feed = feedparser.parse(feed_url)
                
#                 for entry in feed.entries:
#                     update = {
#                         'source': source_name,
#                         'title': entry.title,
#                         'content': entry.summary,
#                         'publication_date': datetime.strptime(
#                             entry.published, 
#                             '%a, %d %b %Y %H:%M:%S %z'
#                         ),
#                         'affected_terms': [],
#                         'impact_level': 'pending'
#                     }
#                     updates.append(update)
                
#                 self.logger.info(f"Successfully fetched {len(feed.entries)} updates from {source_name}")
                    
#             except Exception as e:
#                 self.logger.error(f"Error fetching updates from {source_name}: {str(e)}")
                
#         return updates

#     def analyze_update(self, update: Dict) -> Dict:
#         """Analyze the regulatory update for impact and affected terms."""
#         try:
#             self.logger.info(f"Analyzing update: {update['title']}")
            
#             # Initialize NLP classifier for impact analysis
#             classifier = pipeline(
#                 "zero-shot-classification",
#                 model="facebook/bart-large-mnli",
#                 device=-1
#             )
            
#             # Classify impact level
#             impact_result = classifier(
#                 update['content'],
#                 candidate_labels=["high_impact", "medium_impact", "low_impact"],
#                 multi_label=False
#             )
            
#             # Extract potentially affected legal terms
#             doc = spacy.load("en_core_web_sm")(update['content'])
#             legal_entities = [ent.text for ent in doc.ents if ent.label_ in ["LAW", "ORG"]]
            
#             update['impact_level'] = impact_result['labels'][0]
#             update['affected_terms'] = legal_entities
            
#             self.logger.info(f"Successfully analyzed update: {update['title']}")
#             return update
            
#         except Exception as e:
#             self.logger.error(f"Error analyzing update: {str(e)}")
#             raise

#     def store_update(self, update: Dict) -> int:
#         """Store the regulatory update in the database."""
#         try:
#             self.logger.info(f"Storing update: {update['title']}")
            
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
                
#                 cursor.execute("""
#                     INSERT INTO regulatory_updates 
#                     (source, title, content, publication_date, affected_terms, impact_level)
#                     VALUES (?, ?, ?, ?, ?, ?)
#                 """, (
#                     update['source'],
#                     update['title'],
#                     update['content'],
#                     update['publication_date'],
#                     json.dumps(update['affected_terms']),
#                     update['impact_level']
#                 ))
                
#                 last_id = cursor.lastrowid
#                 self.logger.info(f"Successfully stored update with ID: {last_id}")
#                 return last_id
                
#         except Exception as e:
#             self.logger.error(f"Error storing update: {str(e)}")
#             raise

#     def identify_affected_contracts(self, update_id: int, contracts: List[Dict]):
#         """Identify contracts affected by a regulatory update."""
#         try:
#             self.logger.info(f"Identifying affected contracts for update ID: {update_id}")
            
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
                
#                 # Get the update details
#                 update = cursor.execute(
#                     "SELECT affected_terms, content FROM regulatory_updates WHERE id = ?",
#                     (update_id,)
#                 ).fetchone()
                
#                 if not update:
#                     self.logger.warning(f"No update found with ID: {update_id}")
#                     return
                
#                 affected_terms = json.loads(update[0])
#                 update_content = update[1]
                
#                 affected_count = 0
#                 # Check each contract for potential impact
#                 for contract in contracts:
#                     # Simple term matching - can be enhanced with more sophisticated matching
#                     if any(term.lower() in contract['content'].lower() for term in affected_terms):
#                         cursor.execute("""
#                             INSERT INTO affected_contracts 
#                             (update_id, contract_id, impact_description)
#                             VALUES (?, ?, ?)
#                         """, (
#                             update_id,
#                             contract['id'],
#                             f"Contract contains terms affected by update: {', '.join(affected_terms)}"
#                         ))
#                         affected_count += 1
                
#                 self.logger.info(f"Identified {affected_count} affected contracts")
                
#         except Exception as e:
#             self.logger.error(f"Error identifying affected contracts: {str(e)}")
#             raise

#     def run_update_cycle(self):
#         """Run a complete update cycle."""
#         try:
#             self.logger.info("Starting update cycle")
            
#             # Fetch updates
#             updates = self.fetch_updates()
#             self.logger.info(f"Fetched {len(updates)} updates")
            
#             for update in updates:
#                 # Analyze update
#                 analyzed_update = self.analyze_update(update)
                
#                 # Store update
#                 update_id = self.store_update(analyzed_update)
                
#                 # TODO: Implement contract fetching logic
#                 contracts = []  # This should be replaced with actual contract fetching
                
#                 # Identify affected contracts
#                 self.identify_affected_contracts(update_id, contracts)
            
#             self.logger.info("Update cycle completed successfully")
                
#         except Exception as e:
#             self.logger.error(f"Error in update cycle: {str(e)}")
#             raise

#     def fetch_and_store_updates(self) -> bool:
#         """Fetch new updates and store them in the database."""
#         try:
#             self.logger.info("Starting fetch and store operation")
            
#             updates = self.fetch_updates()
#             stored_count = 0
            
#             for update in updates:
#                 analyzed_update = self.analyze_update(update)
#                 if self.store_update(analyzed_update):
#                     stored_count += 1
            
#             self.logger.info(f"Successfully stored {stored_count} new updates")
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Error in fetch_and_store_updates: {str(e)}")
#             return False

#     def get_recent_updates(self, limit: int = 10) -> List[Dict]:
#         """Get recent updates from the database with improved error handling."""
#         try:
#             self.logger.info(f"Retrieving {limit} recent updates")
            
#             # First fetch new updates
#             self.fetch_and_store_updates()
            
#             # Then retrieve from database
#             with sqlite3.connect(self.db_path) as conn:
#                 conn.row_factory = sqlite3.Row
#                 cursor = conn.cursor()
                
#                 updates = cursor.execute("""
#                     SELECT 
#                         id,
#                         source,
#                         title,
#                         content,
#                         publication_date,
#                         impact_level,
#                         affected_terms,
#                         notification_sent
#                     FROM regulatory_updates
#                     ORDER BY publication_date DESC
#                     LIMIT ?
#                 """, (limit,)).fetchall()
                
#                 result = [dict(update) for update in updates]
#                 self.logger.info(f"Retrieved {len(result)} updates")
#                 return result
                
#         except Exception as e:
#             self.logger.error(f"Error fetching recent updates: {str(e)}")
#             return []


# def schedule_updates():
#     """Schedule regular update checks."""
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     )
#     logger = logging.getLogger(__name__)
    
#     try:
#         logger.info("Initializing regulatory tracker")
#         tracker = RegulatoryTracker()
        
#         # Schedule daily updates
#         schedule.every().day.at("00:00").do(tracker.run_update_cycle)
#         logger.info("Scheduled daily updates at 00:00")
        
#         while True:
#             schedule.run_pending()
#             time.sleep(3600)  # Check every hour
            
#     except Exception as e:
#         logger.error(f"Error in schedule_updates: {str(e)}")
#         raise


# if __name__ == "__main__":
#     schedule_updates()


# import feedparser
# import requests
# from bs4 import BeautifulSoup
# from datetime import datetime
# from transformers import pipeline
# import sqlite3
# import json
# import logging
# from typing import List, Dict
# import schedule
# import time
# import spacy

# class RegulatoryTracker:
#     def __init__(self, db_path: str = "regulatory_updates.db"):
#         # Initialize logger
#         self.logger = logging.getLogger(__name__)
#         self.logger.setLevel(logging.INFO)
        
#         # Add handler if none exists
#         if not self.logger.handlers:
#             handler = logging.StreamHandler()
#             formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#             handler.setFormatter(formatter)
#             self.logger.addHandler(handler)
            
#         self.db_path = db_path
#         self.sources = {
#             "gdpr": "https://edpb.europa.eu/news/news_en.rss",
#             "ccpa": "https://oag.ca.gov/privacy/rss",
#             # Add more regulatory sources as needed
#         }
#         self.setup_database()
        
#     def setup_database(self):
#         """Initialize the database schema."""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 conn.execute("""
#                     CREATE TABLE IF NOT EXISTS regulatory_updates (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         source TEXT NOT NULL,
#                         title TEXT NOT NULL,
#                         content TEXT NOT NULL,
#                         publication_date DATETIME NOT NULL,
#                         affected_terms TEXT,
#                         impact_level TEXT,
#                         processed_date DATETIME DEFAULT CURRENT_TIMESTAMP,
#                         notification_sent BOOLEAN DEFAULT FALSE
#                     )
#                 """)
                
#                 conn.execute("""
#                     CREATE TABLE IF NOT EXISTS affected_contracts (
#                         update_id INTEGER,
#                         contract_id TEXT,
#                         impact_description TEXT,
#                         status TEXT DEFAULT 'pending',
#                         FOREIGN KEY (update_id) REFERENCES regulatory_updates(id)
#                     )
#                 """)
#             self.logger.info("Database setup completed successfully")
#         except Exception as e:
#             self.logger.error(f"Error setting up database: {str(e)}")
#             raise

#     def fetch_updates(self) -> List[Dict]:
#         """Fetch updates from all configured sources."""
#         updates = []
        
#         for source_name, feed_url in self.sources.items():
#             try:
#                 self.logger.info(f"Fetching updates from {source_name}")
#                 feed = feedparser.parse(feed_url)
                
#                 for entry in feed.entries:
#                     update = {
#                         'source': source_name,
#                         'title': entry.title,
#                         'content': entry.summary,
#                         'publication_date': datetime.strptime(
#                             entry.published, 
#                             '%a, %d %b %Y %H:%M:%S %z'
#                         ),
#                         'affected_terms': [],
#                         'impact_level': 'pending'
#                     }
#                     updates.append(update)
                
#                 self.logger.info(f"Successfully fetched {len(feed.entries)} updates from {source_name}")
                    
#             except Exception as e:
#                 self.logger.error(f"Error fetching updates from {source_name}: {str(e)}")
                
#         return updates
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import pipeline
import sqlite3
import json
import logging
from typing import List, Dict
import schedule
import time
import spacy

class RegulatoryTracker:
    def __init__(self, db_path: str = "regulatory_updates.db"):
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.db_path = db_path
        # Updated RSS feed sources with more reliable endpoints
        self.sources = {
            
            # "privacy": "https://www.privacy.gov.ph/feed/",
            # "ico": "https://ico.org.uk/rss/news",
            # "nist": "https://www.nist.gov/news-events/news/rss.xml",
            # "europa": "https://european-union.europa.eu/news-feed.xml",
        }
        self.setup_database()
    def setup_database(self):
        """Initialize the database schema."""
        try:
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
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS affected_contracts (
                        update_id INTEGER,
                        contract_id TEXT,
                        impact_description TEXT,
                        status TEXT DEFAULT 'pending',
                        FOREIGN KEY (update_id) REFERENCES regulatory_updates(id)
                    )
                    
                """)
            self.logger.info("Database setup completed successfully")
        except Exception as e:
            self.logger.error(f"Error setting up database: {str(e)}")
            raise

    def fetch_updates(self) -> List[Dict]:
        """Fetch updates from all configured sources with improved error handling."""
        updates = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for source_name, feed_url in self.sources.items():
            try:
                self.logger.info(f"Fetching updates from {source_name} at {feed_url}")
                
                # First try direct feedparser
                feed = feedparser.parse(feed_url)
                
                if feed.status != 200:
                    # If direct parsing fails, try using requests
                    response = requests.get(feed_url, headers=headers, timeout=15)
                    response.raise_for_status()
                    feed = feedparser.parse(response.text)
                
                if hasattr(feed, 'bozo') and feed.bozo:
                    self.logger.warning(f"Feed error for {source_name}: {feed.bozo_exception}")
                    continue
                
                if not hasattr(feed, 'entries') or not feed.entries:
                    self.logger.warning(f"No entries found in feed from {source_name}")
                    # Try parsing as HTML if RSS fails
                    response = requests.get(feed_url, headers=headers, timeout=15)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # Look for recent news/updates in HTML
                    news_items = soup.find_all(['article', 'div'], class_=['news-item', 'post', 'entry'])
                    
                    for item in news_items[:5]:  # Get up to 5 recent items
                        title = item.find(['h1', 'h2', 'h3'])
                        content = item.find(['p', 'div'], class_=['content', 'summary', 'description'])
                        if title and content:
                            updates.append({
                                'source': source_name,
                                'title': title.get_text().strip(),
                                'content': content.get_text().strip(),
                                'publication_date': datetime.now(),
                                'affected_terms': [],
                                'impact_level': 'pending'
                            })
                    continue
                
                for entry in feed.entries:
                    try:
                        # Handle different date formats
                        pub_date = None
                        date_formats = [
                            '%a, %d %b %Y %H:%M:%S %z',
                            '%Y-%m-%dT%H:%M:%S%z',
                            '%Y-%m-%d %H:%M:%S',
                            '%a, %d %b %Y %H:%M:%S GMT',
                            '%Y-%m-%d'
                        ]
                        
                        for date_format in date_formats:
                            try:
                                pub_date = datetime.strptime(
                                    getattr(entry, 'published', entry.get('updated', '')), 
                                    date_format
                                )
                                break
                            except (ValueError, AttributeError):
                                continue
                        
                        if pub_date is None:
                            pub_date = datetime.now()
                            self.logger.warning(f"Could not parse date for entry in {source_name}, using current time")
                        
                        update = {
                            'source': source_name,
                            'title': getattr(entry, 'title', 'No Title'),
                            'content': getattr(entry, 'summary', 
                                      getattr(entry, 'description', 
                                      getattr(entry, 'content', 'No Content'))),
                            'publication_date': pub_date,
                            'affected_terms': [],
                            'impact_level': 'pending'
                        }
                        
                        # Clean content if it's a list or dict
                        if isinstance(update['content'], (list, dict)):
                            update['content'] = str(update['content'])
                        
                        updates.append(update)
                        self.logger.debug(f"Successfully processed entry: {update['title']}")
                        
                    except Exception as entry_error:
                        self.logger.error(f"Error processing entry from {source_name}: {str(entry_error)}")
                        continue
                
                self.logger.info(f"Successfully fetched {len(feed.entries)} updates from {source_name}")
                    
            except requests.RequestException as e:
                self.logger.error(f"Network error fetching updates from {source_name}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error fetching from {source_name}: {str(e)}")
                
        self.logger.info(f"Total updates fetched across all sources: {len(updates)}")
        return updates

    # Rest of your methods remain the same...

    def analyze_update(self, update: Dict) -> Dict:
        """Analyze the regulatory update for impact and affected terms."""
        try:
            self.logger.info(f"Analyzing update: {update['title']}")
            
            # Initialize NLP classifier for impact analysis
            classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            
            # Classify impact level
            impact_result = classifier(
                update['content'],
                candidate_labels=["high_impact", "medium_impact", "low_impact"],
                multi_label=False
            )
            
            # Extract potentially affected legal terms
            doc = spacy.load("en_core_web_sm")(update['content'])
            legal_entities = [ent.text for ent in doc.ents if ent.label_ in ["LAW", "ORG"]]
            
            update['impact_level'] = impact_result['labels'][0]
            update['affected_terms'] = legal_entities
            
            self.logger.info(f"Successfully analyzed update: {update['title']}")
            return update
            
        except Exception as e:
            self.logger.error(f"Error analyzing update: {str(e)}")
            raise

    def store_update(self, update: Dict) -> int:
        """Store the regulatory update in the database."""
        try:
            self.logger.info(f"Storing update: {update['title']}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO regulatory_updates 
                    (source, title, content, publication_date, affected_terms, impact_level)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    update['source'],
                    update['title'],
                    update['content'],
                    update['publication_date'],
                    json.dumps(update['affected_terms']),
                    update['impact_level']
                ))
                
                last_id = cursor.lastrowid
                self.logger.info(f"Successfully stored update with ID: {last_id}")
                return last_id
                
        except Exception as e:
            self.logger.error(f"Error storing update: {str(e)}")
            raise

    def identify_affected_contracts(self, update_id: int, contracts: List[Dict]):
        """Identify contracts affected by a regulatory update."""
        try:
            self.logger.info(f"Identifying affected contracts for update ID: {update_id}")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the update details
                update = cursor.execute(
                    "SELECT affected_terms, content FROM regulatory_updates WHERE id = ?",
                    (update_id,)
                ).fetchone()
                
                if not update:
                    self.logger.warning(f"No update found with ID: {update_id}")
                    return
                
                affected_terms = json.loads(update[0])
                update_content = update[1]
                
                affected_count = 0
                # Check each contract for potential impact
                for contract in contracts:
                    # Simple term matching - can be enhanced with more sophisticated matching
                    if any(term.lower() in contract['content'].lower() for term in affected_terms):
                        cursor.execute("""
                            INSERT INTO affected_contracts 
                            (update_id, contract_id, impact_description)
                            VALUES (?, ?, ?)
                        """, (
                            update_id,
                            contract['id'],
                            f"Contract contains terms affected by update: {', '.join(affected_terms)}"
                        ))
                        affected_count += 1
                
                self.logger.info(f"Identified {affected_count} affected contracts")
                
        except Exception as e:
            self.logger.error(f"Error identifying affected contracts: {str(e)}")
            raise

    def run_update_cycle(self):
        """Run a complete update cycle."""
        try:
            self.logger.info("Starting update cycle")
            
            # Fetch updates
            updates = self.fetch_updates()
            self.logger.info(f"Fetched {len(updates)} updates")
            
            for update in updates:
                # Analyze update
                analyzed_update = self.analyze_update(update)
                
                # Store update
                update_id = self.store_update(analyzed_update)
                
                # TODO: Implement contract fetching logic
                contracts = []  # This should be replaced with actual contract fetching
                
                # Identify affected contracts
                self.identify_affected_contracts(update_id, contracts)
            
            self.logger.info("Update cycle completed successfully")
                
        except Exception as e:
            self.logger.error(f"Error in update cycle: {str(e)}")
            raise

    def fetch_and_store_updates(self) -> bool:
        """Fetch new updates and store them in the database."""
        try:
            self.logger.info("Starting fetch and store operation")
            
            updates = self.fetch_updates()
            stored_count = 0
            
            for update in updates:
                analyzed_update = self.analyze_update(update)
                if self.store_update(analyzed_update):
                    stored_count += 1
            
            self.logger.info(f"Successfully stored {stored_count} new updates")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in fetch_and_store_updates: {str(e)}")
            return False

    # def get_recent_updates(self, limit: int = 10) -> List[Dict]:
    #     """Get recent updates from the database with improved error handling."""
    #     try:
    #         self.logger.info(f"Retrieving {limit} recent updates")
            
    #         # First fetch new updates
    #         self.fetch_and_store_updates()
            
    #         # Then retrieve from database
    #         with sqlite3.connect(self.db_path) as conn:
    #             conn.row_factory = sqlite3.Row
    #             cursor = conn.cursor()
                
    #             updates = cursor.execute("""
    #                 SELECT 
    #                     id,
    #                     source,
    #                     title,
    #                     content,
    #                     publication_date,
    #                     impact_level,
    #                     affected_terms,
    #                     notification_sent
    #                 FROM regulatory_updates
    #                 ORDER BY publication_date DESC
    #                 LIMIT ?
    #             """, (limit,)).fetchall()
                
    #             result = [dict(update) for update in updates]
    #             self.logger.info(f"Retrieved {len(result)} updates")
    #             return result
                
    #     except Exception as e:
    #         self.logger.error(f"Error fetching recent updates: {str(e)}")
    #         return []

    def get_recent_updates(self, limit: int = 10) -> List[Dict]:
            """Get recent updates from the database with improved Streamlit compatibility."""
            try:
                self.logger.info(f"Retrieving {limit} recent updates")
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    
                    updates = cursor.execute("""
                        SELECT 
                            id,
                            source,
                            title,
                            content,
                            publication_date,
                            impact_level,
                            affected_terms,
                            notification_sent
                        FROM regulatory_updates
                        ORDER BY publication_date DESC
                        LIMIT ?
                    """, (limit,)).fetchall()
                    
                    if not updates:
                        self.logger.warning("No updates found in database")
                        return []
                    
                    formatted_updates = []
                    for update in updates:
                        try:
                            update_dict = dict(update)
                            # Ensure affected_terms is properly formatted
                            if isinstance(update_dict['affected_terms'], str):
                                update_dict['affected_terms'] = json.loads(update_dict['affected_terms'])
                            # Format date string
                            if isinstance(update_dict['publication_date'], str):
                                try:
                                    update_dict['publication_date'] = datetime.strptime(
                                        update_dict['publication_date'],
                                        '%Y-%m-%d %H:%M:%S'
                                    ).strftime('%Y-%m-%d %H:%M:%S')
                                except ValueError:
                                    update_dict['publication_date'] = str(update_dict['publication_date'])
                            formatted_updates.append(update_dict)
                        except Exception as e:
                            self.logger.error(f"Error formatting update: {str(e)}")
                            continue
                    
                    self.logger.info(f"Retrieved and formatted {len(formatted_updates)} updates")
                    return formatted_updates
                    
            except Exception as e:
                self.logger.error(f"Error fetching recent updates: {str(e)}")
                return []

    def get_update_summary(self) -> Dict:
        """Get a summary of updates for Streamlit display."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total count
                total = cursor.execute("SELECT COUNT(*) FROM regulatory_updates").fetchone()[0]
                
                # Get count by impact level
                impact_levels = cursor.execute("""
                    SELECT impact_level, COUNT(*) as count 
                    FROM regulatory_updates 
                    GROUP BY impact_level
                """).fetchall()
                
                # Get count by source
                sources = cursor.execute("""
                    SELECT source, COUNT(*) as count 
                    FROM regulatory_updates 
                    GROUP BY source
                """).fetchall()
                
                return {
                    "total_updates": total,
                    "impact_levels": dict(impact_levels),
                    "sources": dict(sources)
                }
        except Exception as e:
            self.logger.error(f"Error getting update summary: {str(e)}")
            return {
                "total_updates": 0,
                "impact_levels": {},
                "sources": {}
            }

def schedule_updates():
    """Schedule regular update checks."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing regulatory tracker")
        tracker = RegulatoryTracker()
        
        # Schedule daily updates
        schedule.every().day.at("00:00").do(tracker.run_update_cycle)
        logger.info("Scheduled daily updates at 00:00")
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
            
    except Exception as e:
        logger.error(f"Error in schedule_updates: {str(e)}")
        raise


if __name__ == "__main__":
    schedule_updates()
    # tracker = RegulatoryTracker()
    # updates = tracker.get_recent_updates()
    # print(f"\nFound {len(updates)} updates:")
    # for update in updates:
    #     print(f"\nTitle: {update['title']}")
    #     print(f"Source: {update['source']}")
    #     print(f"Date: {update['publication_date']}")
    #     print("-" * 50)

