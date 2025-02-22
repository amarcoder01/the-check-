# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import ssl
# import logging

# def send_email(sender_email, sender_password, recipient_email, subject, body):
#     """
#     Sends an email using the provided credentials and recipient details.

#     Args:
#         sender_email (str): The sender's email address
#         sender_password (str): The sender's email password or app-specific password
#         recipient_email (str): The recipient's email address
#         subject (str): The subject of the email
#         body (str): The body of the email

#     Returns:
#         str: Success or error message
#     """
#     try:
        
#         logging.basicConfig(level=logging.INFO)
#         logger = logging.getLogger(__name__)
        
#         logger.info(f"Attempting to connect to SMTP server for: {sender_email}")
        
        
#         msg = MIMEMultipart()
#         msg["From"] = sender_email
#         msg["To"] = recipient_email
#         msg["Subject"] = subject
        
        
#         msg.attach(MIMEText(body, "plain"))

        
#         context = ssl.create_default_context()

        
#         with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
#             logger.info("Connected to SMTP server, attempting login...")
#             server.login(sender_email, sender_password)
#             logger.info("Login successful, sending message...")
#             server.send_message(msg)
            
#         return "Email sent successfully!"
    
#     except smtplib.SMTPAuthenticationError as e:
#         error_msg = str(e)
#         if "Application-specific password required" in error_msg:
#             return ("Authentication failed: You need to use an App Password. "
#                    "Please go to Google Account > Security > App Passwords to generate one.")
#         elif "Username and Password not accepted" in error_msg:
#             return ("Authentication failed: Username and Password not accepted. "
#                    "If using Gmail, please ensure:\n"
#                    "1. You've enabled 2-Step Verification\n"
#                    "2. You're using an App Password (not your regular password)\n"
#                    "3. The email address is correct")
#         else:
#             logger.error(f"Authentication error: {error_msg}")
#             return f"Authentication failed: {error_msg}"
    
#     except smtplib.SMTPException as e:
#         logger.error(f"SMTP error: {str(e)}")
#         return f"SMTP error occurred: {str(e)}"
    
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         return f"Error sending email: {str(e)}"

#to make as pdf
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import smtplib
import ssl
import logging
from fpdf import FPDF
from datetime import datetime
import json

def format_key_clauses(key_clauses):
    """
    Formats the key clauses dictionary into a readable structure.
    
    Args:
        key_clauses (dict): Raw key clauses dictionary
    
    Returns:
        dict: Formatted key clauses
    """
    formatted = {}
    try:
        if isinstance(key_clauses, str):
            key_clauses = json.loads(key_clauses)
            
        for category, content in key_clauses.items():
            if category == 'key_terms':
                formatted['Key Terms'] = content
            elif category == 'key_clauses':
                formatted.update({
                    k.replace('_', ' ').title(): v 
                    for k, v in content.items()
                })
    except (json.JSONDecodeError, AttributeError):
        formatted['Raw Data'] = str(key_clauses)
    
    return formatted

def create_pdf(summary, key_clauses, risks, regulatory_updates, chat_history):
    """
    Creates a PDF report containing the analysis results.
    """
    pdf = FPDF()
    pdf.add_page()
    
    # Set up styling
    pdf.set_font('Arial', 'B', 16)
    pdf.set_text_color(0, 0, 0)
    
    # Title
    pdf.cell(0, 10, 'Legal Document Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Date
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'R')
    pdf.ln(10)
    
    # Summary Section
    if summary:
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Document Summary', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, str(summary))
        pdf.ln(10)
    
    # Key Clauses Section
    if key_clauses:
        formatted_clauses = format_key_clauses(key_clauses)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Key Clauses Analysis', 0, 1, 'L')
        
        for category, clauses in formatted_clauses.items():
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, str(category), 0, 1, 'L')
            pdf.set_font('Arial', '', 12)
            
            if isinstance(clauses, list):
                for clause in clauses:
                    pdf.multi_cell(0, 10, f'• {str(clause)}')
            else:
                # Handle string or other types
                pdf.multi_cell(0, 10, str(clauses))
            pdf.ln(5)
        pdf.ln(10)
    
    # Risk Analysis Section
    if risks:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Risk Analysis', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, str(risks))
        pdf.ln(10)
    
    # Regulatory Updates Section
    if regulatory_updates:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Regulatory Updates', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        for update in regulatory_updates:
            if isinstance(update, (list, tuple)) and len(update) >= 5:
                pdf.set_font('Arial', 'B', 12)
                pdf.multi_cell(0, 10, f'{update[0]} - {update[1]}')
                pdf.set_font('Arial', '', 12)
                pdf.multi_cell(0, 10, f'Date: {update[3]}\nImpact: {update[4]}\n{update[2]}')
                pdf.ln(5)
            else:
                pdf.multi_cell(0, 10, str(update))
    
    # Chat History Section
    if chat_history:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Chat History', 0, 1, 'L')
        pdf.set_font('Arial', '', 12)
        for message in chat_history:
            if isinstance(message, dict):
                role = message.get('role', '')
                content = message.get('content', '')
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, f'{role.title()}:', 0, 1, 'L')
                pdf.set_font('Arial', '', 12)
                pdf.multi_cell(0, 10, str(content))
                pdf.ln(5)
            else:
                pdf.multi_cell(0, 10, str(message))
    
    return pdf.output(dest='S').encode('latin-1')

# Rest of the code remains the same...
def send_email(sender_email, sender_password, recipient_email, subject, summary, key_clauses=None, 
               risks=None, regulatory_updates=None, chat_history=None):
    """
    Sends an email with PDF attachment containing the analysis results.
    
    Args:
        sender_email (str): The sender's email address
        sender_password (str): The sender's email password or app-specific password
        recipient_email (str): The recipient's email address
        subject (str): The subject of the email
        summary (str): Document summary
        key_clauses (dict, optional): Key clauses analysis
        risks (str, optional): Risk analysis
        regulatory_updates (list, optional): List of regulatory updates
        chat_history (list, optional): Chat conversation history
    
    Returns:
        str: Success or error message
    """
    try:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        logger.info(f"Attempting to connect to SMTP server for: {sender_email}")
        
        # Create the email
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        
        # Add email body
        body = "Please find attached the Legal Document Analysis Report."
        msg.attach(MIMEText(body, "plain"))
        
        # Create and attach PDF
        pdf_data = create_pdf(summary, key_clauses, risks, regulatory_updates, chat_history)
        pdf_attachment = MIMEApplication(pdf_data, _subtype="pdf")
        pdf_attachment.add_header('Content-Disposition', 'attachment', 
                                filename="Legal_Document_Analysis.pdf")
        msg.attach(pdf_attachment)
        
        # Send the email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            logger.info("Connected to SMTP server, attempting login...")
            server.login(sender_email, sender_password)
            logger.info("Login successful, sending message...")
            server.send_message(msg)
        
        return "Email sent successfully with PDF attachment!"
        
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
