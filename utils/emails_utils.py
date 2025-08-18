# utils/emails_utils.py
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from typing import Optional

# -----------------------------------------------------------------------------
# Logger for this utility
# -----------------------------------------------------------------------------
logger = logging.getLogger("EmailsUtils")

# -----------------------------------------------------------------------------
# Main function to send emails via SMTP
# -----------------------------------------------------------------------------
def send_email(
    recipient: str,
    subject: str,
    body: str,
    sender_email: str,
    sender_password: str,
    html: bool = False,
    smtp_server: str = "smtp.gmail.com",
    smtp_port: int = 587,
    timeout: int = 10,
    sender_name: Optional[str] = None,
    use_tls: bool = True
) -> str:
    """
    Send an email using SMTP with optional HTML support.

    Args:
        recipient (str): Recipient's email address.
        subject (str): Subject line text.
        body (str): Email content (HTML or plain text).
        sender_email (str): Sender's email address.
        sender_password (str): SMTP/app password for the sender's account.
        html (bool): If True, sends an HTML email (with plain-text fallback).
        smtp_server (str): SMTP server hostname (default is Gmail's).
        smtp_port (int): SMTP server port (default 587 for TLS).
        timeout (int): Timeout in seconds for the SMTP connection.
        sender_name (Optional[str]): Human-readable name for sender (optional).
        use_tls (bool): Whether to start a TLS-secured connection (recommended).

    Returns:
        str: Status message indicating success or cause of failure.
    """
    try:
        # Create the appropriate type of multipart message:
        # - "alternative" allows for multiple formats (e.g., plain + HTML)
        msg = MIMEMultipart("alternative") if html else MIMEMultipart()

        # Set standard email headers
        msg['From'] = formataddr((sender_name or "", sender_email))
        msg['To'] = recipient
        msg['Subject'] = subject

        # Attach body content depending on format
        if html:
            # Provide a plain text fallback for clients that cannot display HTML
            plain_version = "This email contains HTML content. Please enable HTML view."
            msg.attach(MIMEText(plain_version, "plain"))
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        # Connect to the SMTP server
        with smtplib.SMTP(smtp_server, smtp_port, timeout=timeout) as server:
            if use_tls:
                # Start a secured TLS connection
                server.starttls()
            # Authenticate with the SMTP server
            server.login(sender_email, sender_password)
            # Send the built message
            server.send_message(msg)

        logger.info(f"Email sent to {recipient} with subject: {subject}")
        return "Email sent successfully!"

    except smtplib.SMTPAuthenticationError:
        logger.error("Authentication failed. Check your email and app password.")
        return "Email sending failed: Authentication error."
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error occurred: {e}")
        return f"Email sending failed: SMTP error: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}")
        return f"Email sending failed: {str(e)}"
