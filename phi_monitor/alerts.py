# phi_monitor/alerts.py

import logging
from typing import List, Optional
import smtplib
from email.mime.text import MIMEText
import requests
import os

class AlertManager:
    def __init__(self, verbose: bool = False):
        """
        Initialize the AlertManager.

        Parameters:
        - verbose: Whether to enable detailed logging.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        
        # Configure logger
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def trigger_alert(self, score: float, threshold: float, alert_type: str, alert_methods: Optional[List[str]] = None, message: str = ""):
        """
        Trigger alerts for drift or overfitting detection.

        Parameters:
        - score: The score that triggered the alert.
        - threshold: The threshold value.
        - alert_type: The type of alert (e.g., "drift", "overfit").
        - alert_methods: A list of methods to send the alert (e.g., ["email", "slack"]).
        - message: Custom message for the alert.
        """
        if alert_methods is None:
            alert_methods = ["console"]

        alert_message = (
            f"[{alert_type.upper()} ALERT] {message} "
            f"Score: {score:.4f} exceeds threshold: {threshold:.4f}"
        )

        for method in alert_methods:
            if method == "console":
                print(alert_message)
            elif method == "log":
                self.logger.warning(alert_message)
            elif method == "email":
                self.send_email_alert(
                    subject=f"{alert_type.capitalize()} Alert",
                    body=alert_message,
                    to_email=os.getenv("ALERT_EMAIL_RECIPIENT"),
                    from_email=os.getenv("ALERT_EMAIL_SENDER"),
                    smtp_server=os.getenv("SMTP_SERVER"),
                    smtp_port=int(os.getenv("SMTP_PORT", 465)),
                    smtp_user=os.getenv("SMTP_USERNAME"),
                    smtp_password=os.getenv("SMTP_PASSWORD"),
                )
            elif method == "slack":
                self.send_slack_alert(
                    webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
                    message=alert_message
                )
            else:
                self.logger.error(f"Unsupported alert method: {method}")

    def send_email_alert(
        self,
        subject: str,
        body: str,
        to_email: str,
        from_email: str,
        smtp_server: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
    ):
        """
        Sends an email alert.

        Parameters:
        - subject: Subject of the email.
        - body: Body of the email.
        - to_email: Recipient email address.
        - from_email: Sender email address.
        - smtp_server: SMTP server address.
        - smtp_port: SMTP port.
        - smtp_user: SMTP username.
        - smtp_password: SMTP password.
        """
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_email

        try:
            with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
            self.logger.info(f"Email alert sent to {to_email}.")
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def send_slack_alert(self, webhook_url: str, message: str):
        """
        Sends a Slack alert.

        Parameters:
        - webhook_url: Slack Incoming Webhook URL.
        - message: Message to send.
        """
        payload = {"text": message}
        try:
            response = requests.post(webhook_url, json=payload)
            if response.status_code != 200:
                raise ValueError(
                    f"Request to Slack returned an error {response.status_code}, "
                    f"response: {response.text}"
                )
            self.logger.info("Slack alert sent successfully.")
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
