import os.path
import base64
import csv
from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def gmail_authenticate():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def clean_text(text):
    return text.replace('\r', '').replace('\n', ' ').strip()

def get_emails(service, max_results=100):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    
    email_data = []
    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        headers = msg_data['payload']['headers']
        subject = sender = date = ''
        
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']
            elif header['name'] == 'Date':
                date = header['value']
        
        try:
            parts = msg_data['payload'].get('parts')[0]
            body = parts['body'].get('data')
            if body:
                decoded_body = base64.urlsafe_b64decode(body).decode('utf-8')
                soup = BeautifulSoup(decoded_body, "html.parser")
                body_text = soup.get_text()
            else:
                body_text = ""
        except:
            body_text = ""
        
        email_data.append({
            'Date': date,
            'From': sender,
            'Subject': clean_text(subject),
            'Body': clean_text(body_text)
        })
    
    return email_data

def save_to_csv(email_data, filename='emails.csv'):
    keys = ['Date', 'From', 'Subject', 'Body']
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(email_data)
    print(f"Saved {len(email_data)} emails to {filename}")

if __name__ == '__main__':
    print("Start to crew Gmail...")
    service = gmail_authenticate()
    emails = get_emails(service, max_results=500)
    save_to_csv(emails)