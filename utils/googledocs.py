from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import numpy as np
import time

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/documents', 'https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']



def create_summary_doc(title, context, summary, prompt):
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    try:
        service = build('docs', 'v1', credentials=creds)
        drive_service = build('drive', 'v3', credentials=creds)


        #Create the document with the right title
        document = service.documents().create(body={"title":title}).execute()
        docid = document.get('documentId')
        ## Write 'backwards' for simplicity
        lengths = [len("Context\n")+1, len(f"{context}\n"), len("Summary\n"), len(f"{summary}\n"), len("Prompt\n"), len(f"{prompt}\n")]
        indices = np.cumsum(lengths)
        insertion_request = [
            {
                'insertText': {
                    'location': {
                        'index' : 1
                    },
                    'text' : "Context\n"
                }
            },
                        {
                'insertText': {
                    'location': {
                        'index' : int(indices[0])
                    },
                    'text': f"{context}\n\n"
                }
            },
            {
                'insertText': {
                    'location': {
                        'index' : int(indices[1])
                    },
                    'text' : "Summary\n\n"
                }
            },
            {
                'insertText': {
                    'location': {
                        'index' : int(indices[2])
                    },
                    'text': f"{summary}\n\n"
                }
            },
            {
                'insertText': {
                    'location': {
                        'index' : int(indices[3])
                    },
                    'text' : "Prompt\n"
                }
            },
            {
                'insertText': {
                    'location': {
                        'index' : int(indices[4])
                    },
                    'text': f"{prompt}\n"
                }
            },
            {
                'updateParagraphStyle' : {
                    'range' : {
                        'startIndex': 1,
                        'endIndex': int(indices[0])
                    },
                    'paragraphStyle': {
                        'namedStyleType': 'HEADING_1',
                    },
                    "fields": "namedStyleType,spaceAbove,spaceBelow"
                }
            },
   
            {
                'updateParagraphStyle' : {
                    'range' : {
                        'startIndex': int(indices[1]),
                        'endIndex': int(indices[2])
                    },
                    'paragraphStyle': {
                        'namedStyleType': 'HEADING_1',
                    },
                    "fields": "namedStyleType,spaceAbove,spaceBelow"
                }
            },

            {
                'updateParagraphStyle' : {
                    'range' : {
                        'startIndex': int(indices[3]),
                        'endIndex': int(indices[4])
                    },
                    'paragraphStyle': {
                        'namedStyleType': 'HEADING_1',
                    },
                    "fields": "namedStyleType,spaceAbove,spaceBelow"
                }
            }
        ]
        service.documents().batchUpdate(documentId=docid, body={'requests':insertion_request}).execute()

        ## Now update sharing permission
        ids = []
        def callback(request_id, response, exception):
            if exception:
                # Handle error
                print(exception)
            else:
                print(f'Request_Id: {request_id}')
                print(F'Permission Id: {response.get("id")}')
                ids.append(response.get('id'))

        batch = drive_service.new_batch_http_request()
        user_permission = {
            'type': 'domain',
            'role': 'writer',
            'domain': 'ebi.ac.uk'
        }
        batch.add(drive_service.permissions().create(fileId=docid, body=user_permission, fields='id'))

        batch.execute()

        sharing_link = drive_service.files().get(fileId=docid, fields='webViewLink').execute()['webViewLink']
        return sharing_link

    except HttpError as err:
        print(err)


def create_id_link_spreadsheet(id_link_dict):
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    try:
        sheets_service = build('sheets', 'v4', credentials=creds)
        drive_service = build('drive', 'v3', credentials=creds)


        spreadsheet = {
            'properties': {
                'title': "ChatGPT summaries"
            }
        }
        sheetid = sheets_service.spreadsheets().create(body=spreadsheet, fields='spreadsheetId').execute()['spreadsheetId']
        print(sheetid)
        id_column = ["RNA id"]
        id_column.extend(list(id_link_dict.keys()))

        link_column = ["Document Link"]
        link_column.extend(list(id_link_dict.values()))
        values = [
            id_column,
            link_column
        ]
        print(len(id_column))
        data = [
            {
                'range': f"A1:C{len(id_column)}",
                
                'majorDimension':'COLUMNS'
            },
            # Additional ranges to update ...
        ]

        body = {
            'values': values,
            'majorDimension':'COLUMNS',
            # 'data': data,
            # 'valueInputOption':"RAW",

        }

        result = sheets_service.spreadsheets().values().update(
            spreadsheetId=sheetid, body=body, range=f"A1:C{len(id_column)}", valueInputOption='RAW'
        ).execute()

        print(f"{result.get('updatedCells')} cells updated.")

        batch = drive_service.new_batch_http_request()
        user_permission = {
            'type': 'domain',
            'role': 'writer',
            'domain': 'ebi.ac.uk'
        }
        batch.add(drive_service.permissions().create(fileId=sheetid, body=user_permission, fields='id'))

        batch.execute()

        sharing_link = drive_service.files().get(fileId=sheetid, fields='webViewLink').execute()

        print(f"This is the sharing link for the ID-link sheet: {sharing_link['webViewLink']}")


    except HttpError as err:
        print(err)

def main():
    """Shows basic usage of the Docs API.
    Prints the title of a sample document.
    """
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('docs', 'v1', credentials=creds)

        # Retrieve the documents contents from the Docs service.
        document = service.documents().get(documentId=DOCUMENT_ID).execute()

        print('The title of the document is: {}'.format(document.get('title')))
    except HttpError as err:
        print(err)


if __name__ == '__main__':
    print("Something should happen...")
    link_data = {}
    for n in range(5):
        this_link = create_summary_doc(f"test_{n}", "The context", "The Summary", "The prompt")
        link_data[this_link['id']] = this_link['link']

    create_id_link_spreadsheet(link_data)