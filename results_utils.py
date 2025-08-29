import json
import os
import pandas as pd
import gspread
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Define the required OAuth 2.0 scope
SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',   # For Google Sheets access
    'https://www.googleapis.com/auth/drive.file'      # For Google Drive file access
]

class ResultsAgent:
    def __init__(self, spreadsheet_id: str = "1ljIem8te0tTKrN8N9jOOnPIRh2zMvv2WB_3FBa4ycgA", 
                 credentials_path: str = "credentials.json"):
        
        self.spreadsheet_id = spreadsheet_id
        self.credentials_path = credentials_path
        
    def upload_results(self, csv_file_path: str):
        """Upload classification results to Google Sheets"""
        token_file = 'token.pickle'

        creds = None
        if os.path.exists(token_file):
            with open(token_file, 'rb') as token:
                creds = pickle.load(token)

        # If no valid credentials, perform the OAuth 2.0 flow
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # Start the OAuth flow to get new credentials
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES)
                creds = flow.run_local_server(port=0)

            # Save the credentials for the next time
            with open(token_file, 'wb') as token:
                pickle.dump(creds, token)

        # Now authorize the credentials
        gc = gspread.authorize(creds)

        # Read the results CSV
        df = pd.read_csv(csv_file_path)
        
        # Create simplified results dataframe
        results_df = pd.DataFrame()
        
        # Extract key information for each result
        for _, row in df.iterrows():
            result_row = {
                'Image ID': row.get('inputs.image_id', ''),
                'Predicted Category': row.get('outputs.category', ''),
                'Correct Category': row.get('reference.category', ''),
                'Is Correct': row.get('outputs.category', '') == row.get('reference.category', ''),
                'Certainty': self._extract_certainty(row),
            }
            results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)

        # Convert to format for Google Sheets
        headers = results_df.columns.tolist()
        data = results_df.values.tolist()

        # Upload to Google Sheets
        worksheet = gc.open_by_key(self.spreadsheet_id).sheet1
        worksheet.clear()

        data_to_upload = [headers] + [[str(item) for item in row] for row in data]
        worksheet.update('A1', data_to_upload)
        worksheet.freeze(rows=1)
        
        print(f"Successfully uploaded {len(results_df)} classification results to Google Sheets")

    def _extract_certainty(self, row):
        """Extract certainty information from the row"""
        # Try to get certainty from the certainty column if it exists
        certainty_data = row.get('certainty', '{}')
        if isinstance(certainty_data, str):
            try:
                certainty_dict = json.loads(certainty_data)
                return certainty_dict.get('category', False)
            except:
                return False
        return False

def save_results(results, results_path):
    """Save classification results to CSV"""
    # Convert to pandas DataFrame
    df = pd.DataFrame(results.to_pandas())
    
    # Clean up the inputs column if needed
    if 'inputs.image' not in df.columns and 'inputs.inputs' in df.columns:
        df["inputs.image_id"] = df["inputs.inputs"].apply(lambda x: x.get("image_id", ""))
        df.drop(columns=['inputs.inputs'], inplace=True)
    elif 'inputs.image' in df.columns:
        df.drop(columns=['inputs.image'], inplace=True)

    # Save the results
    df.to_csv(results_path, index=False)
    print(f"Saved classification results to: {results_path}")

def field_match(outputs: dict, reference_outputs: dict) -> bool:
    """Simple field matching for classification task"""
    return outputs.get("category", "") == reference_outputs.get("category", "")
