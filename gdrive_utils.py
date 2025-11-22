import streamlit as st
from google.oauth2 import service_account
from googleapiclient.discovery import build
import io

def get_credentials():
    """
    Get credentials from streamlit secrets.
    Expected structure in secrets.toml:
    [gcp_service_account]
    type = "service_account"
    project_id = "..."
    ...
    """
    if "gcp_service_account" in st.secrets:
        return service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
    else:
        # Fallback or error
        st.error("No 'gcp_service_account' found in secrets.")
        return None

def get_drive_service():
    """Build and return the Drive service."""
    creds = get_credentials()
    if not creds:
        return None
    return build('drive', 'v3', credentials=creds)

def list_files_in_folder(folder_id: str):
    """
    List all files in the specified GDrive folder.
    Returns a list of dicts: {'id': ..., 'name': ...}
    """
    service = get_drive_service()
    if not service:
        return []
    
    results = []
    page_token = None
    
    query = f"'{folder_id}' in parents and trashed = false"
    
    while True:
        response = service.files().list(
            q=query,
            spaces='drive',
            fields='nextPageToken, files(id, name)',
            pageToken=page_token
        ).execute()
        
        files = response.get('files', [])
        results.extend(files)
        
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
            
    return sorted(results, key=lambda x: x['name'])

def read_file_content(file_id: str) -> str:
    """
    Download file content as a string (decoding CP1252 as per original script).
    """
    service = get_drive_service()
    if not service:
        return ""
        
    request = service.files().get_media(fileId=file_id)
    file_content = request.execute()
    
    # Decode using the same encoding as the original script
    return file_content.decode("cp1252", errors="ignore")
