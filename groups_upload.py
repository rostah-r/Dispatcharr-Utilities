import csv
import logging
import os
import sys
import requests
from dotenv import load_dotenv, set_key
from pathlib import Path

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# --- API Utilities ---
def _get_base_url():
    """Gets the base URL from environment variables."""
    return os.getenv("DISPATCHARR_BASE_URL")

def _get_auth_headers():
    """Returns the authorization headers, attempting to log in if no token is found."""
    current_token = os.getenv("DISPATCHARR_TOKEN")
    if not current_token:
        logging.info("DISPATCHARR_TOKEN not found. Attempting to log in...")
        if login():
            load_dotenv(dotenv_path=env_path, override=True)
            current_token = os.getenv("DISPATCHARR_TOKEN")
            if not current_token:
                logging.error("Login seemed to succeed, but token is still not found. Aborting.")
                sys.exit(1)
        else:
            logging.error("Login failed. Please check credentials in .env file. Aborting.")
            sys.exit(1)
    return {
        "Authorization": f"Bearer {current_token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

def login():
    """Logs into Dispatcharr and saves the token to the .env file."""
    username = os.getenv("DISPATCHARR_USER")
    password = os.getenv("DISPATCHARR_PASS")
    base_url = _get_base_url()

    if not all([username, password, base_url]):
        logging.error("DISPATCHARR_USER, DISPATCHARR_PASS, and DISPATCHARR_BASE_URL must be set in the .env file.")
        return False

    login_url = f"{base_url}/api/accounts/token/"
    logging.info(f"Attempting to log in to {base_url}...")

    try:
        resp = requests.post(
            login_url,
            headers={"Content-Type": "application/json"},
            json={"username": username, "password": password}
        )
        resp.raise_for_status()
        data = resp.json()
        token = data.get("access") or data.get("token")

        if token:
            set_key(env_path, "DISPATCHARR_TOKEN", token)
            logging.info("Login successful. Token saved to .env file.")
            return True
        else:
            logging.error("Login failed: No access token found in response.")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Login failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.text}")
        return False

def _refresh_token():
    """Refreshes the authentication token by calling the login function."""
    logging.info("Token expired or invalid. Attempting to refresh...")
    if login():
        load_dotenv(dotenv_path=env_path, override=True)
        logging.info("Token refreshed successfully.")
        return True
    else:
        logging.error("Token refresh failed.")
        return False

def _make_request(method, url, **kwargs):
    """Makes a request with authentication and retry logic."""
    try:
        resp = requests.request(method, url, headers=_get_auth_headers(), **kwargs)
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            if _refresh_token():
                logging.info(f"Retrying {method} request to {url} with new token...")
                resp = requests.request(method, url, headers=_get_auth_headers(), **kwargs)
                resp.raise_for_status()
                return resp
            else:
                raise
        else:
            logging.error(f"HTTP Error: {e.response.status_code} for URL: {url}")
            logging.error(f"Response: {e.response.text}")
            raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")
        raise

# --- Main Functionality ---
def fetch_existing_groups():
    """Fetches existing groups from Dispatcharr."""
    url = f"{_get_base_url()}/api/channels/groups/"
    try:
        response = _make_request("GET", url)
        return {str(g["id"]): g for g in response.json()} if response.status_code == 200 else {}
    except requests.exceptions.RequestException as e:
        logging.error(f"Could not fetch existing groups: {e}")
        return {}

def update_group(group_id, new_name):
    """Updates an existing group in Dispatcharr."""
    url = f"{_get_base_url()}/api/channels/groups/{group_id}/"
    payload = {"name": new_name}
    return _make_request("PATCH", url, json=payload)

def create_group(name):
    """Creates a new group in Dispatcharr."""
    url = f"{_get_base_url()}/api/channels/groups/"
    payload = {"name": name}
    return _make_request("POST", url, json=payload)

def main():
    """Main function to sync groups from a CSV file."""
    csv_file = "csv/groups_template.csv"
    if not os.path.exists(csv_file):
        logging.error(f"Error: The file {csv_file} was not found.")
        sys.exit(1)

    logging.info("📥 Syncing groups from CSV...")
    existing_groups = fetch_existing_groups()

    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            gid, name = row.get("id", "").strip(), row.get("name", "").strip()
            if not gid or not name:
                logging.warning(f"Skipping row with missing id or name: {row}")
                continue

            if gid in existing_groups:
                current_name = existing_groups[gid]["name"]
                if current_name != name:
                    try:
                        update_group(gid, name)
                        logging.info(f"  🔁 Updated group ID {gid}: '{current_name}' → '{name}'")
                    except requests.exceptions.RequestException:
                        logging.error(f"  ❌ Failed to update group ID {gid}")
                else:
                    logging.info(f"  ✅ Group ID {gid} ('{name}') already up-to-date")
            else:
                try:
                    create_group(name)
                    logging.info(f"  ➕ Created new group: {name}")
                except requests.exceptions.RequestException:
                    logging.error(f"  ❌ Failed to create group: {name}")

    logging.info("\n✅ Sync complete!")

if __name__ == "__main__":
    main()