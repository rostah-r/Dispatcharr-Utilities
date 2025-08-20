import os
import logging
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv, set_key

env_path = Path('.') / '.env'

def _get_base_url():
    """Gets the base URL from environment variables."""
    return os.getenv("DISPATCHARR_BASE_URL")

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
    except json.JSONDecodeError:
        logging.error("Login failed: Invalid JSON response from the server.")
        return False

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

def fetch_data_from_url(url):
    """Fetches data from a given URL with authentication and retry logic."""
    try:
        resp = requests.get(url, headers=_get_auth_headers())
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            if _refresh_token():
                logging.info("Retrying request with new token...")
                resp = requests.get(url, headers=_get_auth_headers())
                resp.raise_for_status()
                return resp.json()
            else:
                return None
        else:
            logging.error(f"Error fetching data from {url}: {e}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data from {url}: {e}")
        return None

def patch_request(url, payload):
    """Sends a PATCH request with authentication and retry logic."""
    try:
        resp = requests.patch(url, json=payload, headers=_get_auth_headers())
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            if _refresh_token():
                logging.info("Retrying PATCH request with new token...")
                return requests.patch(url, json=payload, headers=_get_auth_headers())
            else:
                raise
        else:
            logging.error(f"Error patching data to {url}: {e.response.text}")
            raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Error patching data to {url}: {e}")
        raise

def post_request(url, payload):
    """Sends a POST request with authentication and retry logic."""
    try:
        resp = requests.post(url, json=payload, headers=_get_auth_headers())
        resp.raise_for_status()
        return resp
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            if _refresh_token():
                logging.info("Retrying POST request with new token...")
                return requests.post(url, json=payload, headers=_get_auth_headers())
            else:
                raise
        else:
            logging.error(f"Error posting data to {url}: {e.response.text}")
            raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Error posting data to {url}: {e}")
        raise

def fetch_channel_streams(channel_id):
    """Fetch streams for a given channel ID."""
    url = f"{_get_base_url()}/api/channels/channels/{channel_id}/streams/"
    return fetch_data_from_url(url)

def update_channel_streams(channel_id, stream_ids):
    """Updates the streams for a given channel ID."""
    url = f"{_get_base_url()}/api/channels/channels/{channel_id}/"
    data = {"streams": stream_ids}
    patch_request(url, data)
