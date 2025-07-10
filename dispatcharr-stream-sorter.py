
import argparse
import csv
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv, set_key

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
env_path = Path('.') / '.env'


# --- API Utilities (from api_utils.py) ---

def _get_base_url():
    """Gets the base URL from environment variables."""
    return os.getenv("DISPATCHARR_BASE_URL")

def _get_auth_headers():
    """Returns the authorization headers."""
    current_token = os.getenv("DISPATCHARR_TOKEN")
    if not current_token:
        logging.error("DISPATCHARR_TOKEN not found in .env file. Please run the 'login' command first.")
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
        # load_dotenv(dotenv_path=env_path, override=True) # Removed, main() handles this
        logging.info("Token refreshed successfully.")
        return True
    else:
        logging.error("Token refresh failed.")
        return False

def _fetch_data_from_url(url):
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

def _patch_request(url, payload):
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

def _fetch_channel_streams(channel_id):
    """Fetch streams for a given channel ID."""
    url = f"{_get_base_url()}/api/channels/channels/{channel_id}/streams/"
    return _fetch_data_from_url(url)

def _update_channel_streams(channel_id, stream_ids):
    """Updates the streams for a given channel ID."""
    url = f"{_get_base_url()}/api/channels/channels/{channel_id}/"
    # The API expects a list of stream objects with at least 'id'
    data = {"streams": stream_ids}
    _patch_request(url, data)

def _post_request(url, payload):
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


# --- Main Functionality ---

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
            # No need to load_dotenv here, main() will handle it
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


def fetch_streams(output_file):
    """Fetches streams for a given range of channels."""
    try:
        start_range = int(os.getenv("START_CHANNEL", "1"))
        end_range = int(os.getenv("END_CHANNEL", "999"))
    except ValueError:
        logging.error("Invalid input for START_CHANNEL or END_CHANNEL in .env. Please provide valid integers.")
        return

    logging.info(f"Using channel range: {start_range}-{end_range}")

    if start_range > end_range:
        logging.error("Start of range must be less than or equal to end of range.")
        return

    # Fetch groups and channels
    groups = _fetch_data_from_url(f"{_get_base_url()}/api/channels/groups/")
    if not groups:
        logging.error("Could not fetch groups. Aborting.")
        return

    with open("csv/00_channel_groups.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name"])
        for group in groups:
            writer.writerow([group.get("id", ""), group.get("name", "")])
    logging.info("Saved group list to csv/00_channel_groups.csv")

    channels = _fetch_data_from_url(f"{_get_base_url()}/api/channels/channels/")
    if not channels:
        logging.error("Could not fetch channels. Aborting.")
        return

    with open("csv/01_channels_metadata.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = [
            "id", "channel_number", "name", "channel_group_id", "tvg_id",
            "tvc_guide_stationid", "epg_data_id", "stream_profile_id", "uuid",
            "logo_id", "user_level"
        ]
        writer.writerow(headers)
        for ch in channels:
            writer.writerow([ch.get(h, "") for h in headers])
    logging.info("Saved channel metadata to csv/01_channels_metadata.csv")


    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["channel_number", "channel_id", "stream_id", "stream_name", "stream_url"])

        filtered_channels = [
            ch for ch in channels
            if "channel_number" in ch and ch["channel_number"] is not None and start_range <= int(ch["channel_number"]) <= end_range
        ]

        for channel in filtered_channels:
            channel_id = channel.get("id")
            channel_number = channel.get("channel_number")
            channel_name = channel.get("name", "")

            logging.info(f"Fetching streams for channel {channel_number} (ID: {channel_id}) - {channel_name}...")
            streams = _fetch_data_from_url(f"{_get_base_url()}/api/channels/channels/{channel_id}/streams/")
            if not streams:
                logging.warning(f"  No streams found for channel {channel_number} ({channel_name})")
                continue

            for stream in streams:
                writer.writerow([
                    channel_number,
                    channel_id,
                    stream.get("id", ""),
                    stream.get("name", ""),
                    stream.get("url", "")
                ])
            logging.info(f"  Saved {len(streams)} streams for channel {channel_number} ({channel_name})")

    logging.info(f"\nDone! Stream output saved to: {output_file}")


# --- Stream Analysis ---

provider_semaphores = {}
semaphore_lock = threading.Lock()

def _check_ffmpeg_installed():
    """Checks if ffmpeg and ffprobe are installed and in PATH."""
    try:
        subprocess.run(['ffmpeg', '-h'], capture_output=True, check=True, text=True)
        subprocess.run(['ffprobe', '-h'], capture_output=True, check=True, text=True)
        return True
    except FileNotFoundError:
        logging.error("ffmpeg or ffprobe not found. Please install them and ensure they are in your system's PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking ffmpeg/ffprobe installation: {e}")
        return False

def _get_stream_metrics(url, ffmpeg_duration, timeout):
    command = [
        'ffmpeg', '-re', '-v', 'debug', '-user_agent', 'VLC/3.0.14',
        '-i', url, '-t', str(ffmpeg_duration), '-f', 'null', '-'
    ]

    bitrate = "N/A"
    fps = "N/A"
    resolution = "N/A"
    frames_decoded = "N/A"
    frames_dropped = "N/A"
    elapsed = 0

    try:
        start = time.time()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
        elapsed = time.time() - start
        logging.info(f"ffmpeg ran for {elapsed:.2f} seconds on {url}")

        output = result.stderr

        # Parse ffmpeg output for bitrate, frames, etc.
        total_bytes = 0
        for line in output.splitlines():
            if "Statistics:" in line and "bytes read" in line:
                try:
                    parts = line.split("bytes read")
                    size_str = parts[0].strip().split()[-1]
                    total_bytes = int(size_str)
                    if total_bytes > 0 and ffmpeg_duration > 0:
                        bitrate = (total_bytes * 8) / 1000 / ffmpeg_duration
                        logging.debug(f"Raw bitrate calculation for {url}: total_bytes={total_bytes}, duration={ffmpeg_duration}, bitrate={bitrate}")
                except ValueError:
                    pass
            
            # Attempt to parse frames decoded and dropped from summary lines
            if "Input stream #" in line and "frames decoded;" in line:
                decoded_match = re.search(r'(\d+)\s*frames decoded', line)
                errors_match = re.search(r'(\d+)\s*decode errors', line)
                
                if decoded_match: frames_decoded = int(decoded_match.group(1))
                if errors_match: frames_dropped = int(errors_match.group(1))

            # Attempt to parse resolution and fps from stream info (less reliable than ffprobe)
            if "Stream #" in line and "Video:" in line:
                try:
                    if ", " in line:
                        res_match = re.search(r'(\d{3,})x(\d{3,})', line)
                        if res_match:
                            resolution = res_match.group(0)
                        
                        fps_match = re.search(r'(\d+\.?\d*)\s*fps', line)
                        if fps_match:
                            fps = float(fps_match.group(1))
                except:
                    pass

    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout while fetching metrics for: {url}")
        return "Timeout", "N/A", "N/A", "N/A", "N/A", timeout
    except Exception as e:
        logging.error(f"Metrics check failed for {url}: {e}")
        return "Error", "N/A", "N/A", "N/A", "N/A", 0

    # Fallback for resolution and fps using ffprobe if not found in ffmpeg output
    if resolution == "N/A" or fps == "N/A":
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,avg_frame_rate',
                '-of', 'json',
                url
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
            data = json.loads(result.stdout)
            stream = data.get('streams', [{}])[0]
            width = stream.get('width')
            height = stream.get('height')
            fps_str = stream.get('avg_frame_rate', '0/1')

            try:
                num, den = map(int, fps_str.split('/'))
                parsed_fps = round(num / den, 2) if den != 0 else 0
            except (ValueError, ZeroDivisionError):
                parsed_fps = None

            if width and height: resolution = f"{width}x{height}"
            if parsed_fps is not None: fps = parsed_fps

        except Exception as e:
            logging.warning(f"ffprobe fallback failed for {url}: {e}")

    return bitrate, fps, resolution, frames_decoded, frames_dropped, elapsed

def _get_provider_from_url(url):
    """Extracts the hostname as a simple provider identifier."""
    try:
        return urlparse(url).hostname
    except Exception:
        return "unknown_provider"

def _analyze_stream_task(row, ffmpeg_duration, timeout):
    url = row.get('stream_url')
    if not url:
        return row
    provider = _get_provider_from_url(url)
    with semaphore_lock:
        if provider not in provider_semaphores:
            provider_semaphores[provider] = threading.Semaphore(1)
        provider_semaphore = provider_semaphores[provider]
    with provider_semaphore:
        logging.info(f"Processing stream: {row.get('stream_name', 'Unknown')} (Provider: {provider})")
        bitrate, fps, resolution, frames_decoded, frames_dropped, elapsed = _get_stream_metrics(url, ffmpeg_duration, timeout)
        row['timestamp'] = datetime.now().isoformat()
        row['bitrate_kbps'] = bitrate
        row['fps'] = fps
        row['resolution'] = resolution
        row['frames_decoded'] = frames_decoded
        row['frames_dropped'] = frames_dropped
        if isinstance(elapsed, (int, float)) and elapsed < ffmpeg_duration:
            wait_time = ffmpeg_duration - elapsed
            logging.info(f"Waiting additional {wait_time:.2f} seconds before next stream from {provider}")
            time.sleep(wait_time)
    return row

def analyze_streams(input_csv, output_csv, fails_csv, ffmpeg_duration, timeout, max_workers):
    """Analyzes streams from a CSV file for various metrics."""
    if not _check_ffmpeg_installed():
        sys.exit(1)

    try:
        days_to_keep = int(os.getenv('STREAM_LAST_MEASURED', 7))
    except (ValueError, TypeError):
        days_to_keep = 7
        logging.warning("Invalid or missing STREAM_LAST_MEASURED in .env, defaulting to 7 days.")

    last_measured_date = datetime.now() - timedelta(days=days_to_keep)
    # Get all stream URLs from the current input CSV
    current_input_urls = set()
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if 'stream_url' in row:
                current_input_urls.add(row['stream_url'])

    processed_urls = set()
    if days_to_keep > 0 and os.path.exists(output_csv):
        with open(output_csv, newline='', encoding='utf-8') as outfile:
            reader = csv.DictReader(outfile)
            for row in reader:
                if 'stream_url' in row and 'timestamp' in row and row['stream_url'] in current_input_urls:
                    try:
                        row_timestamp = datetime.fromisoformat(row['timestamp'])
                        if row_timestamp > last_measured_date:
                            processed_urls.add(row['stream_url'])
                    except ValueError:
                        pass # Ignore rows with invalid timestamps

    # --- Duplicate Stream Handling ---
    first_occurrence = {}
    duplicates_to_remove = []
    all_input_rows = []
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            all_input_rows.append(row)
            stream_url = row.get('stream_url')
            channel_id = row.get('channel_id')
            stream_id = row.get('stream_id')
            if stream_url and channel_id and stream_id:
                if stream_url not in first_occurrence:
                    first_occurrence[stream_url] = (channel_id, stream_id)
                else:
                    duplicates_to_remove.append({'channel_id': channel_id, 'stream_id': stream_id})
    channels_with_duplicates = defaultdict(list)
    for dup in duplicates_to_remove:
        channels_with_duplicates[dup['channel_id']].append(dup['stream_id'])
    for channel_id, stream_ids_to_remove in channels_with_duplicates.items():
        try:
            current_streams_data = _fetch_channel_streams(channel_id)
            if current_streams_data:
                current_stream_ids = [s['id'] for s in current_streams_data]
                updated_stream_ids = [sid for sid in current_stream_ids if sid not in stream_ids_to_remove]
                if len(updated_stream_ids) < len(current_stream_ids):
                    logging.info(f"Updating channel {channel_id} to remove {len(current_stream_ids) - len(updated_stream_ids)} duplicate streams.")
                    _update_channel_streams(channel_id, updated_stream_ids)
        except Exception as e:
            logging.error(f"Error removing duplicate streams for channel {channel_id}: {e}")
    # --- End Duplicate Stream Handling ---

    streams_to_analyze = []
    current_run_processed_urls = set()
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = ['timestamp'] + reader.fieldnames + ['bitrate_kbps', 'fps', 'resolution', 'frames_decoded', 'frames_dropped']
        for row in reader:
            stream_url = row.get('stream_url')
            if stream_url and stream_url not in processed_urls and stream_url not in current_run_processed_urls and first_occurrence.get(stream_url) == (row.get('channel_id'), row.get('stream_id')):
                streams_to_analyze.append(row)
                current_run_processed_urls.add(stream_url)

    file_exists = os.path.exists(output_csv)
    with open(output_csv, 'a', newline='', encoding='utf-8') as outfile, \
         open(fails_csv, 'a', newline='', encoding='utf-8') as fails_outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        fails_writer = csv.DictWriter(fails_outfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        if not os.path.exists(fails_csv):
            fails_writer.writeheader()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(_analyze_stream_task, row, ffmpeg_duration, timeout): row for row in streams_to_analyze}
            for future in future_to_row:
                try:
                    result_row = future.result()
                    writer.writerow(result_row)
                    if any(str(result_row.get(f)).strip().lower() in ['n/a', 'error', 'timeout'] for f in ['bitrate_kbps', 'fps', 'resolution']):
                        fails_writer.writerow(result_row)
                except Exception as exc:
                    original_row = future_to_row[future]
                    logging.error(f'Stream {original_row.get("stream_name", "Unknown")} generated an exception: {exc}')
                    original_row.update({'timestamp': datetime.now().isoformat(), 'bitrate_kbps': "Exception", 'fps': "Exception", 'resolution': "Exception"})
                    writer.writerow(original_row)
                    fails_writer.writerow(original_row)

# --- Scoring and Sorting ---

def score_streams(input_csv, output_csv):
    """Calculates averages, scores, and sorts streams."""
    # Calculate Averages
    bitrates = defaultdict(list)
    frames_decoded = defaultdict(list)
    frames_dropped = defaultdict(list)
    stream_metadata = {}

    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            stream_id = row.get('stream_id')
            if not stream_id:
                continue
            stream_metadata[stream_id] = {
                'channel_number': row.get('channel_number'),
                'channel_id': row.get('channel_id'),
                'stream_name': row.get('stream_name'),
                'stream_url': row.get('stream_url'),
                'fps': row.get('fps', '').strip(),
                'resolution': row.get('resolution', '').strip()
            }
            try:
                bitrate = float(row.get('bitrate_kbps'))
                bitrates[stream_id].append(bitrate)
            except (ValueError, TypeError):
                pass
            try:
                decoded = float(row.get('frames_decoded'))
                frames_decoded[stream_id].append(decoded)
            except (ValueError, TypeError):
                pass
            try:
                dropped = float(row.get('frames_dropped'))
                frames_dropped[stream_id].append(dropped)
            except (ValueError, TypeError):
                pass

    summary_data = []
    for stream_id, metadata in stream_metadata.items():
        avg_bitrate = round(sum(bitrates[stream_id]) / len(bitrates[stream_id])) if bitrates[stream_id] else "N/A"
        avg_decoded = round(sum(frames_decoded[stream_id]) / len(frames_decoded[stream_id])) if frames_decoded[stream_id] else "N/A"
        avg_dropped = round(sum(frames_dropped[stream_id]) / len(frames_dropped[stream_id])) if frames_dropped[stream_id] else "N/A"
        dropped_percentage = round((avg_dropped / avg_decoded) * 100, 2) if isinstance(avg_decoded, (int, float)) and isinstance(avg_dropped, (int, float)) and avg_decoded > 0 else "N/A"
        summary_data.append({
            'stream_id': stream_id,
            'channel_number': metadata.get('channel_number'),
            'channel_id': metadata.get('channel_id'),
            'stream_name': metadata.get('stream_name'),
            'stream_url': metadata.get('stream_url'),
            'avg_bitrate_kbps': avg_bitrate,
            'avg_frames_decoded': avg_decoded,
            'avg_frames_dropped': avg_dropped,
            'dropped_frame_percentage': dropped_percentage,
            'fps': metadata.get('fps'),
            'resolution': metadata.get('resolution')
        })

    df = pd.DataFrame(summary_data)

    # Score and Sort
    RESOLUTION_SCORES = {
        '3840x2160': 100,
        '1920x1080': 80,
        '1280x720': 50,
        '960x540': 20,
        'Unknown': 0,
        '': 0
    }
    df['avg_bitrate_kbps'] = pd.to_numeric(df['avg_bitrate_kbps'], errors='coerce')
    df['fps'] = pd.to_numeric(df['fps'], errors='coerce')
    df['dropped_frame_percentage'] = pd.to_numeric(df['dropped_frame_percentage'], errors='coerce').fillna(0)
    df['max_bitrate_for_channel'] = df.groupby('channel_number')['avg_bitrate_kbps'].transform('max')
    df['bitrate_score'] = (df['avg_bitrate_kbps'] / (df['max_bitrate_for_channel'] * 0.01)).fillna(0)
    df['resolution_score'] = df['resolution'].astype(str).str.strip().map(RESOLUTION_SCORES).fillna(0)
    df['fps_bonus'] = 0
    fps_bonus_points = int(os.getenv("FPS_BONUS_POINTS", "55"))
    df.loc[df['fps'] >= 50, 'fps_bonus'] = fps_bonus_points
    df['dropped_frames_penalty'] = df['dropped_frame_percentage'] * 1
    df['score'] = (
        df['bitrate_score'] +
        df['resolution_score'] +
        df['fps_bonus'] -
        df['dropped_frames_penalty']
    )
    df.loc[df['avg_bitrate_kbps'].isna(), 'score'] = -1
    df_sorted = df.sort_values(by=['channel_number', 'score'], ascending=[True, False])
    df_sorted.to_csv(output_csv, index=False, na_rep='N/A')
    logging.info(f"Scored and sorted CSV saved as {output_csv}")

# --- Reordering Streams ---

def reorder_streams(input_csv):
    """Reorders streams in Dispatcharr based on the scored and sorted CSV."""
    logging.info(f"Reordering streams based on {input_csv}...")
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Error: {input_csv} not found. Please run the 'score' command first.")
        return

    base_url = _get_base_url()
    if not base_url:
        logging.error("DISPATCHARR_BASE_URL not set in .env file.")
        return

    grouped = df.groupby("channel_id")

    for channel_id, group in grouped:
        stream_ids = group["stream_id"].tolist()
        channel_id = int(channel_id)

        # Fetch current channel data to get the full object for PATCH
        channel_data_url = f"{base_url}/api/channels/channels/{channel_id}/"
        current_data = _fetch_data_from_url(channel_data_url)

        if not current_data:
            logging.error(f"Failed to fetch current data for channel_id {channel_id}. Skipping reorder.")
            continue

        # Update only the streams field with the new ordered list of stream IDs
        current_data["streams"] = stream_ids

        try:
            patch_response = _patch_request(channel_data_url, current_data)
            logging.info(f"Successfully reordered streams for channel ID {channel_id}. Status: {patch_response.status_code}")
        except Exception as e:
            logging.error(f"Failed to reorder streams for channel ID {channel_id}: {e}")

    logging.info("Stream reordering complete.")

def main():
    """Main function to parse arguments and call the appropriate function."""
    load_dotenv(dotenv_path=env_path, override=True)

    parser = argparse.ArgumentParser(
        description="A tool for managing and analyzing Dispatcharr IPTV streams.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Login command
    subparsers.add_parser('login', help='Authenticate with Dispatcharr and save the token.')

    # Fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch channel and stream information.')
    fetch_parser.add_argument('--output', type=str, default='csv/02_grouped_channel_streams.csv', help='The output CSV file for fetched streams.')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze IPTV streams for bitrate, resolution, and FPS.')
    analyze_parser.add_argument('--input', type=str, default='csv/02_grouped_channel_streams.csv', help='Input CSV file with stream URLs.')
    analyze_parser.add_argument('--output', type=str, default='csv/03_iptv_stream_measurements.csv', help='Output CSV file for analyzed stream data.')
    analyze_parser.add_argument('--fails_output', type=str, default='csv/04_fails.csv', help='Output CSV file for streams that failed analysis.')
    analyze_parser.add_argument('--duration', type=int, default=20, help='Duration in seconds for ffmpeg to analyze stream.')
    analyze_parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds for ffmpeg/ffprobe commands.')
    analyze_parser.add_argument('--workers', type=int, default=8, help='Number of concurrent workers for stream analysis.')

    # Score command
    score_parser = subparsers.add_parser('score', help='Calculate averages, score, and sort streams.')
    score_parser.add_argument('--input', type=str, default='csv/03_iptv_stream_measurements.csv', help='Input CSV file with stream measurements.')
    score_parser.add_argument('--output', type=str, default='csv/05_iptv_streams_scored_sorted.csv', help='Output CSV file for scored and sorted streams.')

    # Reorder command
    reorder_parser = subparsers.add_parser('reorder', help='Reorder streams in Dispatcharr based on the scored CSV.')
    reorder_parser.add_argument('--input', type=str, default='csv/05_iptv_streams_scored_sorted.csv', help='Input CSV file with scored and sorted streams.')

    args = parser.parse_args()

    if args.command == 'login':
        login()
    elif args.command == 'fetch':
        fetch_streams(args.output)
    elif args.command == 'analyze':
        analyze_streams(args.input, args.output, args.fails_output, args.duration, args.timeout, args.workers)
    elif args.command == 'score':
        score_streams(args.input, args.output)
    elif args.command == 'reorder':
        reorder_streams(args.input)
    else:
        # Default pipeline if no command is specified
        logging.info("No command specified. Running default pipeline: login -> fetch -> analyze -> score -> reorder")
        if not login():
            logging.error("Login failed. Aborting pipeline.")
            return
        fetch_streams('csv/02_grouped_channel_streams.csv')
        analyze_streams('csv/02_grouped_channel_streams.csv', 'csv/03_iptv_stream_measurements.csv', 'csv/04_fails.csv', 20, 30, 8)
        score_streams('csv/03_iptv_stream_measurements.csv', 'csv/05_iptv_streams_scored_sorted.csv')
        reorder_streams('csv/05_iptv_streams_scored_sorted.csv')


if __name__ == "__main__":
    main()
