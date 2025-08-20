

import argparse
import configparser
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
from dotenv import load_dotenv

from api_utils import (
    _get_base_url,
    fetch_channel_streams,
    fetch_data_from_url,
    login,
    update_channel_streams,
    patch_request,
)

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
def load_config():
    """Loads the configuration from the config.ini file."""
    config = configparser.ConfigParser()
    config_path = Path(__file__).parent / 'config.ini'
    if not config_path.exists():
        logging.error(f"Configuration file not found at: {config_path}")
        sys.exit(1)
    config.read(config_path)
    return config

# --- Main Functionality ---

def fetch_streams(config, output_file):
    """Fetches streams for channels based on group and/or range filters."""
    settings = config['script_settings']
    try:
        group_ids_str = settings.get('channel_group_ids', 'ALL').strip()
        start_range = settings.getint('start_channel', 1)
        end_range = settings.getint('end_channel', 999)
    except ValueError:
        logging.error("Invalid number format in config.ini for start/end channel. Please provide valid integers.")
        return

    # --- Fetch initial data ---
    base_url = _get_base_url()
    if not base_url:
        logging.error("DISPATCHARR_BASE_URL not set in .env file.")
        return

    groups = fetch_data_from_url(f"{base_url}/api/channels/groups/")
    if not groups:
        logging.error("Could not fetch groups. Aborting.")
        return
    with open("csv/00_channel_groups.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name"])
        for group in groups:
            writer.writerow([group.get("id", ""), group.get("name", "")])
    logging.info("Saved group list to csv/00_channel_groups.csv")

    all_channels = fetch_data_from_url(f"{base_url}/api/channels/channels/")
    if not all_channels:
        logging.error("Could not fetch channels. Aborting.")
        return

    # --- Filtering Logic ---
    target_channels = []
    use_group_filter = group_ids_str.upper() != 'ALL'

    if use_group_filter:
        try:
            target_group_ids = {int(gid.strip()) for gid in group_ids_str.split(',')}
            logging.info(f"Filtering for channels in groups: {target_group_ids}")
            target_channels = [ch for ch in all_channels if ch.get('channel_group_id') in target_group_ids]
        except ValueError:
            logging.error(f"Invalid channel_group_ids in config.ini: '{group_ids_str}'. Please use a comma-separated list of numbers.")
            return
    else:
        logging.info("No specific groups selected (ALL). Using channel number range as primary filter.")
        target_channels = all_channels

    # Apply channel number range as a secondary filter
    final_filtered_channels = [
        ch for ch in target_channels
        if ch.get("channel_number") and start_range <= int(ch["channel_number"]) <= end_range
    ]

    if not final_filtered_channels:
        logging.error("Conflict in filters: No channels were found that match BOTH the selected group(s) and the channel number range. Please check your config.ini. Aborting.")
        return

    logging.info(f"Found {len(final_filtered_channels)} channels to process after applying all filters.")

    # --- Write metadata and streams for filtered channels ---
    with open("csv/01_channels_metadata.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        headers = ["id", "channel_number", "name", "channel_group_id", "tvg_id", "tvc_guide_stationid", "epg_data_id", "logo_id"]
        writer.writerow(headers)
        for ch in final_filtered_channels:
            writer.writerow([ch.get(h, "") for h in headers])
    logging.info("Saved channel metadata for filtered channels to csv/01_channels_metadata.csv")

    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Add channel_group_id to the header
        writer.writerow(["channel_number", "channel_id", "channel_group_id", "stream_id", "stream_name", "stream_url"])

        for channel in final_filtered_channels:
            channel_id = channel.get("id")
            channel_number = channel.get("channel_number")
            channel_group_id = channel.get("channel_group_id") # Get group ID
            channel_name = channel.get("name", "")

            logging.info(f"Fetching streams for channel {channel_number} (Group: {channel_group_id}, ID: {channel_id}) - {channel_name})...")
            streams = fetch_channel_streams(channel_id)
            if not streams:
                logging.warning(f"  No streams found for channel {channel_number} ({channel_name})")
                continue

            for stream in streams:
                writer.writerow([
                    channel_number,
                    channel_id,
                    channel_group_id, # Write group ID to the CSV
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

def _get_stream_info(url, timeout):
    """Gets stream information using ffprobe."""
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate',
        '-of', 'json',
        url
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
        if result.stdout:
            data = json.loads(result.stdout)
            return data.get('streams', [])
        return []
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout while fetching stream info for: {url}")
        return []
    except json.JSONDecodeError:
        logging.warning(f"Failed to decode JSON from ffprobe for {url}")
        return []
    except Exception as e:
        logging.error(f"Stream info check failed for {url}: {e}")
        return []

def _check_interlaced_status(url, stream_name, idet_frames, timeout):
    """
    Checks if a video stream is interlaced using ffmpeg's idet filter.
    Returns 'INTERLACED', 'PROGRESSIVE', or 'UNKNOWN' if detection fails.
    """
    idet_command = [
        'ffmpeg', '-user_agent', 'VLC/3.0.14',
        '-analyzeduration', '5000000', '-probesize', '5000000',
        '-i', url, '-vf', 'idet', '-frames:v', str(idet_frames), '-an', '-f', 'null', 'NUL' if os.name == 'nt' else '/dev/null'
    ]

    try:
        idet_result = subprocess.run(idet_command, capture_output=True, text=True, timeout=timeout)
        idet_output = idet_result.stderr

        interlaced_frames = 0
        progressive_frames = 0

        for line in idet_output.splitlines():
            if "Single frame detection:" in line or "Multi frame detection:" in line:
                tff_match = re.search(r'TFF:\s*(\d+)', line)
                bff_match = re.search(r'BFF:\s*(\d+)', line)
                progressive_match = re.search(r'Progressive:\s*(\d+)', line)

                if tff_match: interlaced_frames += int(tff_match.group(1))
                if bff_match: interlaced_frames += int(bff_match.group(1))
                if progressive_match: progressive_frames += int(progressive_match.group(1))
        
        if interlaced_frames > progressive_frames:
            status = "INTERLACED"
        elif progressive_frames > interlaced_frames:
            status = "PROGRESSIVE"
        else:
            status = "UNKNOWN"
            
        return status

    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout checking interlacing for {stream_name} ({url})")
        return "UNKNOWN (Timeout)"
    except Exception as e:
        logging.error(f"Error checking interlacing for {stream_name} ({url}): {e}")
        return "UNKNOWN (Error)"

def _get_bitrate_and_frame_stats(url, ffmpeg_duration, timeout):
    """Gets bitrate and frame statistics using ffmpeg."""
    command = [
        'ffmpeg', '-re', '-v', 'debug', '-user_agent', 'VLC/3.0.14',
        '-i', url, '-t', str(ffmpeg_duration), '-f', 'null', '-'
    ]
    bitrate = "N/A"
    frames_decoded = "N/A"
    frames_dropped = "N/A"
    elapsed = 0
    status = "OK"

    try:
        start = time.time()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, text=True)
        elapsed = time.time() - start
        output = result.stderr
        total_bytes = 0
        for line in output.splitlines():
            if "Statistics:" in line and "bytes read" in line:
                try:
                    parts = line.split("bytes read")
                    size_str = parts[0].strip().split()[-1]
                    total_bytes = int(size_str)
                    if total_bytes > 0 and ffmpeg_duration > 0:
                        bitrate = (total_bytes * 8) / 1000 / ffmpeg_duration
                except ValueError:
                    pass
            if "Input stream #" in line and "frames decoded;" in line:
                decoded_match = re.search(r'(\d+)\s*frames decoded', line)
                errors_match = re.search(r'(\d+)\s*decode errors', line)
                if decoded_match: frames_decoded = int(decoded_match.group(1))
                if errors_match: frames_dropped = int(errors_match.group(1))
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout while fetching bitrate/frames for: {url}")
        status = "Timeout"
        elapsed = timeout
    except Exception as e:
        logging.error(f"Bitrate/frames check failed for {url}: {e}")
        status = "Error"

    return bitrate, frames_decoded, frames_dropped, status, elapsed

def _get_provider_from_url(url):
    """Extracts the hostname and port as a provider identifier."""
    try:
        return urlparse(url).netloc
    except Exception:
        return "unknown_provider"

def _check_stream_for_critical_errors(url, stream_name, timeout, config):
    """
    Runs a specific ffmpeg command to check for critical, provider-side errors.
    Returns a dictionary of identified critical errors.
    """
    settings = config['script_settings']
    hwaccel_mode = settings.get('ffmpeg_hwaccel_mode', 'none').lower()

    # Base command arguments
    ffmpeg_command = [
        'ffmpeg',
        '-probesize', '500000', '-analyzeduration', '1000000',
        '-fflags', '+genpts+discardcorrupt', '-flags', 'low_delay',
        '-flush_packets', '1', '-avoid_negative_ts', 'make_zero',
        '-timeout', '5000000', '-rw_timeout', '5000000',
    ]

    # Hardware acceleration specific arguments
    if hwaccel_mode == 'qsv':
        ffmpeg_command.extend([
            '-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv',
        ])

    # Input and common arguments
    ffmpeg_command.extend([
        '-i', url,
        '-t', '20', # 20 second duration for the check
        '-map', '0:v:0', '-map', '0:a:0?', '-map', '0:s?',
    ])

    # Codec and output arguments
    if hwaccel_mode == 'qsv':
        ffmpeg_command.extend([
            '-c:v', 'hevc_qsv',
        ])
    else: # Default to software encoding
        ffmpeg_command.extend([
            '-c:v', 'libx265',
        ])

    ffmpeg_command.extend([
        '-preset', 'veryfast', '-profile:v', 'main', '-g', '50', '-bf', '1',
        '-b:v', '12000k', '-maxrate', '15000k', '-bufsize', '25000k',
        '-c:a', 'libfdk_aac', '-vbr', '4', '-b:a', '128k', '-ac', '2',
        '-af', 'aresample=async=0', '-fps_mode', 'passthrough',
        '-f', 'null', '-'
    ])

    errors = {
        'err_decode': False,
        'err_discontinuity': False,
        'err_timeout': False,
    }

    try:
        result = subprocess.run(
            ffmpeg_command,
            capture_output=True, # Captures both stdout and stderr
            text=True,
            timeout=timeout
        )
        stderr_output = result.stderr

        if "decode_slice_header error" in stderr_output:
            errors['err_decode'] = True
        if "timestamp discontinuity" in stderr_output:
            errors['err_discontinuity'] = True
        if "Connection timed out" in stderr_output:
            errors['err_timeout'] = True

    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout during critical error check for {stream_name} ({url})")
        errors['err_timeout'] = True
    except Exception as e:
        logging.error(f"Exception during critical error check for {stream_name} ({url}): {e}")

    return errors

def _analyze_stream_task(row, ffmpeg_duration, idet_frames, timeout, retries, retry_delay, config):
    url = row.get('stream_url')
    stream_name = row.get('stream_name', 'Unknown')
    if not url:
        return row

    provider = _get_provider_from_url(url)
    with semaphore_lock:
        if provider not in provider_semaphores:
            provider_semaphores[provider] = threading.Semaphore(1)
        provider_semaphore = provider_semaphores[provider]

    with provider_semaphore:
        logging.info(f"Processing stream: {stream_name} (Provider: {provider})")

        for attempt in range(retries + 1):
            # Initialize fields for each attempt
            row['timestamp'] = datetime.now().isoformat()
            row['video_codec'] = 'N/A'
            row['audio_codec'] = 'N/A'
            row['resolution'] = 'N/A'
            row['fps'] = 'N/A'
            row['interlaced_status'] = 'N/A'
            row['bitrate_kbps'] = 'N/A'
            row['frames_decoded'] = 'N/A'
            row['frames_dropped'] = 'N/A'
            row['status'] = 'N/A'

            # 1. Get Codec, Resolution, FPS from ffprobe
            streams_info = _get_stream_info(url, timeout)
            video_info = next((s for s in streams_info if 'width' in s), None)
            audio_info = next((s for s in streams_info if 'codec_name' in s and 'width' not in s), None)

            if video_info:
                row['video_codec'] = video_info.get('codec_name')
                row['resolution'] = f"{video_info.get('width')}x{video_info.get('height')}"
                fps_str = video_info.get('avg_frame_rate', '0/1')
                try:
                    num, den = map(int, fps_str.split('/'))
                    row['fps'] = round(num / den, 2) if den != 0 else 0
                except (ValueError, ZeroDivisionError):
                    row['fps'] = 0
            
            if audio_info:
                row['audio_codec'] = audio_info.get('codec_name')

            # 2. Get Bitrate and Frame Drop stats from ffmpeg
            bitrate, frames_decoded, frames_dropped, status, elapsed = _get_bitrate_and_frame_stats(url, ffmpeg_duration, timeout)
            row['bitrate_kbps'] = bitrate
            row['frames_decoded'] = frames_decoded
            row['frames_dropped'] = frames_dropped
            row['status'] = status

            # 3. Check for interlacing if stream is OK so far
            if status == "OK":
                row['interlaced_status'] = _check_interlaced_status(url, stream_name, idet_frames, timeout)
            else:
                row['interlaced_status'] = "N/A"

            # 4. Perform critical error check
            critical_errors = _check_stream_for_critical_errors(url, stream_name, timeout, config)
            row.update(critical_errors)

            # If the main status is OK, break the retry loop
            if status == "OK":
                break

            # If not the last attempt, wait before retrying
            if attempt < retries:
                logging.warning(f"Stream '{stream_name}' failed with status '{status}'. Retrying in {retry_delay} seconds... ({attempt + 1}/{retries})")
                time.sleep(retry_delay)

        # Respect ffmpeg duration to avoid hammering provider
        if isinstance(elapsed, (int, float)) and elapsed < ffmpeg_duration:
            wait_time = ffmpeg_duration - elapsed
            logging.info(f"Waiting additional {wait_time:.2f} seconds before next stream from {provider}")
            time.sleep(wait_time)

    return row

def analyze_streams(config, input_csv, output_csv, fails_csv, ffmpeg_duration, idet_frames, timeout, max_workers, retries, retry_delay):
    """Analyzes streams from a CSV file for various metrics and saves results incrementally."""
    if not _check_ffmpeg_installed():
        sys.exit(1)

    settings = config['script_settings']

    # --- Load and Filter Data ---
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Input CSV not found: {input_csv}")
        return

    try:
        start_range = settings.getint('start_channel', 1)
        end_range = settings.getint('end_channel', 999)
        group_ids_str = settings.get('channel_group_ids', 'ALL').strip()
    except ValueError:
        logging.error("Invalid start_channel or end_channel in config.ini. Aborting analyze.")
        return

    if group_ids_str.upper() != 'ALL':
        try:
            target_group_ids = {int(gid.strip()) for gid in group_ids_str.split(',')}
            df['channel_group_id'] = pd.to_numeric(df['channel_group_id'], errors='coerce')
            df = df[df['channel_group_id'].isin(target_group_ids)]
        except ValueError:
            logging.error(f"Invalid channel_group_ids in config.ini: '{group_ids_str}'. Aborting analyze.")
            return

    df['channel_number'] = pd.to_numeric(df['channel_number'], errors='coerce')
    df.dropna(subset=['channel_number'], inplace=True)
    df = df[df['channel_number'].between(start_range, end_range)]

    if df.empty:
        logging.warning(f"No streams found in {input_csv} for the specified filters. Nothing to analyze.")
        return

    # --- Prune Recently Analyzed Streams ---
    try:
        days_to_keep = settings.getint('stream_last_measured_days', 7)
    except (ValueError, TypeError):
        days_to_keep = 7
        logging.warning("Invalid or missing stream_last_measured_days in config.ini, defaulting to 7 days.")

    if days_to_keep > 0 and os.path.exists(output_csv):
        try:
            df_processed = pd.read_csv(output_csv)
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
            last_measured_date = datetime.now() - timedelta(days=days_to_keep)
            recent_urls = df_processed[df_processed['timestamp'] > last_measured_date]['stream_url'].unique()
            df = df[~df['stream_url'].isin(recent_urls)]
        except Exception as e:
            logging.warning(f"Could not read or parse existing measurements file '{output_csv}'. Re-analyzing all streams. Error: {e}")

    # --- Duplicate Stream Handling (API removal part) ---
    duplicates = df[df.duplicated(subset=['stream_url'], keep='first')]
    if not duplicates.empty:
        channels_with_duplicates = duplicates.groupby('channel_id')['stream_id'].apply(list).to_dict()
        for channel_id, stream_ids_to_remove in channels_with_duplicates.items():
            try:
                current_streams_data = fetch_channel_streams(channel_id)
                if current_streams_data:
                    current_stream_ids = [s['id'] for s in current_streams_data]
                    updated_stream_ids = [sid for sid in current_stream_ids if sid not in stream_ids_to_remove]
                    if len(updated_stream_ids) < len(current_stream_ids):
                        logging.info(f"Updating channel {channel_id} to remove {len(current_stream_ids) - len(updated_stream_ids)} duplicate streams.")
                        update_channel_streams(channel_id, updated_stream_ids)
            except Exception as e:
                logging.error(f"Error removing duplicate streams for channel {channel_id}: {e}")

    # --- Prepare Final List for Analysis ---
    df.drop_duplicates(subset=['stream_url'], keep='first', inplace=True)

    if df.empty:
        logging.info("All filtered streams have been analyzed recently. Nothing to do.")
        return

    streams_to_analyze = df.to_dict('records')

    # --- Execute Analysis and Write Incrementally ---
    final_columns = [
        'channel_number', 'channel_id', 'stream_id', 'stream_name', 'stream_url',
        'channel_group_id', 'timestamp', 'video_codec', 'audio_codec', 'interlaced_status',
        'status', 'bitrate_kbps', 'fps', 'resolution', 'frames_decoded', 'frames_dropped',
        'err_decode', 'err_discontinuity', 'err_timeout'
    ]
    
    # Ensure the output directory exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(fails_csv).parent.mkdir(parents=True, exist_ok=True)

    # Check if files exist to determine if we need to write headers
    output_exists = os.path.exists(output_csv)
    fails_exists = os.path.exists(fails_csv)

    try:
        with open(output_csv, 'a', newline='', encoding='utf-8') as f_out, \
             open(fails_csv, 'a', newline='', encoding='utf-8') as f_fails:

            writer_out = csv.DictWriter(f_out, fieldnames=final_columns, extrasaction='ignore', lineterminator='\n')
            writer_fails = csv.DictWriter(f_fails, fieldnames=final_columns, extrasaction='ignore', lineterminator='\n')

            if not output_exists or os.path.getsize(output_csv) == 0:
                writer_out.writeheader()
            if not fails_exists or os.path.getsize(fails_csv) == 0:
                writer_fails.writeheader()

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_row = {executor.submit(_analyze_stream_task, row, ffmpeg_duration, idet_frames, timeout, retries, retry_delay, config): row for row in streams_to_analyze}
                
                for future in future_to_row:
                    try:
                        result_row = future.result()
                        
                        # Write to the main measurements file
                        writer_out.writerow(result_row)
                        f_out.flush()  # Flush buffer to disk
                        
                        # If the stream failed, write to the fails file
                        if result_row.get('status') != 'OK':
                            writer_fails.writerow(result_row)
                            f_fails.flush() # Flush buffer to disk
                            
                    except Exception as exc:
                        original_row = future_to_row[future]
                        logging.error(f'Stream {original_row.get("stream_name", "Unknown")} generated an exception: {exc}')
                        
                        # Update row with error info and write to both files
                        original_row.update({'timestamp': datetime.now().isoformat(), 'status': "Exception"})
                        default_errors = {'err_decode': False, 'err_discontinuity': False, 'err_timeout': True}
                        original_row.update(default_errors)
                        
                        writer_out.writerow(original_row)
                        writer_fails.writerow(original_row)
                        f_out.flush()
                        f_fails.flush()

        logging.info(f"Incremental analysis complete. Results saved to {output_csv} and {fails_csv}")

        # --- Final Cleanup: Deduplicate the results file ---
        logging.info(f"Deduplicating final results in {output_csv}...")
        df_final = pd.read_csv(output_csv)
        
        # Ensure consistent data types before dropping duplicates
        df_final['stream_id'] = pd.to_numeric(df_final['stream_id'], errors='coerce')
        df_final.dropna(subset=['stream_id'], inplace=True)
        df_final['stream_id'] = df_final['stream_id'].astype(int)
        
        # Keep the latest entry for each stream_id
        df_final.sort_values(by='timestamp', ascending=True, inplace=True)
        df_final.drop_duplicates(subset=['stream_id'], keep='last', inplace=True)
        
        # Reorder columns to the desired final order
        df_final = df_final.reindex(columns=final_columns)

        df_final.to_csv(output_csv, index=False, na_rep='N/A')
        logging.info(f"Successfully deduplicated and saved final results to {output_csv}")

    except Exception as e:
        logging.error(f"An error occurred during incremental writing or final deduplication: {e}")

# --- Scoring and Sorting ---

def score_streams(config, input_csv, output_csv, update_stats=False):
    """Calculates averages, scores, and sorts streams based on config."""
    settings = config['script_settings']

    # Use a DataFrame for easier manipulation
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Input CSV not found: {input_csv}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        return

    # --- Filtering based on config.ini ---
    try:
        start_range = settings.getint('start_channel', 1)
        end_range = settings.getint('end_channel', 999)
        group_ids_str = settings.get('channel_group_ids', 'ALL').strip()
    except ValueError:
        logging.error("Invalid start_channel or end_channel in config.ini. Aborting score.")
        return

    # Filter by group first if specified
    if group_ids_str.upper() != 'ALL':
        try:
            target_group_ids = {int(gid.strip()) for gid in group_ids_str.split(',')}
            df['channel_group_id'] = pd.to_numeric(df['channel_group_id'], errors='coerce')
            df = df[df['channel_group_id'].isin(target_group_ids)]
        except ValueError:
            logging.error(f"Invalid channel_group_ids in config.ini: '{group_ids_str}'. Aborting score.")
            return

    # Then filter by channel number range
    df['channel_number'] = pd.to_numeric(df['channel_number'], errors='coerce')
    df.dropna(subset=['channel_number'], inplace=True)
    df = df[df['channel_number'].between(start_range, end_range)]

    if df.empty:
        logging.warning(f"No streams found in {input_csv} for the specified filters. Nothing to score.")
        return
    # --- End Filtering ---

    # Convert types, handling potential errors
    df['bitrate_kbps'] = pd.to_numeric(df['bitrate_kbps'], errors='coerce')
    df['frames_decoded'] = pd.to_numeric(df['frames_decoded'], errors='coerce')
    df['frames_dropped'] = pd.to_numeric(df['frames_dropped'], errors='coerce')

    # Group by stream_id and calculate averages
    summary = df.groupby('stream_id').agg(
        avg_bitrate_kbps=('bitrate_kbps', 'mean'),
        avg_frames_decoded=('frames_decoded', 'mean'),
        avg_frames_dropped=('frames_dropped', 'mean')
    ).reset_index()

    # Merge with the latest metadata for each stream
    latest_meta = df.drop_duplicates(subset='stream_id', keep='last')
    summary = pd.merge(summary, latest_meta.drop(columns=['bitrate_kbps', 'frames_decoded', 'frames_dropped']), on='stream_id')

    # Calculate dropped frame percentage
    summary['dropped_frame_percentage'] = (summary['avg_frames_dropped'] / summary['avg_frames_decoded'] * 100).fillna(0)

    # Score and Sort
    RESOLUTION_SCORES = {
        '3840x2160': 100, '1920x1080': 80, '1280x720': 50,
        '960x540': 20, 'Unknown': 0, '': 0
    }
    summary['resolution_score'] = summary['resolution'].astype(str).str.strip().map(RESOLUTION_SCORES).fillna(0)
    
    fps_bonus_points = settings.getint("fps_bonus_points", 55)
    summary['fps_bonus'] = 0
    summary.loc[pd.to_numeric(summary['fps'], errors='coerce').fillna(0) >= 50, 'fps_bonus'] = fps_bonus_points
    
    summary['max_bitrate_for_channel'] = summary.groupby('channel_id')['avg_bitrate_kbps'].transform('max')
    summary['bitrate_score'] = (summary['avg_bitrate_kbps'] / (summary['max_bitrate_for_channel'] * 0.01)).fillna(0)
    
    summary['dropped_frames_penalty'] = summary['dropped_frame_percentage'] * 1

    # Calculate penalty for critical errors
    error_columns = ['err_decode', 'err_discontinuity', 'err_timeout']
    for col in error_columns:
        summary[col] = pd.to_numeric(summary[col], errors='coerce').fillna(0)
    summary['error_penalty'] = summary[error_columns].sum(axis=1) * 25

    summary['score'] = (
        summary['bitrate_score'] +
        summary['resolution_score'] +
        summary['fps_bonus'] -
        summary['dropped_frames_penalty'] -
        summary['error_penalty']
    )
    summary.loc[summary['avg_bitrate_kbps'].isna(), 'score'] = -1
    
    df_sorted = summary.sort_values(by=['channel_number', 'score'], ascending=[True, False])
    
    # Ensure all columns are present for the final CSV
    final_columns = [
        'stream_id', 'channel_number', 'channel_id', 'channel_group_id', 'stream_name', 'stream_url',
        'avg_bitrate_kbps', 'avg_frames_decoded', 'avg_frames_dropped', 'dropped_frame_percentage',
        'fps', 'resolution', 'video_codec', 'audio_codec', 'interlaced_status', 'status', 'score', 'error_penalty'
    ]
    for col in final_columns:
        if col not in df_sorted.columns:
            df_sorted[col] = 'N/A' # Add missing columns with default value

    df_sorted = df_sorted[final_columns] # Ensure correct order
    df_sorted.to_csv(output_csv, index=False, na_rep='N/A')
    logging.info(f"Scored and sorted CSV saved as {output_csv}")
    if update_stats:
        update_stream_stats(output_csv)


def update_stream_stats(csv_path):
    """Updates stream stats on the server from a CSV file."""
    base_url = _get_base_url()
    if not base_url:
        logging.error("DISPATCHARR_BASE_URL not set in .env file.")
        return

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"CSV file not found at: {csv_path}")
        return

    for _, row in df.iterrows():
        stream_id = row.get("stream_id")
        if not stream_id:
            continue

        # Construct the stream stats payload from the CSV row
        stream_stats_payload = {
            "resolution": row.get("resolution"),
            "source_fps": row.get("fps"),
            "video_codec": row.get("video_codec"),
            "audio_codec": row.get("audio_codec"),
            "ffmpeg_output_bitrate": row.get("avg_bitrate_kbps"),
        }

        # Clean up the payload, removing any None values
        stream_stats_payload = {k: v for k, v in stream_stats_payload.items() if pd.notna(v)}

        if not stream_stats_payload:
            logging.info(f"No data to update for stream {stream_id}. Skipping.")
            continue

        # Construct the URL for the specific stream
        stream_url = f"{base_url}/api/channels/streams/{int(stream_id)}/"

        try:
            # Fetch the existing stream data to get the current stream_stats
            existing_stream_data = fetch_data_from_url(stream_url)
            if not existing_stream_data:
                logging.warning(
                    f"Could not fetch existing data for stream {stream_id}. Skipping."
                )
                continue

            # Get the existing stream_stats or an empty dict
            existing_stats = existing_stream_data.get("stream_stats") or {}
            if isinstance(existing_stats, str):
                try:
                    existing_stats = json.loads(existing_stats)
                except json.JSONDecodeError:
                    existing_stats = {}

            # Merge the existing stats with the new payload
            updated_stats = {**existing_stats, **stream_stats_payload}

            # Send the PATCH request with the updated stream_stats
            patch_payload = {"stream_stats": updated_stats}
            logging.info(f"Updating stream {stream_id} with: {patch_payload}")
            patch_request(stream_url, patch_payload)

        except Exception as e:
            logging.error(f"An error occurred while updating stream {stream_id}: {e}")


# --- Reordering Streams ---

def reorder_streams(config, input_csv):
    """Reorders streams in Dispatcharr based on the scored and sorted CSV."""
    logging.info(f"Reordering streams based on {input_csv}...")
    settings = config['script_settings']
    try:
        start_range = settings.getint('start_channel', 1)
        end_range = settings.getint('end_channel', 999)
        group_ids_str = settings.get('channel_group_ids', 'ALL').strip()
        logging.info(f"Applying reordering to channel range: {start_range}-{end_range} and groups: {group_ids_str}")
    except ValueError:
        logging.error("Invalid start_channel or end_channel in config.ini. Aborting reorder.")
        return

    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        logging.error(f"Error: {input_csv} not found. Please run the 'score' command first.")
        return

    # Filter by group first if specified
    if group_ids_str.upper() != 'ALL':
        try:
            target_group_ids = {int(gid.strip()) for gid in group_ids_str.split(',')}
            df['channel_group_id'] = pd.to_numeric(df['channel_group_id'], errors='coerce')
            df = df[df['channel_group_id'].isin(target_group_ids)]
        except ValueError:
            logging.error(f"Invalid channel_group_ids in config.ini: '{group_ids_str}'. Aborting reorder.")
            return

    # Then filter by channel number range
    df['channel_number'] = pd.to_numeric(df['channel_number'], errors='coerce')
    df.dropna(subset=['channel_number'], inplace=True)
    df = df[df['channel_number'].between(start_range, end_range)]

    if df.empty:
        logging.warning(f"No streams found in {input_csv} for the specified filters. Nothing to reorder.")
        return

    df['stream_id'] = pd.to_numeric(df['stream_id'], errors='coerce')
    df['channel_id'] = pd.to_numeric(df['channel_id'], errors='coerce')
    df.dropna(subset=['stream_id', 'channel_id'], inplace=True)
    df['stream_id'] = df['stream_id'].astype(int)
    df['channel_id'] = df['channel_id'].astype(int)

    grouped = df.groupby("channel_id")

    for channel_id, group in grouped:
        sorted_stream_ids_from_csv = group["stream_id"].tolist()
        
        current_streams_from_api = fetch_channel_streams(channel_id)
        if current_streams_from_api is None:
            logging.warning(f"Could not fetch current streams for channel ID {channel_id}. Skipping reorder.")
            continue

        current_stream_ids_set = {s['id'] for s in current_streams_from_api}
        validated_sorted_ids = [sid for sid in sorted_stream_ids_from_csv if sid in current_stream_ids_set]
        csv_ids_set = set(sorted_stream_ids_from_csv)
        new_unscored_ids = [sid for sid in current_stream_ids_set if sid not in csv_ids_set]
        final_stream_id_list = validated_sorted_ids + new_unscored_ids
        
        if not final_stream_id_list:
            logging.warning(f"No valid streams to reorder for channel ID {channel_id}. Skipping.")
            continue
        
        try:
            update_channel_streams(channel_id, final_stream_id_list)
            logging.info(f"Successfully reordered streams for channel ID {channel_id}.")
        except Exception as e:
            logging.error(f"An exception occurred while reordering streams for channel ID {channel_id}: {e}")

    logging.info("Stream reordering complete.")

def retry_failed_streams(config, input_csv, fails_csv, ffmpeg_duration, idet_frames, timeout, max_workers):
    """Retries analysis for streams that previously failed."""
    if not os.path.exists(input_csv):
        logging.error(f"Input file not found: {input_csv}. Cannot retry failed streams.")
        return

    if not _check_ffmpeg_installed():
        sys.exit(1)

    all_rows = []
    fieldnames = []
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames or []
        all_rows = list(reader)

    required_cols = ['video_codec', 'audio_codec', 'interlaced_status', 'status']
    for col in required_cols:
        if col not in fieldnames:
            fieldnames.append(col)

    failed_streams = [row for row in all_rows if row.get('status') != 'OK']

    if not failed_streams:
        logging.info("No failed streams to retry.")
        return

    logging.info(f"Retrying analysis for {len(failed_streams)} failed streams...")

    updated_rows = {row['stream_id']: row for row in all_rows}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {executor.submit(_analyze_stream_task, row, ffmpeg_duration, idet_frames, timeout, 0, 0, config): row for row in failed_streams}
        for future in future_to_row:
            try:
                result_row = future.result()
                stream_id = result_row.get('stream_id')
                if stream_id:
                    updated_rows[stream_id] = result_row
            except Exception as exc:
                original_row = future_to_row[future]
                logging.error(f'Stream {original_row.get("stream_name", "Unknown")} generated an exception during retry: {exc}')
                original_row.update({'timestamp': datetime.now().isoformat(), 'status': "Retry Exception"})
                updated_rows[original_row['stream_id']] = original_row

    with open(input_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows.values())

    new_fails = [row for row in updated_rows.values() if row.get('status') != 'OK']
    with open(fails_csv, 'w', newline='', encoding='utf-8') as fails_outfile:
        writer = csv.DictWriter(fails_outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_fails)

    logging.info(f"Retry complete. Updated {input_csv} and {fails_csv}.")

def main():
    """Main function to parse arguments and call the appropriate function."""
    load_dotenv()
    config = load_config()

    parser = argparse.ArgumentParser(
        description="A tool for managing and analyzing Dispatcharr IPTV streams.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser('login', help='Authenticate with Dispatcharr and save the token.')
    
    fetch_parser = subparsers.add_parser('fetch', help='Fetch channel and stream information.')
    fetch_parser.add_argument('--output', type=str, default='csv/02_grouped_channel_streams.csv')

    analyze_parser = subparsers.add_parser('analyze', help='Analyze IPTV streams.')
    analyze_parser.add_argument('--input', type=str, default='csv/02_grouped_channel_streams.csv')
    analyze_parser.add_argument('--output', type=str, default='csv/03_iptv_stream_measurements.csv')
    analyze_parser.add_argument('--fails_output', type=str, default='csv/04_fails.csv')
    analyze_parser.add_argument('--duration', type=int, default=10, help='Duration in seconds for ffmpeg to analyze stream.')
    analyze_parser.add_argument('--idet-frames', type=int, default=500)
    analyze_parser.add_argument('--timeout', type=int, default=30)
    analyze_parser.add_argument('--workers', type=int, default=8)
    analyze_parser.add_argument('--retries', type=int, default=1)
    analyze_parser.add_argument('--retry-delay', type=int, default=10)

    score_parser = subparsers.add_parser('score', help='Score and sort streams.')
    score_parser.add_argument('--input', type=str, default='csv/03_iptv_stream_measurements.csv')
    score_parser.add_argument('--output', type=str, default='csv/05_iptv_streams_scored_sorted.csv')
    score_parser.add_argument('--update-stats', action='store_true', help='Update stream stats on the server after scoring.')

    reorder_parser = subparsers.add_parser('reorder', help='Reorder streams in Dispatcharr.')
    reorder_parser.add_argument('--input', type=str, default='csv/05_iptv_streams_scored_sorted.csv')

    retry_parser = subparsers.add_parser('retry', help='Retry analysis for failed streams.')
    retry_parser.add_argument('--input', type=str, default='csv/03_iptv_stream_measurements.csv')
    retry_parser.add_argument('--fails-output', type=str, default='csv/04_fails.csv')
    retry_parser.add_argument('--duration', type=int, default=20)
    retry_parser.add_argument('--idet-frames', type=int, default=500)
    retry_parser.add_argument('--timeout', type=int, default=30)
    retry_parser.add_argument('--workers', type=int, default=8)

    args = parser.parse_args()

    if args.command == 'login':
        login()
    elif args.command == 'fetch':
        fetch_streams(config, args.output)
    elif args.command == 'analyze':
        analyze_streams(config, args.input, args.output, args.fails_output, args.duration, args.idet_frames, args.timeout, args.workers, args.retries, args.retry_delay)
    elif args.command == 'score':
        score_streams(config, args.input, args.output, args.update_stats)
    elif args.command == 'reorder':
        reorder_streams(config, args.input)
    elif args.command == 'retry':
        retry_failed_streams(config, args.input, args.fails_output, args.duration, args.idet_frames, args.timeout, args.workers)
    else:
        logging.info("No command specified. Running default pipeline: fetch -> analyze -> score -> reorder")
        fetch_streams(config, 'csv/02_grouped_channel_streams.csv')
        analyze_streams(config, 'csv/02_grouped_channel_streams.csv', 'csv/03_iptv_stream_measurements.csv', 'csv/04_fails.csv', 20, 500, 30, 8, 1, 10)
        score_streams(config, 'csv/03_iptv_stream_measurements.csv', 'csv/05_iptv_streams_scored_sorted.csv', update_stats=True)
        reorder_streams(config, 'csv/05_iptv_streams_scored_sorted.csv')

if __name__ == "__main__":
    main()
