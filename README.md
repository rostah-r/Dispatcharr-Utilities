# Dispatcharr Utilities

This repository contains Python scripts designed to interact with a Dispatcharr instance, enabling advanced stream management, analysis, and bulk data operations.

## Project Structure

-   `api_utils.py`: Contains centralized utility functions for interacting with the Dispatcharr API, including authentication, data fetching, and channel/stream updates. All other scripts leverage these functions.
-   `channels_upload.py`: Facilitates the bulk upload and synchronization of channels to Dispatcharr from a CSV file.
-   `config.ini`: The primary configuration file for all scripts. It defines script settings, channel filtering ranges, and scoring parameters.
-   `dispatcharr-stream-sorter.py`: The main script for fetching, analyzing, scoring, and reordering streams within Dispatcharr. It uses `ffmpeg` for stream analysis and integrates with `config.ini` for its operational parameters.
-   `groups_upload.py`: Enables the bulk upload and synchronization of channel groups to Dispatcharr from a CSV file.
-   `requirements.txt`: Lists all Python dependencies required to run the scripts.
-   `csv/`: This directory stores various CSV files used by the scripts:
    -   `00_channel_groups.csv`: Stores fetched channel group metadata.
    -   `01_channels_metadata.csv`: Stores fetched channel metadata, including new IDs after channel restoration.
    -   `02_grouped_channel_streams.csv`: Contains raw stream data fetched from Dispatcharr, including `channel_group_id`.
    -   `03_iptv_stream_measurements.csv`: Stores detailed analysis results for each stream.
    -   `04_fails.csv`: Logs streams that failed analysis.
    -   `05_iptv_streams_scored_sorted.csv`: Contains streams sorted by quality score.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd dispatcharr-utilities
    ```

2.  **Install dependencies:**
    Ensure you have Python 3 and `ffmpeg` installed and available in your system's PATH. Then, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

All non-sensitive configuration is managed through `config.ini`. Sensitive information (like API keys) should be stored in a `.env` file.

1.  **Create your environment file (`.env`):**
    Copy `.env.template` to `.env` and fill in your Dispatcharr credentials:
    ```bash
    cp .env.template .env
    ```
    Example `.env` content:
    ```ini
    DISPATCHARR_BASE_URL="http://your-dispatcharr-instance.com:9191"
    DISPATCHARR_USER="your-username"
    DISPATCHARR_PASS="your-password"
    DISPATCHARR_TOKEN='' # This will be populated automatically after login
    ```

2.  **Configure `config.ini`:**
    Open `config.ini` and adjust settings under the `[script_settings]` section. This includes input/output file paths, channel filtering ranges (by number and group ID), and scoring parameters.

## Usage

### `dispatcharr-stream-sorter.py`

This is the primary script for managing stream quality. It automates fetching, analyzing, scoring, and reordering streams.

```bash
python dispatcharr-stream-sorter.py [command] [options]
```

If no command is specified, it runs a default pipeline: `fetch` -> `analyze` -> `score` -> `reorder` (with stream stats updated).

**Commands:**

-   `login`: Authenticates with Dispatcharr and saves the token.
    ```bash
    python dispatcharr-stream-sorter.py login
    ```
-   `fetch`: Fetches channel and stream metadata. Output includes `channel_group_id`.
    ```bash
    python dispatcharr-stream-sorter.py fetch [--output <output_file>]
    ```
-   `analyze`: Analyzes streams using `ffmpeg`. Configurable duration, frames, and workers.
    ```bash
    python dispatcharr-stream-sorter.py analyze [--input <input_file>] [--output <output_file>] [--fails_output <fails_file>] [--duration <seconds>] [--idet-frames <frames>] [--timeout <seconds>] [--workers <number>] [--retries <number>] [--retry-delay <seconds>]
    ```
-   `score`: Calculates scores and sorts streams based on quality metrics. Can also update stream stats on the server.
    ```bash
    python dispatcharr-stream-sorter.py score [--input <input_file>] [--output <output_file>] [--update-stats]
    ```
-   `reorder`: Updates stream order in Dispatcharr based on scores.
    ```bash
    python dispatcharr-stream-sorter.py reorder [--input <input_file>]
    ```

### Upload Utilities (`groups_upload.py`, `channels_upload.py`)

These scripts are used for bulk creation or updating of channel groups and channels from CSV files. They are particularly useful for initial setup or restoring data.

-   **`groups_upload.py`**
    Synchronizes channel groups using `csv/groups_template.csv`.
    ```bash
    python groups_upload.py
    ```

-   **`channels_upload.py`**
    Synchronizes channels using `csv/channels_template.csv` or a specified CSV. It automatically refreshes `csv/01_channels_metadata.csv`.
    ```bash
    python channels_upload.py [--csv_file <path_to_csv>]
    ```
