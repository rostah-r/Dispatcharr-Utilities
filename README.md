# Dispatcharr Utilities

This tool provides a set of utilities for analyzing and managing IPTV streams in a Dispatcharr instance.

## Features

-   **Login:** Authenticate with your Dispatcharr instance and save the token for future use.
-   **Fetch:** Fetch channel and stream information from your Dispatcharr instance and save it to CSV files.
-   **Analyze:** Analyze IPTV streams using `ffmpeg` to gather metrics like bitrate, resolution, and FPS.
-   **Score:** Score and sort streams based on the analyzed metrics to identify the best quality streams for each channel.
-   **Reorder:** Reorder streams in your Dispatcharr instance based on the generated scores.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd dispatcharr-utilities
    ```

2.  **Install dependencies:**
    Make sure you have Python 3 and `ffmpeg` installed and available in your system's PATH. You can install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create your configuration file:**
    Copy the `.env.template` file to `.env`:
    ```bash
    cp .env.template .env
    ```

2.  **Edit `.env`:**
    Open the newly created `.env` file and fill in your Dispatcharr instance details and adjust any parameters as needed.

    ```ini
    DISPATCHARR_BASE_URL="http://your-dispatcharr-instance.com:9191"
    DISPATCHARR_USER="your-username"
    DISPATCHARR_PASS="your-password"
    # DISPATCHER_TOKEN can be left blank, the script will fill it in
    DISPATCHARR_TOKEN=''
    # Channel range for fetching streams (used by 'fetch' command).
    # Set START_CHANNEL and END_CHANNEL to encompass all your Dispatcharr Channel numbers.
    # Example: START_CHANNEL="1" and END_CHANNEL="1" to analyze only Channel 1.
    # Example: START_CHANNEL="1" and END_CHANNEL="10" to analyze a range of channels.
    START_CHANNEL="1"
    END_CHANNEL="999"
    # Number of days to keep stream analysis measurements (used by 'analyze' command).
    # Streams older than this will be re-analyzed.
    # Set to "0" to force re-analysis of all channels in your specified range.
    STREAM_LAST_MEASURED="7"
    # Bonus points added to a stream's score if its FPS is 50 or higher (used by 'score' command).
    # Set to "0" if you prefer streams to be ranked solely by Bitrate & Resolution.
    # A value like "55" is recommended to boost channels with higher FPS, useful for sports streams.
    FPS_BONUS_POINTS="55"
    ```

## Usage

The tool is used via the command line. Here is the basic syntax:

```bash
python dispatcharr-stream-sorter.py [command] [options]
```

If no command is specified, the tool will run through a default pipeline: `login` -> `fetch` -> `analyze` -> `score` -> `reorder`.

### Commands

#### `login`

Authenticates with your Dispatcharr instance and saves the access token to your `.env` file. This command must be run successfully before other commands can interact with the API.

```bash
python dispatcharr-stream-sorter.py login
```

#### `fetch`

Retrieves channel and stream metadata from your Dispatcharr instance and saves it into CSV files (`csv/00_channel_groups.csv`, `csv/01_channels_metadata.csv`, and `csv/02_grouped_channel_streams.csv`).

```bash
python dispatcharr-stream-sorter.py fetch [--output <output_file>]
```

#### `analyze`

Processes the streams listed in the input CSV (defaulting to `csv/02_grouped_channel_streams.csv`) using `ffmpeg` to measure bitrate, FPS, and resolution. Results are appended to `csv/03_iptv_stream_measurements.csv`.

```bash
python dispatcharr-stream-sorter.py analyze [--input <input_file>] [--output <output_file>] [--fails_output <fails_file>] [--duration <seconds>] [--timeout <seconds>] [--workers <number>]
```

#### `score`

Calculates average metrics for each stream from the analysis results (defaulting to `csv/03_iptv_stream_measurements.csv`), then scores and sorts them based on quality criteria. The final sorted list is saved to `csv/05_iptv_streams_scored_sorted.csv`.

```bash
python dispatcharr-stream-sorter.py score [--input <input_file>] [--output <output_file>]
```

#### `reorder`

Reads the scored and sorted stream data from the input CSV (defaulting to `csv/05_iptv_streams_scored_sorted.csv`) and updates the stream order for each channel in your Dispatcharr instance via the API.

```bash
python dispatcharr-stream-sorter.py reorder [--input <input_file>]
```

### Example Workflow

To run the complete analysis and reordering pipeline, simply execute the script without any arguments:

```bash
python dispatcharr-stream-sorter.py
```

This single command will perform the following steps automatically:

1.  **Login:** Authenticate with your Dispatcharr instance.
2.  **Fetch:** Retrieve channel and stream information.
3.  **Analyze:** Measure stream quality using `ffmpeg`.
4.  **Score:** Calculate scores and sort streams based on quality.
5.  **Reorder:** Update the stream order in your Dispatcharr instance.

You can also run individual commands (e.g., `python dispatcharr-stream-sorter.py fetch`) if you need to perform specific steps separately.
