import requests
import pandas as pd
import io

# Base URL of the running Flask application
BASE_URL = 'http://127.0.0.1:5000'

def test_json_endpoint():
    """
    Tests the /predict endpoint with a sample JSON payload.
    """
    print("--- Testing JSON Endpoint (/predict) ---")

    # Sample data for prediction (can be one or more records)
    sample_data = [
        # A record that might look like a 'Worm'
        {
            'proto': 'tcp', 'state': 'FIN', 'dur': 0.05, 'sbytes': 1500, 'dbytes': 200,
            'sttl': 31, 'dttl': 29, 'sloss': 5, 'dloss': 1, 'service': '-',
            'sload': 250000, 'dload': 35000, 'spkts': 10, 'dpkts': 4
        },
        # A record that might look 'Normal'
        {
            'proto': 'udp', 'state': 'CON', 'dur': 0.1, 'sbytes': 100, 'dbytes': 80,
            'sttl': 60, 'dttl': 62, 'sloss': 0, 'dloss': 0, 'service': 'dns',
            'sload': 8000, 'dload': 6400, 'spkts': 1, 'dpkts': 1
        }
    ]

    try:
        response = requests.post(f"{BASE_URL}/predict", json=sample_data)
        response.raise_for_status()  # Raise an exception for bad status codes
        print("Response from server:")
        print(response.json())
        print("-" * 35 + "\n")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        print("Is the Flask server running? Start it with 'python app.py'")

def test_csv_endpoint():
    """
    Tests the /predict/csv endpoint by uploading a sample CSV file.
    """
    print("--- Testing CSV Endpoint (/predict/csv) ---")

    # Create a sample DataFrame and convert it to a CSV in memory
    sample_df = pd.DataFrame([
        {
            'proto': 'tcp', 'state': 'FIN', 'dur': 1.2, 'sbytes': 2400, 'dbytes': 5000,
            'sttl': 31, 'dttl': 29, 'sloss': 8, 'dloss': 12, 'service': 'http',
            'sload': 16000, 'dload': 33333, 'spkts': 20, 'dpkts': 15
        },
        {
            'proto': 'udp', 'state': 'INT', 'dur': 0.0001, 'sbytes': 50, 'dbytes': 0,
            'sttl': 254, 'dttl': 0, 'sloss': 0, 'dloss': 0, 'service': 'dns',
            'sload': 4000000, 'dload': 0, 'spkts': 1, 'dpkts': 0
        }
    ])

    # Use io.StringIO to treat the string as a file
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0) # Rewind the buffer to the beginning

    try:
        # The file key must be 'file' as expected by the Flask app
        files = {'file': ('test_data.csv', csv_buffer, 'text/csv')}
        response = requests.post(f"{BASE_URL}/predict/csv", files=files)
        response.raise_for_status()
        print("Response from server:")
        print(response.json())
        print("-" * 35 + "\n")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        print("Is the Flask server running? Start it with 'python app.py'")

if __name__ == '__main__':
    test_json_endpoint()
    test_csv_endpoint() 