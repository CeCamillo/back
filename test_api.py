import requests
import json
import pandas as pd
import io

# Base URL of the running FastAPI application
BASE_URL = "http://127.0.0.1:5000"

# Sample data mimicking the UNSW-NB15 dataset features.
# The model was trained on data where 'attack_cat' was 'Worms', 'Backdoor', or 'Normal'.
# These samples represent the raw data before one-hot encoding and scaling,
# which is what the API expects.

# Sample 1: A "Normal" traffic example
normal_data = {
    "dur": 0.000011, "proto": "udp", "service": "-", "state": "INT", "spkts": 2,
    "dpkts": 0, "sbytes": 104, "dbytes": 0, "rate": 90909.0902, "sttl": 254,
    "dttl": 0, "sload": 37818180.0, "dload": 0.0, "sloss": 0, "dloss": 0,
    "sinpkt": 0.000011, "dinpkt": 0.0, "sjit": 0.0, "djit": 0.0, "swin": 0,
    "stcpb": 0, "dtcpb": 0, "dwin": 0, "tcprtt": 0.0, "synack": 0.0, "ackdat": 0.0,
    "smean": 52, "dmean": 0, "trans_depth": 0, "response_body_len": 0,
    "ct_srv_src": 2, "ct_state_ttl": 2, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1,
    "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 1, "is_ftp_login": 0,
    "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1, "ct_srv_dst": 2,
    "is_sm_ips_ports": 0
}

# Sample 2: A "Backdoor" attack example
backdoor_data = {
    "dur": 0.043303, "proto": "tcp", "service": "-", "state": "FIN", "spkts": 10,
    "dpkts": 8, "sbytes": 838, "dbytes": 354, "rate": 392.5825, "sttl": 31,
    "dttl": 29, "sload": 122298.7, "dload": 56231.6, "sloss": 2, "dloss": 2,
    "sinpkt": 4.636111, "dinpkt": 5.408571, "sjit": 298.5429, "djit": 164.2429,
    "swin": 255, "stcpb": 1061872917, "dtcpb": 1856710408, "dwin": 255,
    "tcprtt": 0.000783, "synack": 0.000693, "ackdat": 0.00009, "smean": 84,
    "dmean": 44, "trans_depth": 0, "response_body_len": 0, "ct_srv_src": 3,
    "ct_state_ttl": 0, "ct_dst_ltm": 1, "ct_src_dport_ltm": 1,
    "ct_dst_sport_ltm": 1, "ct_dst_src_ltm": 3, "is_ftp_login": 0,
    "ct_ftp_cmd": 0, "ct_flw_http_mthd": 0, "ct_src_ltm": 1, "ct_srv_dst": 3,
    "is_sm_ips_ports": 0
}


def test_feature_importance_endpoint():
    """Tests the new /features/importances endpoint."""
    print("--- Testing: Feature Importance Endpoint ---")
    try:
        response = requests.get(f"{BASE_URL}/features/importances")
        response.raise_for_status()
        response_data = response.json()
        print("Status Code:", response.status_code)
        print("Response contains", len(response_data), "features.")
        # Print first feature as a sample
        if response_data:
            print("Top feature example:", json.dumps(response_data[0], indent=2))
        print("-" * 35 + "\n")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")


def test_single_json_prediction():
    """Tests the /predict endpoint with a single JSON object for the new response structure."""
    print("--- Testing: Single JSON Prediction (Enhanced Response) ---")
    try:
        response = requests.post(f"{BASE_URL}/predict", json=normal_data)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Response JSON:", json.dumps(response.json(), indent=2))
        # Add checks for new structure
        data = response.json()
        assert 'input_data' in data
        assert 'result' in data
        assert 'prediction' in data['result']
        assert 'confidence' in data['result']
        print("SUCCESS: Response has the new enhanced structure.")
        print("-" * 35 + "\n")
    except (requests.exceptions.RequestException, AssertionError) as e:
        print(f"An error occurred or assertion failed: {e}")


def test_batch_json_prediction():
    """Tests the /predict endpoint with a list of JSON objects for the new response structure."""
    print("--- Testing: Batch JSON Prediction (Enhanced Response) ---")
    try:
        data = [normal_data, backdoor_data]
        response = requests.post(f"{BASE_URL}/predict", json=data)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Response JSON:", json.dumps(response.json(), indent=2))
        # Add checks for new structure
        res_data = response.json()
        assert isinstance(res_data, list)
        assert 'input_data' in res_data[0]
        assert 'result' in res_data[0]
        print("SUCCESS: Response has the new enhanced structure.")
        print("-" * 35 + "\n")
    except (requests.exceptions.RequestException, AssertionError) as e:
        print(f"An error occurred or assertion failed: {e}")


def test_csv_prediction():
    """Tests the /predict/csv endpoint for the new response structure."""
    print("--- Testing: CSV File Prediction (Enhanced Response) ---")
    try:
        # Create a DataFrame and convert it to a CSV in-memory file
        df = pd.DataFrame([normal_data, backdoor_data])
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Rewind buffer to the beginning

        # 'files' should be a dictionary where the key is the name of the file field
        files = {'file': ('test_data.csv', csv_buffer, 'text/csv')}

        response = requests.post(f"{BASE_URL}/predict/csv", files=files)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Response JSON:", json.dumps(response.json(), indent=2))
        # Add checks for new structure
        res_data = response.json()
        assert isinstance(res_data, list)
        assert 'input_data' in res_data[0]
        assert 'result' in res_data[0]
        print("SUCCESS: Response has the new enhanced structure.")
        print("-" * 35 + "\n")
    except (requests.exceptions.RequestException, AssertionError) as e:
        print(f"An error occurred or assertion failed: {e}")


if __name__ == "__main__":
    # Ensure the server is running before executing this script
    print("Starting API tests...")
    test_feature_importance_endpoint()
    test_single_json_prediction()
    test_batch_json_prediction()
    test_csv_prediction()
    print("All tests completed.")
