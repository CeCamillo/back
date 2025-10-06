1. Machine Learning Model: Training and Preparation (Offline Process)
This is the foundational stage where the "intelligence" of the application is created. This process is done once, before the application is deployed.

Data Loading and Cleaning:

Load the UNSW-NB15 training dataset, which contains examples of normal network traffic and various attacks, including worms and backdoors.   

Perform initial data cleaning and preparation.

Data Preprocessing:

Convert categorical features (like protocol type) into a numerical format using one-hot encoding.

Scale all numerical features to a standard range (e.g., 0 to 1) to ensure the model treats all features fairly.

Handling Class Imbalance:

Apply a specific technique to address the fact that there are far fewer attack samples than normal ones. This involves either creating synthetic minority class samples (using a method like SMOTE) or adjusting the model's learning algorithm to place a higher penalty on misclassifying the rare attack classes. This step is crucial for making the model effective at detecting rare threats.   

Model Training and Tuning:

Train a Random Forest classifier on the preprocessed and balanced data.   

Systematically tune the model's hyperparameters to find the optimal settings for the best performance.

Saving the Pipeline:

Persist (save to disk) the entire trained pipeline using a library like joblib. This includes three essential files:   

model.joblib: The trained Random Forest model itself.

scaler.joblib: The scaler object used to normalize the training data.

model_columns.joblib: The exact list and order of data columns the model was trained on.

2. Backend API (The Server)
The backend is the engine of the application. It runs on a server, waits for requests from the user interface, and performs the actual analysis.

Initialization:

Upon starting, the server immediately loads the three saved files (model.joblib, scaler.joblib, model_columns.joblib) into memory, ready for use.   

Create an API Endpoint:

Establish a specific URL route (e.g., /predict) that listens for incoming POST requests containing a data file.   

Process Incoming Data:

When a user uploads a file, the server receives it and reads it into a Pandas DataFrame. It must support both CSV and Parquet file formats.   

It then applies the exact same preprocessing steps used during training, using the loaded artifacts to ensure consistency:

Reorders and validates columns against model_columns.joblib.

Applies the saved scaler from scaler.joblib to the numerical data.

Perform Prediction:

The preprocessed user data is fed into the loaded Random Forest model (model.joblib).   

The model classifies each row (each network connection) as 'Normal', 'Worm', or 'Backdoor'.

Return Results:

The server aggregates the prediction results.

It formats this information into a structured JSON response, including counts of total connections, normal connections, and detected anomalies, along with the calculated performance metrics.   

This JSON response is sent back to the frontend.

3. Frontend Application (The User Dashboard)
This is the user-facing web interface that allows for interaction with the system.

File Upload Interface:

Display a simple and intuitive web page with a title, a short description, and a file uploader widget.   

The user can use this widget to select and upload their network data file (in CSV or Parquet format).

Communicate with the Backend:

Once the user uploads a file, the frontend sends the file data in an HTTP request to the backend's /predict endpoint.   

It then waits to receive the JSON analysis from the server.

Display the Analysis Report:

Upon receiving the JSON response, the dashboard parses the data and presents it in a clear, graphical report.   

The report must display:

Summary Counts: Total connections analyzed, number of normal connections, and the number of detected worms and backdoors.

Performance Metrics: Key evaluation metrics like Precision, Recall, and F1-Score to provide a scientifically sound assessment of the analysis quality.   

Visualizations: Use charts and tables to make the results easy to understand at a glance.   
