# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import os
# import logging

# # Configuration
# TEST_DATA_DIR = 'data/lunar/test/data/S15_GradeA/'
# MODEL_FILE = 'moonquake_model.pth'  # Path to your saved model
# SEQUENCE_LENGTH = 100  # Same as used during training

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class LSTMClassifier(nn.Module):
#     def __init__(self, input_size):
#         super(LSTMClassifier, self).__init__()
#         self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
#         self.dropout1 = nn.Dropout(0.2)
#         self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
#         self.dropout2 = nn.Dropout(0.2)
#         self.fc = nn.Linear(32, 1)  # Matches train.py

#     def forward(self, x):
#         out, _ = self.lstm1(x)
#         out = self.dropout1(out)
#         out, _ = self.lstm2(out)
#         out = self.dropout2(out)
#         out = self.fc(out[:, -1, :])
#         return out.squeeze()

# def read_csv_file(csv_file):
#     # Read the CSV and handle missing values represented by -1 or -1.0
#     data = pd.read_csv(csv_file)
#     data.replace(-1, np.nan, inplace=True)
#     data.replace(-1.0, np.nan, inplace=True)
#     return data

# def clean_data(data):
#     # Handle missing values
#     data = data.copy()
#     data['velocity(m/s)'] = data['velocity(m/s)'].ffill().bfill().fillna(0)
#     return data

# def normalize_data(data):
#     # Normalize the velocity data
#     mean = data['velocity(m/s)'].mean()
#     std = data['velocity(m/s)'].std()
#     data['velocity(m/s)'] = (data['velocity(m/s)'] - mean) / std
#     return data

# def main():
#     # Check for GPU
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Load the model
#     input_size = 4  # Update based on your actual number of features
#     model = LSTMClassifier(input_size=input_size).to(device)
    
#     if not os.path.exists(MODEL_FILE):
#         print(f"Model file not found: {MODEL_FILE}")
#         return

#     try:
#         # Load state_dict with weights_only=True to handle FutureWarning
#         state_dict = torch.load(MODEL_FILE, map_location=device, weights_only=True)
#         model.load_state_dict(state_dict)
#         print(f"Loaded model from {MODEL_FILE}")
#     except AttributeError:
#         # Fallback if weights_only is not supported
#         state_dict = torch.load(MODEL_FILE, map_location=device)
#         model.load_state_dict(state_dict)
#         print(f"Loaded model from {MODEL_FILE} (weights_only not supported)")

#     except RuntimeError as e:
#         print("Error loading model state_dict. Ensure the model architecture matches the saved model.")
#         print(str(e))
#         return

#     model.eval()

#     # Test file
#     test_file = 'xa.s15.00.mhz.1973-04-04HR00_evid00098'  # Replace with your test file name (without extension)
#     test_csv_file = os.path.join(TEST_DATA_DIR, f'{test_file}.csv')

#     if not os.path.exists(test_csv_file):
#         print(f"Test file not found: {test_csv_file}")
#         return

#     # Load and preprocess test data
#     test_data = read_csv_file(test_csv_file)
#     test_data = clean_data(test_data)
#     test_data = normalize_data(test_data)
#     test_velocity = test_data['velocity(m/s)'].values

#     # Compute additional features if necessary (e.g., derivative, rolling statistics)
#     # Ensure these match the features used during training
#     derivative = np.diff(test_velocity, prepend=test_velocity[0])
#     rolling_mean = pd.Series(test_velocity).rolling(window=10, min_periods=1).mean().values
#     rolling_std = pd.Series(test_velocity).rolling(window=10, min_periods=1).std().fillna(0).values

#     # Stack features
#     test_features = np.stack((test_velocity, derivative, rolling_mean, rolling_std), axis=1)

#     # Reshape test data
#     # No need to reshape if features are already correctly stacked

#     # Create sequences for test data
#     logger.info("Creating sequences for test data")
#     num_sequences = len(test_features) - SEQUENCE_LENGTH + 1
#     if num_sequences <= 0:
#         print("No test sequences created. Check the sequence length and test data size.")
#         return
#     X_test = np.array([test_features[i:i+SEQUENCE_LENGTH] for i in range(num_sequences)])

#     # Convert to tensor
#     X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

#     # Make predictions
#     print("Making predictions on test data...")
#     predictions = []
#     batch_size = 128  # Adjust as needed
#     with torch.no_grad():
#         for i in tqdm(range(0, len(X_test), batch_size), desc="Predicting"):
#             X_batch = X_test[i:i+batch_size]
#             outputs = model(X_batch)
#             probs = torch.sigmoid(outputs)  # Apply Sigmoid to outputs
#             predictions.extend(probs.cpu().numpy())

#     predictions = np.array(predictions)

#     # Detect events (you may need to adjust the threshold)
#     threshold = 0.5
#     events = np.where(predictions > threshold)[0]

#     # Plot test data and predictions
#     plt.figure(figsize=(15, 5))
#     plt.plot(test_data['time_rel(sec)'], test_velocity, label='Normalized Velocity')
#     plt.plot(test_data['time_rel(sec)'][SEQUENCE_LENGTH-1:], predictions, label='Event Probability', color='red')

#     # Mark detected events on the plot
#     event_times = test_data['time_rel(sec)'].iloc[events + SEQUENCE_LENGTH - 1]
#     for event_time in event_times:
#         plt.axvline(x=event_time, color='green', linestyle='--', label='Detected Event')

#     # Remove duplicate labels in legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())

#     plt.title('Test Data and Predictions with Detected Events')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Normalized Velocity / Event Probability')
#     plt.show()

#     print("Detected events at:")
#     for event_time in event_times:
#         print(f"Time: {event_time} seconds")

#     # If you have actual quake times, you can mark them as well
#     # For example, if you have a list of actual quake times:
#     # actual_quake_times = [...]  # Replace with actual times if available
#     # for quake_time in actual_quake_times:
#     #     plt.axvline(x=quake_time, color='blue', linestyle='-', label='Actual Quake')

# if __name__ == "__main__":
#     main()


# main.py
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],  # Allows all origins
  allow_credentials=True,
  allow_methods=["*"],  # Allows all methods
  allow_headers=["*"],  # Allows all headers
)

class LSTMClassifier(nn.Module):
  def __init__(self, input_size):
      super(LSTMClassifier, self).__init__()
      self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
      self.dropout1 = nn.Dropout(0.2)
      self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
      self.dropout2 = nn.Dropout(0.2)
      self.fc = nn.Linear(32, 1)

  def forward(self, x):
      out, _ = self.lstm1(x)
      out = self.dropout1(out)
      out, _ = self.lstm2(out)
      out = self.dropout2(out)
      out = self.fc(out[:, -1, :])
      return out.squeeze()

# Load the model (you'll need to adjust the path)
MODEL_FILE = 'moonquake_model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(input_size=4).to(device)
model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
model.eval()

SEQUENCE_LENGTH = 100

def read_csv_file(file):
  content = file.file.read()
  data = pd.read_csv(io.StringIO(content.decode('utf-8')))
  data.replace(-1, np.nan, inplace=True)
  data.replace(-1.0, np.nan, inplace=True)
  return data

def clean_data(data):
  data = data.copy()
  data['velocity(m/s)'] = data['velocity(m/s)'].ffill().bfill().fillna(0)
  return data

def normalize_data(data):
  mean = data['velocity(m/s)'].mean()
  std = data['velocity(m/s)'].std()
  data['velocity(m/s)'] = (data['velocity(m/s)'] - mean) / std
  return data

def process_csv(file):
  data = read_csv_file(file)
  data = clean_data(data)
  data = normalize_data(data)
  
  velocity = data['velocity(m/s)'].values
  derivative = np.diff(velocity, prepend=velocity[0])
  rolling_mean = pd.Series(velocity).rolling(window=10, min_periods=1).mean().values
  rolling_std = pd.Series(velocity).rolling(window=10, min_periods=1).std().fillna(0).values
  
  features = np.stack((velocity, derivative, rolling_mean, rolling_std), axis=1)
  
  num_sequences = len(features) - SEQUENCE_LENGTH + 1
  X = np.array([features[i:i+SEQUENCE_LENGTH] for i in range(num_sequences)])
  
  return X, data['time_rel(sec)'].values

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  X, times = process_csv(file)
  X = torch.tensor(X, dtype=torch.float32).to(device)
  
  predictions = []
  with torch.no_grad():
      for i in range(0, len(X), 128):
          X_batch = X[i:i+128]
          outputs = model(X_batch)
          probs = torch.sigmoid(outputs)
          predictions.extend(probs.cpu().numpy())
  
  predictions = np.array(predictions)
  
  # Detect events
  threshold = 0.5
  events = np.where(predictions > threshold)[0]
  event_times = times[events + SEQUENCE_LENGTH - 1]
  
  return {
      "predictions": predictions.tolist(),
      "times": times.tolist(),
      "event_times": event_times.tolist()
  }

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)