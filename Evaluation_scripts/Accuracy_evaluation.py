import os
import cv2 as cv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define paths
BASE_DIR = r"/home/mynah/Desktop/HINDI (copy)/Dataset 2_filtered"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, "images")
TEST_IMG_DIR = os.path.join(TEST_DIR, "images")


# Define image properties
IMG_HEIGHT = 32
IMG_WIDTH = 128
CHANNELS = 1
TIME_STEPS = IMG_WIDTH // 4  # 32

# Load labels
train_labels_df = pd.read_csv(os.path.join(TRAIN_DIR, "train_labels.csv"), sep=",", quotechar='"', encoding="utf-8")
test_labels_df = pd.read_csv(os.path.join(TEST_DIR, "test_labels.csv"), sep=",", quotechar='"', encoding="utf-8")

# Extract unique characters
all_labels = ''.join(train_labels_df["Label"].tolist() + test_labels_df["Label"].tolist())
unique_chars = sorted(set(all_labels))
unique_chars.append('[BLANK]')  # CTC blank token
char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
NUM_CLASSES = len(unique_chars)
print("Number of unique characters:", NUM_CLASSES)
print("Unique characters:", unique_chars)

# Max label length
MAX_LABEL_LEN = max(train_labels_df["Label"].str.len().max(), test_labels_df["Label"].str.len().max())
print("Max label length:", MAX_LABEL_LEN)

# Function to encode labels
BLANK_IDX = char_to_idx['[BLANK]']
def encode_label(label):
    return [char_to_idx[char] for char in label] + [BLANK_IDX] * (MAX_LABEL_LEN - len(label))

# Custom Dataset class
class OCRDataset(Dataset):
    def __init__(self, image_dir, labels_df):
        self.image_dir = image_dir
        self.labels_df = labels_df
        
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["Filename"])
        label = row["Label"]
        
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Image {img_path} not found.")
            return None
            
        h, w = img.shape
        if w < IMG_WIDTH:
            img = np.pad(img, ((0, 0), (0, IMG_WIDTH - w)), mode='constant', constant_values=255)
        elif w > IMG_WIDTH:
            img = cv.resize(img, (IMG_WIDTH, h))
        img = cv.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype("float32") / 255.0
        img = torch.FloatTensor(img).unsqueeze(0)  # Add channel dimension
        
        encoded_label = encode_label(label)
        label_tensor = torch.LongTensor(encoded_label)
        label_length = torch.tensor([len(label)], dtype=torch.long)
        input_length = torch.tensor([TIME_STEPS], dtype=torch.long)
        
        return {
            "images": img,
            "labels": label_tensor,
            "input_length": input_length,
            "label_length": label_length,
            "true_label": label  # Store the original label for comparison
        }


# Load test dataset
test_dataset = OCRDataset(TEST_IMG_DIR, test_labels_df)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # RNN layers
        self.rnn = nn.Sequential(
            nn.LSTM(128 * (IMG_HEIGHT // 4), 256, bidirectional=True, dropout=0.3, batch_first=True)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(512, NUM_CLASSES)  # 512 = 256 * 2 (bidirectional)
        
    def forward(self, x):
        # CNN
        x = self.cnn(x)
        # Reshape for RNN
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width) -> (batch, width, height, channels)
        x = x.reshape(batch_size, TIME_STEPS, -1)
        # RNN
        x, _ = self.rnn(x)
        # FC
        x = self.fc(x)
        return x

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(device)
            true_labels = batch["true_label"]
            
            outputs = model(images)
            outputs = outputs.log_softmax(2)
            outputs = outputs.permute(1, 0, 2)
            
            _, predicted = torch.max(outputs, dim=2)
            predicted = predicted.permute(1, 0)
            
            for i in range(len(true_labels)):
                true_label = true_labels[i]
                pred_sequence = []
                prev_char = -1
                for t in range(predicted.size(1)):
                    char_idx = predicted[i, t].item()
                    if char_idx != prev_char and char_idx != BLANK_IDX:
                        pred_sequence.append(idx_to_char[char_idx])
                    prev_char = char_idx
                
                predicted_text = ''.join(pred_sequence)
                print(f"True Label: {true_label}, Predicted: {predicted_text}")
                
                if predicted_text == true_label:
                    correct_predictions += 1
                total_samples += 1
    
    if total_samples > 0:
        accuracy = (correct_predictions / total_samples) * 100
        print(f"\nEvaluation Results:")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Total Samples: {total_samples}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No valid samples found in the test dataset.")

# Initialize model
model = CRNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load saved model weights
try:
    model.load_state_dict(torch.load("/home/mynah/Desktop/HINDI (copy)/finedtuned.pth", map_location=device))
    print("Loaded saved model weights.")
except FileNotFoundError:
    print("Model file 'finedtuned.pth' not found. Please check the path.")
except RuntimeError as e:
    print(f"Error loading state dict: {e}")

# Run evaluation
evaluate_model(model, test_loader, device)