import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset_loader import PrescriptionDataset  # Ensure your dataset_loader.py works correctly
import os

# ✅ Hyperparameters
BATCH_SIZE = 16
IMG_HEIGHT = 32
IMG_WIDTH = 128
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Data Preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ✅ Load Dataset
dataset_path = "C:/Users/visha/prescription_ocr/synthetic_prescription_dataset"
train_dataset = PrescriptionDataset(dataset_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, train_dataset.char_map))

# ✅ OCR Model: CNN + BiLSTM
class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.lstm = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # CNN feature extraction
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x = x.view(x.size(0), -1, 128)  # (B, Sequence_length, Features)
        x, _ = self.lstm(x)  # Pass through LSTM
        x = self.fc(x)  # Fully connected layer
        return x

# ✅ Collate Function for DataLoader
def collate_fn(batch, char_map):
    images, labels = zip(*batch)
    images = torch.stack(images)  # Convert list of images to tensor

    # Convert text labels to numerical sequences
    label_sequences = [[char_map[c] for c in label if c in char_map] for label in labels]
    
    # Convert list of lists into tensor
    label_lengths = torch.tensor([len(seq) for seq in label_sequences], dtype=torch.long)
    label_tensor = torch.cat([torch.tensor(seq, dtype=torch.long) for seq in label_sequences])

    return images, label_tensor, label_lengths

# ✅ Initialize Model
num_classes = len(train_dataset.char_map) + 1  # +1 for CTC blank token
model = OCRModel(num_classes).to(DEVICE)

# ✅ Loss & Optimizer
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels, label_lengths in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)  # (B, Seq_len, num_classes)
            outputs = outputs.log_softmax(2)  # CTC Loss expects log probabilities

            # ✅ Fix input_lengths to match batch size
            batch_size = outputs.size(0)
            sequence_length = outputs.size(1)
            input_lengths = torch.full((batch_size,), sequence_length, dtype=torch.long).to(DEVICE)

            # Compute loss (CTC)
            loss = criterion(outputs.permute(1, 0, 2), labels, input_lengths, label_lengths)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss:.4f}")

    # ✅ Save the model
    torch.save(model.state_dict(), "prescription_ocr.pth")
    print("✅ Model saved as prescription_ocr.pth")

# ✅ Start Training
if __name__ == "__main__":
    train()

