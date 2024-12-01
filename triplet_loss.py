import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
from tqdm import tqdm

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Triplet Loss Network architecture
class TripletLossNetwork(nn.Module):
    def __init__(self):
        super(TripletLossNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove final layer for embedding output

    def forward(self, x):
        return self.resnet(x)

# Triplet Loss function
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.pairwise_distance(anchor, positive, 2)
        distance_negative = torch.pairwise_distance(anchor, negative, 2)
        loss = torch.mean(torch.clamp(distance_positive - distance_negative + self.margin, min=0.0))
        return loss

# Dataset to load triplets of images
class TripletDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Create triplets of images with labels
        for label, class_folder in enumerate(os.listdir(image_folder)):
            class_path = os.path.join(image_folder, class_folder)
            images = [os.path.join(class_path, img) for img in os.listdir(class_path) if img.endswith('.jpg')]
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    # Create anchor-positive pairs (same class)
                    self.image_paths.append((images[i], images[j], images[i]))  # (anchor, positive, negative)
                    self.labels.append(1)  # Same class

            # Add dissimilar triplets from other classes
            for other_label, other_class_folder in enumerate(os.listdir(image_folder)):
                if label != other_label:
                    other_class_path = os.path.join(image_folder, other_class_folder)
                    other_images = [os.path.join(other_class_path, img) for img in os.listdir(other_class_path) if img.endswith('.jpg')]
                    for img in images:
                        for neg_img in other_images:
                            # Create anchor-negative triplets (different class)
                            self.image_paths.append((img, img, neg_img))  # (anchor, positive, negative)
                            self.labels.append(0)  # Different class

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img1_path, img2_path, img3_path = self.image_paths[idx]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        img3 = Image.open(img3_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3  # Returning triplets

# Transformations
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Dataset and DataLoader
dataset = TripletDataset(image_folder='labels_sep/ss/', transform=transform)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Model, optimizer, and loss function
model = TripletLossNetwork().to(device)  # Move model to GPU if available
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = TripletLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for img1, img2, img3 in tqdm(train_loader):
        img1, img2, img3 = img1.to(device), img2.to(device), img3.to(device)  # Move images to GPU

        optimizer.zero_grad()
        
        # Forward pass for triplets
        output1 = model(img1)
        output2 = model(img2)
        output3 = model(img3)
        
        # Compute loss using triplet loss
        loss = criterion(output1, output2, output3)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), 'triplet_loss_model.pth')

