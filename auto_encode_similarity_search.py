import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import faiss

# ----------------------------
# 1. Define the Autoencoder
# ----------------------------

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # [B, 16, 112, 112]
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # [B, 32, 56, 56]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # [B, 64, 28, 28]
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),# [B, 128, 14, 14]
            nn.ReLU(True)
        )
        self.fc_enc = nn.Linear(128 * 14 * 14, latent_dim)
        
        # Decoder
        self.fc_dec = nn.Linear(latent_dim, 128 * 14 * 14)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # [B, 64, 28, 28]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # [B, 32, 56, 56]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # [B, 16, 112, 112]
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),   # [B, 3, 224, 224]
            nn.Sigmoid()  # To ensure the output is between 0 and 1
        )
        
    def forward(self, x):
        # Encode
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc_enc(x)
        
        # Decode
        x = self.fc_dec(latent)
        x = x.view(x.size(0), 128, 14, 14)
        x = self.decoder(x)
        return x, latent

# ----------------------------
# 2. Define Dataset and Dataloaders
# ----------------------------

from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

def get_dataloaders(dataset_dir, batch_size=128, valid_split=0.1):
    """
    Prepares training and validation dataloaders.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        batch_size (int): Number of samples per batch.
        valid_split (float): Fraction of training data to use for validation.
    
    Returns:
        DataLoader, DataLoader: Training and validation dataloaders.
    """
    # Image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
    ])
    
    # Load dataset
    dataset = ImageFolder(root=dataset_dir, transform=transform)
    
    # Split dataset into training and validation
    total_size = len(dataset)
    valid_size = int(valid_split * total_size)
    train_size = total_size - valid_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, valid_loader

# ----------------------------
# 3. Training Function
# ----------------------------

def train_autoencoder(model, train_loader, valid_loader, device, num_epochs=20, learning_rate=1e-3):
    """
    Trains the autoencoder model.
    
    Args:
        model (nn.Module): Autoencoder model.
        train_loader (DataLoader): Training dataloader.
        valid_loader (DataLoader): Validation dataloader.
        device (torch.device): Computation device.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
    
    Returns:
        nn.Module: Trained model.
        List[float], List[float]: Lists of training and validation losses.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    valid_losses = []
    
    best_valid_loss = float('inf')
    patience = 5
    trigger_times = 0
    
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            inputs, _ = data  # We don't need labels
            inputs = inputs.to(device)
            
            # Forward pass
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_train_loss:.4f}")
        
        # Validation Phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, _ = data
                inputs = inputs.to(device)
                
                # Forward pass
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
                
                running_loss += loss.item() * inputs.size(0)
        
        epoch_valid_loss = running_loss / len(valid_loader.dataset)
        valid_losses.append(epoch_valid_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {epoch_valid_loss:.4f}")
        
        # Early Stopping
        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_autoencoder.pth')
            print("Best model saved.")
        else:
            trigger_times += 1
            print(f"Early Stopping Trigger: {trigger_times}/{patience}")
            if trigger_times >= patience:
                print("Early stopping!")
                break
    
    # Plot training and validation losses
    plt.figure(figsize=(10,5))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()
    
    return model, train_losses, valid_losses

# ----------------------------
# 4. Extract Latent Vectors
# ----------------------------

def extract_latent_vectors(model, dataloader, device):
    """
    Extracts latent vectors from the encoder.
    
    Args:
        model (nn.Module): Trained autoencoder model.
        dataloader (DataLoader): Dataloader for the dataset.
        device (torch.device): Computation device.
    
    Returns:
        np.ndarray: Array of latent vectors.
        List[str]: Corresponding image paths.
    """
    model.eval()
    latent_vectors = []
    image_paths = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting Latent Vectors"):
            inputs, paths = data
            inputs = inputs.to(device)
            outputs, latent = model(inputs)
            latent = latent.cpu().numpy()
            latent_vectors.append(latent)
            image_paths.extend(paths)
    
    latent_vectors = np.vstack(latent_vectors)
    return latent_vectors, image_paths

# To include image paths, redefine the Dataset class

class ImageDatasetWithPaths(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.transform = image_folder.transform
        self.imgs = image_folder.imgs  # List of tuples (path, class_idx)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path  # Return image and its path

# ----------------------------
# 5. Implementing FAISS for Similarity Search
# ----------------------------

def build_faiss_index(latent_vectors, index_path='faiss_index.index'):
    """
    Builds and saves a FAISS index from latent vectors.
    
    Args:
        latent_vectors (np.ndarray): Array of latent vectors. Shape: (N, D)
        index_path (str): Path to save the FAISS index.
    
    Returns:
        faiss.Index: Built FAISS index.
    """
    d = latent_vectors.shape[1]  # Dimensionality of latent vectors
    index = faiss.IndexFlatL2(d)  # Using L2 distance for similarity
    index.add(latent_vectors)
    faiss.write_index(index, index_path)
    print(f"FAISS index built and saved to '{index_path}'.")
    return index

def search_similar_images(query_vector, index, image_paths, top_k=5):
    """
    Searches for similar images based on the query vector.
    
    Args:
        query_vector (np.ndarray): Latent vector of the query image. Shape: (d,)
        index (faiss.Index): FAISS index containing latent vectors.
        image_paths (List[str]): List of image paths corresponding to the index.
        top_k (int): Number of similar images to retrieve.
    
    Returns:
        List[Tuple[str, float]]: List of similar image paths with their L2 distances.
    """
    # Ensure the query_vector is 2D
    if len(query_vector.shape) == 1:
        query = np.expand_dims(query_vector, axis=0).astype('float32')  # Shape: (1, d)
    elif len(query_vector.shape) == 2 and query_vector.shape[0] == 1:
        query = query_vector.astype('float32')  # Shape: (1, d)
    else:
        raise ValueError(f"Unexpected query vector shape: {query_vector.shape}")
    
    distances, indices = index.search(query, top_k)
    
    similar_images = []
    for idx, dist in zip(indices[0], distances[0]):
        similar_images.append((image_paths[idx], dist))
    
    return similar_images


def extract_latent_vectors_with_paths(model, dataloader, device):
    """
    Extracts latent vectors and image paths from the encoder.
    
    Args:
        model (nn.Module): Trained autoencoder model.
        dataloader (DataLoader): Dataloader for the dataset.
        device (torch.device): Computation device.
    
    Returns:
        np.ndarray: Array of latent vectors.
        List[str]: Corresponding image paths.
    """
    model.eval()
    latent_vectors = []
    image_paths = []
    
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting Latent Vectors"):
            inputs, paths = data  # ImageFolder returns image and label, not paths
            inputs = inputs.to(device)
            outputs, latent = model(inputs)
            latent = latent.cpu().numpy()
            latent_vectors.append(latent)
            image_paths.extend(paths)
    
    latent_vectors = np.vstack(latent_vectors)
    return latent_vectors, image_paths


# ----------------------------
# 6. Visualization Function
# ----------------------------


def visualize_similar_images(query_image_path, similar_images):
    """
    Visualizes the query image alongside its similar images.
    
    Args:
        query_image_path (str): Path to the query image.
        similar_images (List[Tuple[str, float]]): List of similar image paths with similarity scores.
    """
    plt.figure(figsize=(15, 5))
    
    # Display query image
    plt.subplot(1, len(similar_images)+1, 1)
    try:
        query_img = Image.open(query_image_path).convert("RGB")
        plt.imshow(query_img)
        plt.title("Query Image")
        plt.axis('off')
    except Exception as e:
        print(f"Error loading query image: {e}")
    
    # Display similar images
    for i, (img_path, dist) in enumerate(similar_images):
        plt.subplot(1, len(similar_images)+1, i+2)
        try:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.title(f"Dist: {dist:.2f}")
            plt.axis('off')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    plt.show()

# ----------------------------
# 7. Complete Script Execution
# ----------------------------

def main():
    # Define paths and parameters
    dataset_dir = './labels_sep/tops/MEN/'  # Replace with your actual dataset path
    query_image_path = './labels_sep/tops/MEN/Denim/MEN-Denim-id_00000089-01_7_additional.jpg'  # Replace with your query image path
    
    batch_size = 128
    num_epochs = 10
    learning_rate = 1e-3
    latent_dim = 128
    top_k = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Prepare Data
    train_loader, valid_loader = get_dataloaders(dataset_dir, batch_size=batch_size, valid_split=0.1)
    
    # Step 2: Initialize Model
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    print(;;)
    # Step 3: Train Autoencoder
    # model, train_losses, valid_losses = train_autoencoder(
    #     model, train_loader, valid_loader, device,
    #     num_epochs=num_epochs, learning_rate=learning_rate
    # )
    
    # Step 4: Load Best Model
    model.load_state_dict(torch.load('best_autoencoder.pth'))
    model.eval()
    print("Best autoencoder model loaded.")
    
    # Step 5: Create a DataLoader with image paths
    # Define a custom dataset to return image paths
    train_dataset_full = ImageFolder(root=dataset_dir, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]))
    
    train_dataset_with_paths = ImageDatasetWithPaths(train_dataset_full)
    train_loader_with_paths = DataLoader(train_dataset_with_paths, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Step 6: Extract Latent Vectors
    latent_vectors, image_paths = extract_latent_vectors_with_paths(model, train_loader_with_paths, device)
    print(f"Extracted latent vectors shape: {latent_vectors.shape}")
    
    # Step 7: Build FAISS Index
    faiss_index = build_faiss_index(latent_vectors, 'faiss_index.index')
    
    # Step 8: Perform Similarity Search
    if not os.path.exists(query_image_path):
        print(f"Query image '{query_image_path}' does not exist. Please provide a valid path.")
        return
    
    # Extract latent vector for the query image
    try:
        query_image = Image.open(query_image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        query_image_tensor = preprocess(query_image).unsqueeze(0).to(device)
        with torch.no_grad():
            _, query_latent = model(query_image_tensor)
            query_latent = query_latent.cpu().numpy().squeeze(0)  # Shape: (d,)
    except Exception as e:
        print(f"Error processing query image: {e}")
        return
    
    # Search for similar images
    similar_images = search_similar_images(query_latent, faiss_index, image_paths, top_k=top_k)
    
    # Print similar images and their distances
    print("Top 5 similar images:")
    for img, dist in similar_images:
        print(f"{img} - Distance: {dist:.4f}")
    
    # Visualize the results
    visualize_similar_images(query_image_path, similar_images)

# ----------------------------
# 8. Custom Dataset Class with Image Paths
# ----------------------------

class ImageDatasetWithPaths(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.transform = image_folder.transform
        self.imgs = image_folder.imgs  # List of tuples (path, class_idx)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path  # Return image and its path

# ----------------------------
# 9. Execute the Script
# ----------------------------

if __name__ == "__main__":
    main()
