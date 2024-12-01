import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# ----------------------------
# 1. Feature Extraction Functions
# ----------------------------

def get_feature_extractor(device):
    """
    Initializes the ResNet50 model for feature extraction.
    
    Args:
        device (torch.device): Computation device.
    
    Returns:
        nn.Module: Feature extractor model.
        torchvision.transforms.Compose: Image preprocessing transformations.
    """
    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=True)
    
    # Remove the final classification layer
    modules = list(model.children())[:-1]  # Remove the last fc layer
    model = nn.Sequential(*modules)
    
    model.to(device)
    model.eval()
    
    # Define image preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ResNet50 normalization
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return model, preprocess

def load_image_paths(dataset_dir):
    """
    Recursively loads all image file paths from the dataset directory.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
    
    Returns:
        List[str]: List of image file paths.
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith(supported_formats):
                image_paths.append(os.path.join(root, file))
    return image_paths

def extract_features(model, preprocess, image_paths, device, batch_size=32):
    """
    Extracts feature vectors from images using the provided model.
    
    Args:
        model (nn.Module): Feature extractor model.
        preprocess (callable): Image preprocessing transformations.
        image_paths (List[str]): List of image file paths.
        device (torch.device): Computation device.
        batch_size (int): Number of images per batch.
    
    Returns:
        np.ndarray: Array of feature vectors.
        List[str]: Corresponding image paths.
    """
    features = []
    valid_image_paths = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Features"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                image = preprocess(image)
                images.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        
        if not images:
            continue
        
        images = torch.stack(images).to(device)
        with torch.no_grad():
            batch_features = model(images).squeeze(-1).squeeze(-1)  # ResNet50 outputs (batch_size, 2048, 1, 1)
            batch_features = batch_features.cpu().numpy()
            features.append(batch_features)
            valid_image_paths.extend(batch_paths[:len(batch_features)])
    
    features = np.vstack(features)
    return features, valid_image_paths

# ----------------------------
# 2. LSH Implementation
# ----------------------------

class LSH:
    def __init__(self, hash_size=32, input_dim=2048, num_tables=5, device='cpu'):
        """
        Initializes the LSH with random hyperplanes.
        
        Args:
            hash_size (int): Number of hash bits per table.
            input_dim (int): Dimensionality of input feature vectors.
            num_tables (int): Number of hash tables.
            device (str): Computation device.
        """
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_tables = num_tables
        self.device = device
        
        # Initialize random hyperplanes for each table
        self.hyperplanes = []
        for _ in range(self.num_tables):
            # Random hyperplanes: (hash_size, input_dim)
            planes = torch.randn(self.hash_size, self.input_dim).to(self.device)
            self.hyperplanes.append(planes)
        
        # Initialize hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.num_tables)]
    
    def _hash(self, vectors, planes):
        """
        Hashes vectors using the provided hyperplanes.
        
        Args:
            vectors (np.ndarray): Feature vectors to hash. Shape: (N, D)
            planes (torch.Tensor): Hyperplanes. Shape: (hash_size, D)
        
        Returns:
            List[str]: List of binary hash codes.
        """
        vectors = torch.from_numpy(vectors).to(self.device)  # (N, D)
        projections = torch.matmul(vectors, planes.t())      # (N, hash_size)
        bits = (projections > 0).int()                      # (N, hash_size)
        hash_codes = bits.numpy().astype(str)                # Convert to strings for hashing
        hash_codes = [''.join(bits_row) for bits_row in hash_codes]
        return hash_codes
    
    def index(self, vectors, image_paths):
        """
        Indexes the vectors into hash tables.
        
        Args:
            vectors (np.ndarray): Feature vectors to index. Shape: (N, D)
            image_paths (List[str]): Corresponding image paths.
        """
        for table_idx in range(self.num_tables):
            planes = self.hyperplanes[table_idx]
            hash_codes = self._hash(vectors, planes)  # List of hash codes
            for hash_code, img_path in zip(hash_codes, image_paths):
                self.hash_tables[table_idx][hash_code].append(img_path)
    
    def query(self, vector, top_k=5):
        """
        Queries the LSH to find similar images.
        
        Args:
            vector (np.ndarray): Query feature vector. Shape: (D,)
            top_k (int): Number of similar images to retrieve.
        
        Returns:
            List[str]: List of similar image paths.
        """
        candidates = set()
        for table_idx in range(self.num_tables):
            planes = self.hyperplanes[table_idx]
            hash_code = self._hash(vector.reshape(1, -1), planes)[0]
            matched = self.hash_tables[table_idx].get(hash_code, [])
            candidates.update(matched)
        
        return list(candidates)[:top_k]

# ----------------------------
# 3. Similarity Functions
# ----------------------------

def compute_cosine_similarity(vec1, vec2):
    """
    Computes the cosine similarity between two vectors.
    
    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.
    
    Returns:
        float: Cosine similarity score.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)

def find_top_k_similar(query_vec, candidate_vecs, candidate_paths, top_k=5):
    """
    Finds the top-K similar images based on cosine similarity.
    
    Args:
        query_vec (np.ndarray): Query feature vector.
        candidate_vecs (np.ndarray): Candidate feature vectors.
        candidate_paths (List[str]): Corresponding image paths.
        top_k (int): Number of top similar images to retrieve.
    
    Returns:
        List[Tuple[str, float]]: List of tuples containing image paths and similarity scores.
    """
    similarities = []
    for vec, path in zip(candidate_vecs, candidate_paths):
        sim = compute_cosine_similarity(query_vec, vec)
        similarities.append((path, sim))
    
    # Sort based on similarity scores in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

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
    for i, (img_path, score) in enumerate(similar_images):
        plt.subplot(1, len(similar_images)+1, i+2)
        try:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.title(f"Score: {score:.4f}")
            plt.axis('off')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    plt.show()


def main():
    # Define paths and parameters
    dataset_dir = './tops/'  # Replace with your actual dataset path
    query_image_path = './tops//MEN-Denim-id_00000089-01_7_additional.jpg'  # Replace with your query image path
    
    embedding_dim = 2048  # ResNet50's output dimension before the final classification layer
    hash_size = 2048        # Number of hash bits per table
    num_tables = 7        # Number of hash tables
    top_k = 5             # Number of similar images to retrieve
    batch_size = 64       # Number of images per batch during feature extraction
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Step 1: Initialize Feature Extractor
    model, preprocess = get_feature_extractor(device)
    
    # Step 2: Load Image Paths
    image_paths = load_image_paths(dataset_dir)
    print(f"Total images found: {len(image_paths)}")
    
    # Step 3: Extract Features
    features, valid_image_paths = extract_features(model, preprocess, image_paths, device, batch_size=batch_size)
    print(f"Extracted features shape: {features.shape}")
    
    # Step 4: Initialize and Index LSH
    lsh = LSH(hash_size=hash_size, input_dim=embedding_dim, num_tables=num_tables, device=device)
    lsh.index(features, valid_image_paths)
    print("LSH indexing completed.")
    
    # Step 5: Query Similar Images
    if not os.path.exists(query_image_path):
        print(f"Query image '{query_image_path}' does not exist. Please provide a valid path.")
        return
    
    # Extract query image feature
    try:
        query_image = Image.open(query_image_path).convert("RGB")
        query_image_tensor = preprocess(query_image).unsqueeze(0).to(device)
        with torch.no_grad():
            query_feature = model(query_image_tensor).squeeze(-1).squeeze(-1).cpu().numpy()
    except Exception as e:
        print(f"Error processing query image: {e}")
        return
    
    # Query LSH for similar images
    similar_candidates = lsh.query(query_feature, top_k=top_k*2)  # Retrieve more candidates to account for possible duplicates
    
    # Filter out the query image if it's in the candidates
    similar_candidates = [img for img in similar_candidates if img != query_image_path]
    
    # Extract features of candidates
    candidate_indices = [valid_image_paths.index(img) for img in similar_candidates]
    candidate_features = features[candidate_indices]
    candidate_paths = similar_candidates
    
    # Find top-K similar images based on cosine similarity
    top_k_similar = find_top_k_similar(query_feature, candidate_features, candidate_paths, top_k=top_k)
    
    # Print similar images and their scores
    print("Top 5 similar images:")
    for img, score in top_k_similar:
        print(f"{img} - Cosine Similarity: {score:.4f}")
    
    # Visualize the results
    visualize_similar_images(query_image_path, top_k_similar)

if __name__ == "__main__":
    main()
