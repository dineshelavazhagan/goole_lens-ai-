import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss
import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------
# 1. Define Functions
# ----------------------------

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

def generate_embeddings(model, preprocess, image_paths, device, batch_size=64):
    """
    Generates embeddings for all images using CLIP.
    
    Args:
        model (nn.Module): Pre-trained CLIP model.
        preprocess (callable): Image preprocessing function.
        image_paths (List[str]): List of image file paths.
        device (torch.device): Computation device.
        batch_size (int): Number of images per batch.
    
    Returns:
        np.ndarray: Array of embeddings.
        List[str]: Corresponding image paths.
    """
    embeddings = []
    valid_image_paths = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating Embeddings"):
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
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize embeddings
            embeddings.append(image_features.cpu().numpy())
            valid_image_paths.extend(batch_paths[:len(images)])  # Only add successfully processed images
    
    embeddings = np.vstack(embeddings).astype('float32')
    return embeddings, valid_image_paths

def build_faiss_index(embeddings, faiss_index_path, embedding_dim=512):
    """
    Builds a FAISS index for the given embeddings.
    
    Args:
        embeddings (np.ndarray): Normalized embeddings.
        faiss_index_path (str): Path to save the FAISS index.
        embedding_dim (int): Dimension of the embeddings.
    
    Returns:
        faiss.Index: Built FAISS index.
    """
    index = faiss.IndexFlatIP(embedding_dim)  # Using Inner Product for cosine similarity
    index.add(embeddings)
    faiss.write_index(index, faiss_index_path)
    print(f"FAISS index built and saved to '{faiss_index_path}'.")
    return index

def search_similar_images(query_image_path, model, preprocess, device, index, image_paths, top_k=5):
    """
    Searches for similar images to the query image using CLIP and FAISS.
    
    Args:
        query_image_path (str): Path to the query image.
        model (nn.Module): Pre-trained CLIP model.
        preprocess (callable): Image preprocessing function.
        device (torch.device): Computation device.
        index (faiss.Index): FAISS index containing embeddings.
        image_paths (List[str]): List of image paths corresponding to the index.
        top_k (int): Number of similar images to retrieve.
    
    Returns:
        List[Tuple[str, float]]: List of tuples containing image paths and similarity scores.
    """
    try:
        image = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading query image {query_image_path}: {e}")
        return []
    
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(image)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)
        query_embedding = query_embedding.cpu().numpy().astype('float32')
    
    faiss.normalize_L2(query_embedding)
    
    distances, indices = index.search(query_embedding, top_k)
    similar_images = []
    for idx, distance in zip(indices[0], distances[0]):
        similar_images.append((image_paths[idx], distance))
    
    return similar_images

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
            plt.title(f"Score: {score:.2f}")
            plt.axis('off')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    plt.show()

# ----------------------------
# 2. Main Execution Flow
# ----------------------------

def main():
    # Paths and Parameters
    dataset_dir = './tops/'       # Replace with your actual dataset path
    model_save_path = 'clip_model.pt'            # Path to save the CLIP model (optional)
    faiss_index_path = 'clip_faiss.index'        # Path to save the FAISS index
    image_paths_save = 'image_paths.npy'         # Path to save image paths

    embedding_dim = 512                           # CLIP's embedding dimension for ViT-B/32
    batch_size = 64                               # Adjust based on your GPU memory
    top_k = 5                                     # Number of similar images to retrieve

    # Step 1: Load Image Paths
    image_paths = load_image_paths(dataset_dir)
    print(f"Total images found: {len(image_paths)}")

    # Step 2: Load CLIP Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Step 3: Generate Embeddings
    embeddings, valid_image_paths = generate_embeddings(model, preprocess, image_paths, device, batch_size=batch_size)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Save valid image paths
    np.save(image_paths_save, valid_image_paths)
    print("Valid image paths saved successfully.")

    # Step 4: Build FAISS Index
    index = build_faiss_index(embeddings, faiss_index_path, embedding_dim=embedding_dim)

    # Step 5: Perform Similarity Search
    # Example Query Image Path
    query_image = './tops/MEN-Denim-id_00000080-01_7_additional.jpg'  # Replace with your query image path

    if not os.path.exists(query_image):
        print(f"Query image '{query_image}' does not exist. Please provide a valid path.")
    else:
        # Load image paths corresponding to the FAISS index
        image_paths = np.load(image_paths_save).tolist()
        
        # Perform similarity search
        similar_images = search_similar_images(
            query_image_path=query_image,
            model=model,
            preprocess=preprocess,
            device=device,
            index=index,
            image_paths=image_paths,
            top_k=top_k
        )
        
        # Print similar images and their scores
        print("Top 5 similar images:")
        for img, score in similar_images:
            print(f"{img} - Similarity Score: {score:.4f}")
        
        # Visualize the results
        visualize_similar_images(query_image, similar_images)

if __name__ == "__main__":
    main()
