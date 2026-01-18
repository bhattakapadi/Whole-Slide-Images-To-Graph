# ========== Fix PNG decompression issues ==========
# Configure PIL to handle large PNG files commonly found in medical imaging
# This prevents errors when processing WSI files with embedded metadata
from PIL import Image, ImageFile, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024  # Increase text chunk size limit for large PNG annotations
ImageFile.LOAD_TRUNCATED_IMAGES = True              # Allow loading truncated images (common in WSIs)

# Import standard libraries
import sys
import os
import shutil
import h5py  # For reading coordinate files in HDF5 format
import openslide  # For reading Whole Slide Images (WSI)
import torch  
from torchvision import transforms  
import networkx as nx  # Graph construction and manipulation
import numpy as np  
import logging  # Logging for monitoring
import matplotlib.pyplot as plt  # Visualization
from concurrent.futures import ThreadPoolExecutor  
from torch import amp  # Automatic Mixed Precision for faster inference

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from torch_geometric.data import Data, Batch  # Graph neural network data structures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformation pipeline for patches
# Standard ImageNet normalization used by pre-trained ResNet models
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet standard deviation
])


# Add CLAM repository to path (This is where, the pretrained resenet model is set)--> this is changeable 
#   from models.resnet_custom_dep import resnet50_baseline ---> please change the location of model, if you want with the address
sys.path.append('/user/m.bhattakapadi/u17196/CLAM')

def loadModel():
    """
    Loads a pre-trained ResNet-50 model for feature extraction from tissue patches.
    
    The model is specifically modified for WSI analysis (resnet_custom_dep).
    This is typically a ResNet-50 backbone pre-trained on ImageNet, adapted for
    medical imaging tasks. The model extracts 1024-dimensional feature vectors
    from each 224x224 patch, capturing hierarchical visual patterns useful for
    downstream graph-based analysis.
    
    The model is set to evaluation mode (model.eval()) and moved to the
    appropriate device (GPU/CPU) for inference.
    
    Returns:
        torch.nn.Module: Pre-trained ResNet-50 model ready for feature extraction
    """
    from models.resnet_custom_dep import resnet50_baseline 
    model = resnet50_baseline(pretrained=True)  # Load with ImageNet weights
    model.eval() 
    model.to(device)  
    return model


def fast_transform(patch):
    """
    Optimized patch transformation using numpy and PyTorch operations.
    
    This function is faster than the standard torchvision.transforms pipeline
    because it avoids PIL operations and uses PyTorch's efficient tensor operations.
    It performs the same transformations as the 'transform' pipeline above but
    with better performance for batch processing.
    
    Steps:
    1. Convert PIL image to numpy array and normalize to [0, 1]
    2. Convert to tensor and change from HWC (Height-Width-Channels) to CHW format
    3. Resize to 224x224 using bilinear interpolation
    4. Normalize using ImageNet statistics
    
    Args:
        patch: PIL Image object of a tissue patch
        
    Returns:
        torch.Tensor: Normalized tensor of shape (3, 224, 224) ready for model input
    """
    patch = np.array(patch).astype(np.float32) / 255.0 
    patch = torch.tensor(patch).permute(2, 0, 1)  # HWC -> CHW (PyTorch format)
    patch = torch.nn.functional.interpolate(
        patch.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0) 
    patch = (patch - torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return patch


def process_patches_and_extract_graph(wsi_path: str, h5_path: str, model, patch_size: int = 128, batch_size: int = 256) -> tuple:
    """
    Extract features from WSI patches and construct an initial graph structure.
    
    This is the core function that performs parallel patch extraction and feature
    extraction using the ResNet model. It processes patches in batches with
    multithreading for I/O operations and uses mixed precision for faster GPU inference.
    
    The function:
    1. Reads patch coordinates from HDF5 file
    2. Opens the WSI using OpenSlide
    3. Creates an empty graph where each node represents a patch
    4. Processes patches in parallel using ThreadPoolExecutor
    5. Extracts features using the ResNet model with mixed precision on GPU
    6. Adds nodes to graph with features and coordinates as attributes
    
    Mixed Precision Strategy:
    - On GPU: Uses bfloat16 if supported (A100, newer GPUs), otherwise float16
    - On CPU: Uses float32 for stability
    
    Args:
        wsi_path (str): Path to the whole slide image file (.svs, .tiff, etc.)
        h5_path (str): Path to HDF5 file containing patch coordinates
        model: Pre-trained ResNet model for feature extraction
        patch_size (int): Size of patches to extract from WSI (default: 128)
        batch_size (int): Number of patches processed in each batch (default: 256)
    
    Returns:
        tuple: (G, coords) where:
            G: NetworkX graph with nodes containing patch features and coordinates
            coords: Array of all patch coordinates from the HDF5 file
    
    Raises:
        Exception: If any error occurs during processing
    """
    try:
        # Read patch coordinates from HDF5 file
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]  # Array of (x, y) coordinates
        logger.info(f"length of the coordinates: {len(coords)}")

        # Open WSI file - OpenSlide supports various WSI formats
        slide = openslide.OpenSlide(wsi_path)
        G = nx.Graph()  # Create empty undirected graph

        # Helper function for parallel patch reading and transformation
        def read_and_transform(coord):
            x, y = coord
            try:
                # Read patch from WSI at specified coordinate and level 0 (highest resolution)
                patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                return fast_transform(patch)  # Apply optimized transformation
            except Exception as e:
                logger.warning(f"Skipping ({x}, {y}): {e}")
                return None

        # Process patches in parallel using thread pool
        with ThreadPoolExecutor(max_workers=8) as executor:
            # Process coordinates in batches for efficient GPU utilization
            for i in range(0, len(coords), batch_size):
                batch_coords = coords[i:i+batch_size]
                
                # Submit patch reading tasks to thread pool
                futures = [executor.submit(read_and_transform, tuple(c)) for c in batch_coords]
                results = [f.result() for f in futures]

                # Filter out failed patches (None results)
                valid = [(j+i, tuple(batch_coords[j]), img) for j, img in enumerate(results) if img is not None]
                if not valid:
                    continue

                # Unpack valid patches
                indices, coords_valid, tensors = zip(*valid)
                batch_tensor = torch.stack(tensors).to(device)  # Stack into batch tensor

                # Extract features using ResNet model
                with torch.no_grad():  # Disable gradient calculation for inference
                    if device.type == 'cuda':
                        # Use mixed precision on GPU for faster inference
                        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        with amp.autocast(device_type='cuda', dtype=dtype):
                            batch_features = model(batch_tensor).cpu().numpy()
                    else:
                        # CPU fallback - use float32 for stability
                        batch_features = model(batch_tensor.float()).cpu().numpy()

                # Add nodes to graph with features and coordinates
                for idx, coord, feat in zip(indices, coords_valid, batch_features):
                    G.add_node(idx, feature=feat, coord=coord)

        slide.close()  # Close WSI file
        logger.info(f"Fast extraction done: {G.number_of_nodes()} nodes created")
        return G, coords

    except Exception as e:
        logger.error(f"Error in process_patches_and_extract_graph: {e}")
        raise


def drawGraph(G: nx.Graph):
    """
    Visualize the graph with nodes positioned according to their spatial coordinates.
    
    Creates a 2D plot where each node is placed at its (x, y) coordinate from the WSI.
    Isolated nodes (nodes without edges) are shown in blue, connected nodes in red.
    
    This visualization helps verify that the graph construction is working correctly
    and shows the spatial distribution of tissue patches.
    
    Args:
        G (nx.Graph): NetworkX graph with 'coord' attributes for each node
    
    Raises:
        Exception: If visualization fails
    """
    try:
        # Extract coordinates from node attributes
        positions = {node: data['coord'] for node, data in G.nodes(data=True)}
        isolated_nodes = set(nx.isolates(G))  # Find nodes without edges
        node_colors = ["blue" if node in isolated_nodes else "red" for node in G.nodes()]
        
        # Create visualization
        plt.figure(figsize=(10, 10))
        nx.draw(
            G,
            pos=positions,
            node_size=5,
            node_color=node_colors,
            edge_color="gray",
            with_labels=False,
            width=0.5
        )
        plt.title("Patch Graph (Blue = Isolated Nodes)")
        plt.axis("equal")  # Keep aspect ratio consistent
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in drawGraph: {e}")
        raise


def drawGraph2(G: nx.Graph, mirror_fix: bool = True):
    """
    Enhanced graph visualization with optional Y-axis mirroring.
    
    In image coordinates, the origin is typically at the top-left, while in
    mathematical plots, the origin is at the bottom-left. This function can
    flip the Y-axis to match the orientation of the original WSI.
    
    Args:
        G (nx.Graph): NetworkX graph with 'coord' attributes
        mirror_fix (bool): If True, flip Y-axis to match image coordinates
    
    Raises:
        Exception: If visualization fails
    """
    try:
        # Extract all coordinates for calculating bounds
        coords_array = np.array([data['coord'] for _, data in G.nodes(data=True)])
        positions = {node: tuple(data['coord']) for node, data in G.nodes(data=True)}
        isolated_nodes = set(nx.isolates(G))
        node_colors = ["blue" if node in isolated_nodes else "red" for node in G.nodes()]
        
        plt.figure(figsize=(10, 10))
        
        if mirror_fix:
            # Flip Y axis: image coordinates have Y increasing downward,
            # plot coordinates have Y increasing upward
            max_y = coords_array[:,1].max()
            positions = {node: (x, max_y - y) for node, (x, y) in positions.items()}
        
        nx.draw(
            G,
            pos=positions,
            node_size=5,
            node_color=node_colors,
            edge_color="gray",
            with_labels=False,
            width=0.5
        )
        plt.title("Patch Graph (Blue = Isolated Nodes)")
        plt.axis("equal")
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in drawGraph: {e}")
        raise


def getGraphEdges(G: nx.Graph, coords: np.ndarray, patch_size: int) -> nx.Graph:
    """
    Create edges between neighboring patches based on spatial proximity.
    
    Uses an 8-connected neighborhood (Moore neighborhood) to connect patches
    that are adjacent in the WSI. This creates a graph structure that preserves
    the spatial relationships between tissue patches, which is crucial for
    graph neural networks that analyze tissue architecture.
    
    Edge creation considers:
    1. Only patches that exist in the coordinate set
    2. Only if both nodes exist in the graph (some patches may have been skipped)
    3. Avoids duplicate edges
    
    Args:
        G (nx.Graph): Graph with nodes (from process_patches_and_extract_graph)
        coords (np.ndarray): Array of all patch coordinates (N, 2)
        patch_size (int): Size of patches used (determines neighbor distance)
    
    Returns:
        nx.Graph: Graph with edges added between neighboring patches
    
    Raises:
        Exception: If edge creation fails
    """
    try:
        logger.info("Graph edge extraction started")
        
        if not G.nodes():
            logger.warning("No nodes available for edge creation")
            return G
            
        # 8-connected neighborhood offsets (Moore neighborhood)
        # Includes diagonal neighbors for more complete connectivity
        offsets = [(-patch_size, -patch_size), (-patch_size, 0), (-patch_size, patch_size),
                  (0, -patch_size), (0, patch_size),
                  (patch_size, -patch_size), (patch_size, 0), (patch_size, patch_size)]
        
        # Create lookup structures for efficient neighbor checking
        coord_set = set((x, y) for x, y in coords)
        coord_to_index = {tuple(coord): idx for idx, coord in enumerate(coords)}
        
        edge_count = 0
        # Iterate through all coordinates to create edges
        for idx, (x, y) in enumerate(coords):
            if idx not in G.nodes():  # Skip if patch wasn't successfully processed
                continue
                
            # Check all 8 possible neighbor positions
            for dx, dy in offsets:
                neighbor = (x + dx, y + dy)
                if neighbor in coord_set:  # Check if neighbor exists in coordinate set
                    n2 = coord_to_index[neighbor]
                    if n2 in G.nodes() and not G.has_edge(idx, n2):  # Avoid duplicates
                        G.add_edge(idx, n2)
                        edge_count += 1
        
        logger.info(f"Graph edge extraction completed with {edge_count} edges")
        return G
        
    except Exception as e:
        logger.error(f"Error in getGraphEdges: {e}")
        raise


def getGraph(wsi_path: str, h5_path: str, patch_size: int = 256, visualize: bool = False):
    """
    Main pipeline function to construct a graph from a Whole Slide Image.
    
    This function orchestrates the entire graph construction process:
    1. Loads the pre-trained ResNet model
    2. Extracts features from patches and creates graph nodes
    3. Adds edges between neighboring patches
    4. Optionally visualizes the resulting graph
    
    The resulting graph represents the WSI as a spatial graph where:
    - Nodes: Tissue patches with ResNet features
    - Edges: Spatial adjacency between patches
    
    This graph can be used for various downstream tasks:
    - Graph neural networks for WSI classification
    - Tissue structure analysis
    - Cancer detection and grading
    
    Args:
        wsi_path (str): Path to whole slide image file
        h5_path (str): Path to HDF5 file containing patch coordinates
        patch_size (int): Size of patches to extract (default: 256)
        visualize (bool): Whether to display the graph visualization (default: False)
    
    Returns:
        nx.Graph: Complete graph representation of the WSI
    
    Raises:
        Exception: If graph construction fails at any step
    """
    try:
        model = loadModel()  # Load feature extraction model
        logger.info(f"WSI to graph construction started for {wsi_path}")

        # Step 1: Extract features and create graph nodes
        G, coords = process_patches_and_extract_graph(wsi_path, h5_path, model, patch_size=patch_size)
        
        # Step 2: Add edges between neighboring patches
        G = getGraphEdges(G, coords, patch_size)

        # Optional visualization
        if visualize:
            drawGraph(G)

        logger.info("WSI to graph construction completed successfully")
        return G

    except Exception as e:
        logger.error(f"Error in getGraph: {e}")
        raise


def nx_to_pyg_data(G: nx.Graph) -> 'torch_geometric.data.Data':
    """
    Convert a NetworkX graph to PyTorch Geometric Data format.
    
    PyTorch Geometric (PyG) requires graphs in a specific tensor format:
    - x: Node feature matrix of shape [num_nodes, num_features]
    - edge_index: Edge connectivity in COO format of shape [2, num_edges]
    
    This conversion enables using the graph with PyG's graph neural network
    layers and models.
    
    Args:
        G (nx.Graph): NetworkX graph with node features
    
    Returns:
        torch_geometric.data.Data: PyG Data object
    
    Raises:
        Exception: If conversion fails
    """
    try:
        # Create mapping from NetworkX node IDs to consecutive indices
        node_map = {nid: i for i, nid in enumerate(G.nodes())}
        
        # Extract node features (ResNet feature vectors)
        features = [G.nodes[nid]['feature'] for nid in G.nodes()]
        x = torch.tensor(np.array(features), dtype=torch.float)  # Feature matrix
        
        # Convert edges to PyG format (list of [src, dst] pairs)
        edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Shape: [2, E]
        
        return Data(x=x, edge_index=edge_index)
    except Exception as e:
        logger.error(f"Error in nx_to_pyg_data: {e}")
        raise


def batch_nx_graphs(graph_list: list) -> 'torch_geometric.data.Batch':
    """
    Batch multiple NetworkX graphs into a single PyTorch Geometric Batch.
    
    PyTorch Geometric supports batching multiple graphs into a single Batch object,
    which is essential for efficient mini-batch training of graph neural networks.
    
    The Batch object automatically handles:
    - Concatenation of node features
    - Re-indexing of edge indices
    - Tracking graph boundaries within the batch
    
    Args:
        graph_list (list): List of NetworkX graphs to batch
    
    Returns:
        torch_geometric.data.Batch: Batched graph data
    
    Raises:
        ValueError: If no valid graphs are provided
        Exception: If batching fails
    """
    try:
        # Convert each NetworkX graph to PyG Data format
        data_list = [nx_to_pyg_data(G) for G in graph_list if G is not None and len(G.nodes()) > 0]
        if not data_list:
            raise ValueError("No valid graphs to batch")
        
        # Use PyG's batching utility to combine graphs
        return Batch.from_data_list(data_list)
    except Exception as e:
        logger.error(f"Error in batch_nx_graphs: {e}")
        raise