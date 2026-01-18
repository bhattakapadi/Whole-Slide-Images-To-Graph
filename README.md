# Graph Construction from Whole Slide Images

## Overview
This project provides a framework for constructing graph representations directly from Whole Slide Images (WSIs).  
The graph construction pipeline is implemented in **`wsi2graph2_v2.py`**, which outlines how WSIs are converted into graph structures for downstream graph-based analysis.

The framework operates by dividing a high-resolution slide into image patches, extracting feature embeddings for each patch, and building a spatial graph that preserves local neighborhood relationships.

## Patch Extraction
The whole slide image is first divided into small, fixed-size patches.  
In this project, we use the **CLAM framework**, which provides the **coordinates of the top-left pixel** for each extracted patch.

These coordinates, along with the original whole slide images, are essential inputs for graph construction.

## Node Embedding
Each image patch is treated as a node in the graph.

- A **ResNet-18** model pretrained on **ImageNet** is used for feature extraction.
- The model outputs a **512-dimensional embedding** for each patch.
- ResNet-18 is open-source and widely used for visual feature learning.

## Graph Construction
Edges between nodes are created based on spatial proximity:

- **k-nearest neighbors (k = 8)** are used to connect each node to its nearest spatial neighbors.
- This approach helps preserve the **spatial structure** of the tissue in the graph representation.

## Performance Considerations
Graph construction can be computationally expensive, especially for large whole slide images with many patches.

To improve efficiency:
- Graphs can be **precomputed and saved**.
- During training or inference, the precomputed graphs are loaded directly, significantly speeding up the overall pipeline.

## Model Flexibility and Classification Tasks
The constructed graphs can be used with different downstream models.

In this work, the model is used for **multi-organ classification**. However, the framework is not restricted to this setting and can be applied to **organ-specific classification** by modifying the **final classification layer** to match the required number of output classes.

## Summary
- `wsi2graph2_v2.py` provides a framework for WSI-to-graph construction
- WSIs are divided into patches using CLAM
- Patch coordinates and WSIs are used to construct graphs
- ResNet-18 provides 512-dimensional node embeddings
- k-NN (k = 8) preserves spatial relationships
- Graphs can be precomputed to improve performance
- The framework supports flexible downstream classification tasks
