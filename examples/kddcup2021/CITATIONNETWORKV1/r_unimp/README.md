# R-UNIMP for CITATIONNETWORKV1 using PGL

The code is adapted from the MAG240M R-UniMP implementation for the CITATIONNETWORKV1 dataset using the outputacm.txt dataset.

## Dataset Format

The CITATIONNETWORKV1 dataset (outputacm.txt) is in the format:
- Each paper starts with `#*title`
- Authors are indicated by `#@author1,author2`
- Year is indicated by `#tyear`
- Conference/venue is indicated by `#cvenue`
- Index is indicated by `#indexN`
- References are indicated by `#!` followed by cited paper indexes

## Installation Requirements

```
pip install -r requirements.txt
```

Or install manually:
```
paddle>=2.0.0
pgl>=2.1.2
numpy
tqdm
pyyaml
tensorboardX
easydict
scipy
scikit-learn
```

## Quick Start

### Option 1: Using the main script (Recommended)

```bash
python main.py
```

This script will:
1. Automatically check if preprocessing is needed
2. Parse the outputacm.txt dataset
3. Build graph structures
4. Update configuration with dataset information
5. Provide options to train and/or run inference

### Option 2: Manual execution

#### Step 1: Parse the dataset

```bash
cd dataset
python parse_citationnetwork.py
```

This will parse the outputacm.txt file and create:
* `processed/papers.pkl`: Paper information (title, authors, year, venue)
* `processed/citation_edges.npy`: Citation edges between papers
* `processed/author_paper_edges.pkl`: Author-paper relationships
* `processed/paper_features.npy`: Paper feature embeddings (768-dim)
* `processed/paper_labels.npy`: Paper classification labels based on venue
* `processed/metadata.pkl`: Dataset metadata

#### Step 2: Build graph structures

```bash
python build_graphs.py
cd ..
```

This will create the PGL graph structures:
* `processed/paper_citation_graph`: Paper-to-paper citation graph
* `processed/author_paper_graph_src`: Author-to-paper bipartite graph  
* `processed/author_paper_graph_dst`: Paper-to-author bipartite graph
* `processed/combined_features.npy`: Combined node features for all nodes
* `processed/train_idx.npy`, `processed/val_idx.npy`, `processed/test_idx.npy`: Data splits

#### Step 3: Train the model

```bash
python r_unimp_train.py --conf configs/r_unimp_citationnetwork.yaml
```

#### Step 4: Run inference

```bash
python r_unimp_infer.py --conf configs/r_unimp_citationnetwork.yaml
```

## Configuration

The configuration file `configs/r_unimp_citationnetwork.yaml` contains all model and training parameters. Key parameters:

- `model.num_class`: Number of venue classes (auto-updated from dataset)
- `model.input_size`: Feature dimension (768)
- `model.hidden_size`: Hidden layer size (512)
- `model.num_layers`: Number of GNN layers (2)
- `model.edge_type`: Number of edge types (3: citation, author->paper, paper->author)
- `batch_size`: Training batch size (512)
- `lr`: Learning rate (0.001)
- `epochs`: Number of training epochs (50)

## Model Architecture

The R-UniMP model consists of:

1. **Multi-relational GNN**: Handles different edge types (citations, author-paper relationships)
2. **Label propagation**: Uses labeled nodes to help classify unlabeled ones
3. **Attention mechanism**: Combines information from different edge types
4. **Metapath2vec integration**: Incorporates structural embeddings

## Output

- **Training**: Model checkpoints saved to `output/citationnetwork_runimp/`
- **Inference**: Predictions saved to `output/citationnetwork_runimp/predictions/`
- **Logs**: Training logs and TensorBoard logs in the output directory

## Performance

Training and evaluation metrics (accuracy, loss) will be reported during the training process and logged to TensorBoard for visualization.

## File Structure

```
CITATIONNETWORKV1/r_unimp/
├── README.md
├── requirements.txt
├── main.py                    # Main execution script
├── r_unimp_train.py          # Training script
├── r_unimp_infer.py          # Inference script
├── utils.py                  # Utility functions
├── optimization.py           # Optimizer configuration
├── configs/
│   └── r_unimp_citationnetwork.yaml
├── dataset/
│   ├── outputacm.txt         # Input dataset
│   ├── parse_citationnetwork.py
│   ├── build_graphs.py
│   ├── data_generator_citationnetwork.py
│   └── processed/            # Generated after preprocessing
└── models/
    ├── __init__.py
    ├── r_unimp.py
    └── gat_conv.py
```
