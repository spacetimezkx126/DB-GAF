
# DB-GAF: A Novel Graph-Based Dual-Branch Framework for Legal Case Retrieval with Attribute Filtering
This repository provides the implementation of **DB-GAF**, a novel graph-based dual-branch framework for legal case retrieval with attribute filtering. The model leverages a graph-based approach to retrieve legal cases by considering both the graph structure of the data and the attribute filtering to improve retrieval accuracy.

## Paper
**DB-GAF: A Novel Graph-Based Dual-Branch Framework for Legal Case Retrieval with Attribute Filtering**  
Link to the paper: [DB-GAF Paper](#) (Add the link to your paper here)


## Framework
The framework composed of 4 modules for the whole paper.
![Framework](images/framework.jpg)
## Model Architecture
The graph construction method for DB-GAF and attribute filtering.
![DB-GAF Architecture](images/db-gaf.jpg)

## DataSets
- lecard
- lecardv2
- processed_text ()

## Usage

For module III:

python dual_branch_contrast_graph.py --exp_name db_contrast_graph --dataset lecard

For mudule IV:

python basic_graph_models.py --exp_name basic_graph_eagatv2_efwf --model EAGATv2-EFWF --dataset lecardv2 --split_file 2_l --choose in 

