# Knowledge Graph Generation

## Overview

This project aims to generate a knowledge graph from PDF files by extracting text and identifying relations within the text. The generated knowledge graph can be visualized through a Streamlit-based frontend application. This project is an enhancement of work carried out in 2023-2024, available at the following link: [KG-generation Repository](https://github.com/daJster/KG-generation/).

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Software Requirements](#software-requirements)
4. [Get the models](#get-the-models)
5. [Pipeline Explanation](#pipeline-explanation)
6. [Running the Application](#running-the-application)
7. [Folder Structure](#folder-structure)

## Introduction

The knowledge graph generation process involves extracting text from PDF files, identifying relationships within the text, and visualizing the relationships in a graph. This README provides a comprehensive guide on setting up the project and running the application.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/daJster/KG-generation.git
cd KG-generation
```

## Software Requirements
Install the required packages using pip :
```bash
pip3 install -r requirements.txt
```

Ensure the following Python packages are installed:

- nltk
- llm
- pygraft
- rdflib
- torch
- torchvision
- torchaudio
- pyvis
- PyMuPDF
- langdetect
- transformers
- -U sentence-transformers
- streamlit
- flask
- flask_cors
- neo4j
- GoogleNews
- googlesearch-python
- codecarbon
- wikipedia
- PyPDF2
- unidecode
- sentence_transformers
- spacy
- deep_translator
- hdbscan
- groq
- peft
- accelerate
- huggingface_hub 
- datasets
- pykeen
- streamlit
- networkx

## Get the Models

To obtain the fine-tuned models required for running the pipeline, which will be stored in the `./models` directory, follow these steps:

1. **Run the notebooks:**
   - Execute the cells in `./src/finetuning/all_mini_finetuning.ipynb`: This performs fine-tuning on two embedding models, **all-Mini** and **bge-small**, for a classification task.
   - Execute the cells in `./src/finetuning/finetuning_usable_text.ipynb`: This fine-tunes **bge-small** for a different classification task.

2. **Run the following commands:**
   ```bash
   cd src/finetuning
   ```
Then:
```bash
python3 mrebel_finetuning.py
```

This script fine-tunes MRebel, a model used to extract triplets from text, specializing it in the economic domain.

Finally, the `./src/finetuning/data` directory contains the training data required for fine-tuning. Most of this data was generated using the scripts `generate_article_triples.py` and `generate_text_chunks.py`, located in the `./src/finetuning` directory.

### Memgraph Installation

1) Install Docker if not already installed. Refer to the [official documentation](https://docs.docker.com/get-docker/) for installation instructions.

2) Start Docker by running the following command:

```bash
sudo service docker start
```

3) Install Memgraph using the following commands if not already installed:

```bash
sudo docker run -p 7687:7687 -p 7444:7444 -p 3000:3000 --name memgraph memgraph/memgraph-platform
```

4) Start Memgraph using the following command:

```bash
sudo docker start memgraph
```
5) Check if Memgraph is running by visiting http://localhost:3000 in your web browser.

## Pipeline Explanation

The pipeline consists of the following steps:

1) **File Upload**: Users upload PDF files through the Streamlit interface.
2) **Text Processing**: Extract text using PyMuPDF, detect its language, and translate it to English if necessary.
3) **Semantic Segmentation**: Divide the text into meaningful segments to enhance knowledge extraction.
4) **Relation Extraction**: Identify and extract structured knowledge in the form of triplets (subject, relation, object).
5) **Entity Merging and Lemmatization**: Normalize and merge duplicate entities to ensure consistency. A lemmatization stage ensures that entities and relationships have a basic form.
6) **Graph Construction**: Store the extracted knowledge in a structured graph representation, with an other merge stage to avoid having redundancies in the graph database (Memgraph).
7) **Admin interface**: Automize pipeline use with streamlit and choose any pdf file to run in our model.
8) **User Interface (UI)**: Create a Streamlit frontend application (app.py) for users to interact with the generated knowledge graph.

This pipeline automates the transformation of unstructured text into a structured knowledge graph, enabling efficient knowledge extraction and visualization.

## Running the Application

Both the admin and user interfaces need memgraph to be running. Ensure that memgraph is running before proceeding (refer to the [Memgraph Installation](#memgraph-installation) section for instructions).

### Admin interface
Run the main.py file for the first time to build streamlit web-app and to execute the knowledge graph generation process :

```bash
python3 src/pipeline/main.py
```

Then run the streamlit web-app :

```bash
streamlit run src/pipeline/main.py
```

If you encounter any issues, run the following command in the terminal and run the Streamlit app again.

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

Visit http://localhost:8501 in your web browser to access the admin web-app.


### User interface (graph visualization)
To run the UI interface :
1) Go to the web-app folder :
```bash
cd src/web-app
```
2) Run the app.py file :
```bash
python3 app.py
```

> :warning: **DO NOT run python3 src/web-app directly**: otherwise you won't be able to use radius search in the graph

## Folder Structure
```
├── KGG_2024.pdf
├── KGG_Presentation.pdf
├── README.md
├── articles_KGG
├── docker-compose.yml
├── dockerfile
├── documentation.html
├── generate_doc.py
├── requirements.txt
└── src
    ├── evaluation
    │   ├── pykeen_metrics.py
    │   └── tf_idf.py
    ├── finetuning
    │   ├── all_mini_finetuning.ipynb
    │   ├── data
    │   ├── finetuning_usable_text.ipynb
    │   ├── generate_article_triples.py
    │   ├── generate_text_chunks.py
    │   └── mrebel_finetuning.py
    ├── pipeline
    │   ├── KB_generation.py
    │   ├── clustering_merge.py
    │   ├── llama.py
    │   ├── main.py
    │   ├── params.py
    │   ├── pre_merge.py
    │   ├── semantic_segmentation.py
    │   ├── text_selection.py
    │   └── translation.py
    └── web-app
        ├── app.py
        ├── assets
        ├── forms
        ├── game_of_thrones.html
        ├── graph.html
        ├── index.html
        └── lib
```  
