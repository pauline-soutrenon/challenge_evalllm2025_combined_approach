# EvalLLM2025 Information extraction challenge - Combined approach

## Description
This repository contains our work for the [EvalLLM2025 challenge](https://evalllm2025.sciencesconf.org/resource/page/id/5) about information extraction (Named Entity Recognition and Relations Extraction).

We began with an **exploratory data analysis** phase to better understand the training dataset.

Then, we worked on **two appoaches**:
- a **LLM-only approach** (see [this repository](https://github.com/LucieBader/Challenge_EvalLLM2025)),
- a **combined approach** leveraging CamemBERT-bio-GLiNER for initial entity recognition, followed by a small LLM for fine-grained classification.

## Prerequisites

⚠️ **Training and test datasets are not provided in the repository.** They should be manually added to the data/ folder, in either train_data/ or test_data/, as appropriate.

## Data exploration
Data exploration code can be found in the `data_exploration/` folder.

### Installation
1. Create a Conda environment or a venv:
```bash
cd data_exploration/
conda create -n challenge_evalllm2025 python=3.12.9
conda activate challenge_evalllm2025
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Dowload a SpaCy model:
```bash
python -m spacy download fr_core_news_sm
```

### Run
First, the Jupyter notebook `data_exploration.ipynb` can be used to generate statistics on the training dataset. The resulting images are saved in the `training_data__images/` directory and a .csv file is also generated. It also works with the test dataset, although not all images will be generated in that case.

Then, the Python script `plot_document_graphes.py` can be used to generate a network graph for each event in every document (i.e., one graph per document-event pair). This step is limited to the training data, since entities and their labels are not available in the test dataset. The resulting images are saved in the `training_data__images/documents_graphes/` directory. A .txt file is also created, containing the number of entities for each event.

## Combined approach
Combined approach pipeline can be found in the `combined_approach_pipeline/` folder.

### Workflow
![Combined approach workflow](documentation/combined_approach_workflow_en.png)

The following steps will be performed:
1. **NER with Camembert bio GLiNER** and a simplified list of labels
2. **Fine-grained classification with LLM** to specify labels previously predicted (for train data, it is possible to do a mapping to have results)
3. **NER post-processing** to :
    - correct start and end positions if needed ;
    - retrieve all mentions of the same entity in the text ;
    - add unique ids ;
    - handle the non-overlapping of entities.
4. **NER results exploration**
5. **RE with LLM**
6. **RE postprocessing**

### Installation
1. Create a Conda environment or a venv:
```bash
cd combined_approach_pipeline/
conda create -n challenge_evalllm2025 python=3.12.9
conda activate challenge_evalllm2025
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Install [Ollama](https://ollama.com/download)

### Configuration file
All parameters ...

### Run

1. Before running the pipeline, make sure to :
    - Put your test or training data in the `data/` folder. 
    - Modify the configuration file (`combined_approach_pipeline/config.yml`) with at least:
        - the file to process (`data_path` argument) ;
        - the Ollama url (`ollama_url` argument) ;
        - the Ollama model to use for NER (`ollama_model_for_ner` argument) ;
        - the Ollama model to use for RE (`ollama_model_for_re` argument).
    - Activate you Conda environment or venv: `conda activate challenge_evalllm2025`
    - Have a Ollama server running

2. Run the pipeline:
```bash
python combined_approach_pipeline.py
```

3. Results of each steps are available in predictions path (see `config.yml` file). "predictions/predictions_{timestamp}/{timestamp}_predictions_last_version.json"