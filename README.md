# EvalLLM2025 Information extraction challenge - Combined approach

## Description
This repository contains our work for the [EvalLLM2025 challenge](https://evalllm2025.sciencesconf.org/resource/page/id/5) about information extraction (Named Entity Recognition and Relations Extraction).

We began with an **exploratory data analysis** phase to better understand the training dataset.

Then, we worked on **two appoaches**:
- a **LLM-only approach** (see the repository at: …),
- a **combined approach** leveraging CamemBERT-bio-GLiNER for initial entity recognition, followed by a small LLM for fine-grained classificationt.

## Data exploration

###

## Combined approach

### Workflow
![Combined approach workflow](documentation/combined_approach_workflow_en.png)

The following steps will be performed:
1. NER with Camembert bio GLiNER with a simplified list of labels
2. Fine-grained classification with LLM to refine labels previously predicted (for train data, it is possible to do a mapping to have some results)
3. NER post-processing: correct start and end positions if needed and remove duplicates entities
4. RE with LLM
5. NER postprocessing

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

3. Results of each steps are available in predictions path (see `config.yml` file).