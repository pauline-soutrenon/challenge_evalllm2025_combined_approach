# Challenge EvalLLM2025 Extraction d'information - approche combinée

## Description
This repository contains our work for the [EvalLLM2025 challenge](https://evalllm2025.sciencesconf.org/resource/page/id/5) about information extraction (Named Entity Recognition and Relations Extraction).

We tested two different approaches: using only LLMs (see the repository at: ) and using a combined approach that leverages CamemBERT Bio GLiNER along with an LLM to refine the results.

## Workflow
[Combined approach workflow](documentation/combined_approach_workflow.png)

The following steps will be performed:
1. NER with Camembert bio GLiNER with a simplifie list of labels
2. NER with LLM to refine labels previously predicted (for train data, it is possoble to do a mapping to have some results)
3. NER postprocessing: correct start and end positions if needed and remove duplicates entities
4. NER results exploration
5. RE with LLM
6. NER postprocessing

## Installation
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

## Run

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

3. Results of each steps are available in predictions path (see `config.yml` file). Final output is called: `INRIA_mission_defense_et_securite_3.json`.