import os
import time
import yaml
import logging
from datetime import datetime
from typing import Dict
import step_1__NER_with_camembert_bio_gliner, step_2__NER_with_LLM, step_3__NER_postprocess, step_4__NER_results_exploration, step_5__RE_with_LLM, step_6__RE_postprocess

# Configure logging
log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'NER_pipeline.log')
# Create logs directory if it doesn't exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(file_formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

global timestamp
timestamp = datetime.today().strftime('%Y%m%d%H%M%S')


def load_config() -> Dict:
    """
    Load configuration from YAML file
    """
    logging.info("Load configuration")
    with open('config.yml', 'r') as f:
        return yaml.safe_load(f)


def main():
    logging.info("### START NER PIPELINE ###")

    config = load_config()

    start_time = time.time()

    logging.info("### STEP 1: NER with Camembert Bio GLiNER ###")
    data_filename = config["data_filename"]
    mapping_name = config["mapping_name"]
    labels = config["tested_labels"][mapping_name]
    combined_approach = config["combined_approach"]
    ollama_url = config["ollama_url"]
    predictions_path = config["predictions_path"].replace("{timestamp}", timestamp)
    json_output_step_1 = config["json_output_step_1"].replace("{timestamp}", timestamp)
    csv_output_step_1 = config["csv_output_step_1"].replace("{timestamp}", timestamp)
    step_1__NER_with_camembert_bio_gliner.main(data_filename, labels, predictions_path, json_output_step_1, csv_output_step_1)

    csv_output_step_2 = config["csv_output_step_2"].replace("{timestamp}", timestamp)
    ollama_model_for_ner = config["ollama_model_for_ner"]
    if combined_approach:
        logging.info(f"### STEP 2: NER labels refining with {ollama_model_for_ner} ###")
        llm_attempts_requests_for_ner = config["llm_attempts_requests_for_ner"]
        simple_mapping = config["simple_mapping"][mapping_name]
        json_output_step_2 = config["json_output_step_2"].replace("{timestamp}", timestamp)
        step_2__NER_with_LLM.fine_grained_classification_with_llm(json_output_step_1, json_output_step_2, csv_output_step_1, csv_output_step_2, simple_mapping, ollama_url, ollama_model_for_ner, llm_attempts_requests_for_ner)
    else:
        logging.info("### STEP 2: NER labels mapping on reference ###")
        complete_mapping = config["complete_mapping"][mapping_name]
        step_2__NER_with_LLM.remap_true_entities(csv_output_step_1, complete_mapping, csv_output_step_2)

    logging.info("### STEP 3: NER postprocessing ###")
    json_output_step_3 = config["json_output_step_3"].replace("{timestamp}", timestamp)
    csv_output_step_3 = config["csv_output_step_3"].replace("{timestamp}", timestamp)
    step_3__NER_postprocess.main(json_output_step_2, json_output_step_3, csv_output_step_2, csv_output_step_3)

    logging.info("### STEP 4: NER results exploration ###")
    xlsx_output_step_4 = config["xlsx_output_step_4"].replace("{timestamp}", timestamp)
    graph_step_4 = config["graph_step_4"].replace("{timestamp}", timestamp)
    step_4__NER_results_exploration.main(csv_output_step_3, xlsx_output_step_4, graph_step_4, combined_approach)

    ollama_model_for_re = config["ollama_model_for_re"]
    logging.info(f"### STEP 5: RE with {ollama_model_for_re} ###")
    llm_attempts_requests_for_re = config["llm_attempts_requests_for_re"]
    json_output_step_5 = config["json_output_step_5"].replace("{timestamp}", timestamp)
    step_5__RE_with_LLM.main(json_output_step_3, json_output_step_5, llm_attempts_requests_for_re, ollama_url, ollama_model_for_re)

    logging.info("### STEP 6: RE postprocessing ###")
    json_output_step_6 = config["json_output_step_6"].replace("{timestamp}", timestamp)
    step_6__RE_postprocess.main(json_output_step_5, json_output_step_6)

    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    logging.info(f"--- TIME EXECUTION: {elapsed_time} seconds / {hours}h {minutes}m {seconds}s ---")

    logging.info("### END NER PIPELINE ###")


if __name__ == '__main__':
    main()
