import ast
import json
import logging
import requests
import jsonschema
import pandas as pd
from tqdm import tqdm
import get_prompt_for_llm
from typing import Dict, List


def process_simple_mapping(entities: List, simple_mapping: Dict) -> List:
    """
    Process simple mapping for non-merged labels

    Args:
        entities (List): list of entities to process
        simple_mapping (Dict): dictionnary containing the mapping

    Returns:
        List: list of entities processed
    """
    for entity in entities:
        original_label = entity.get('label')
        if original_label in simple_mapping:
            entity['label'] = simple_mapping[original_label]
    return entities


def request_ollama_structured(prompt: str, model: str, ollama_url: str, schema) -> str:
    """
    Send a prompt to the local Ollama API and retrieves the generated response

    Args:
        prompt (str): prompt to send
        model (str): Ollama model to use

    Raises:
        Exception: if the API request fails (i.e., the response status code is not 200)

    Returns:
        str: generated response
    """
    response = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "format": "json",
            "json_schema": schema,
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["response"]
    else:
        raise Exception(f"Erreur {response.status_code} : {response.text}")


def request_per_list(label_type: str, doc_id: int, prompt: str, ollama_model: str, ollama_url: str, schema: Dict, llm_attempts_for_ner: int) -> Dict:
    """
    Run request per label list (e.g.: Maladie, Date, Period)

    Args:
        label_type (str): label type
        doc_id (int): document id
        prompt (str): prompt to send
        ollama_model (str): Ollama model to use
        ollama_url (str): Ollama url
        schema (Dict): schema used for validation
        llm_attempts_for_ner (int): number of LLM attemps requests

    Returns:
        Dict: LLM response
    """
    if len(prompt) > 8000:
        logging.warning(f"\t[DOC {doc_id}] Prompt is too long for label type {label_type} ({len(prompt)} caracters) and will be truncated.")

    for attempt in range(1, llm_attempts_for_ner + 1):
        # logging.debug("\tprompt: ", prompt, "\n")
        response = request_ollama_structured(prompt, ollama_model, ollama_url, schema)
        response = str(response).replace("non infectieuses", "non infectieuse")
        parsed_response = json.loads(response)
        # logging.debug("\tparsed_response: ", parsed_response)
        is_llm_output_valid = validate_llm_output(doc_id, parsed_response, schema)
        if is_llm_output_valid:
            return parsed_response
        else:
            logging.info(f"\t[DOC {doc_id}] Retry LLM request for label type {label_type} ({attempt}/{llm_attempts_for_ner})")

    logging.warning(f"\t[DOC {doc_id}] No valid output after {llm_attempts_for_ner} tentatives for label type {label_type}.")
    return {}


def validate_llm_output(doc_id: int, llm_output: dict, schema: Dict) -> bool:
    """
    Validate LLM output according to a schema

    Args:
        doc_id (int): document id
        llm_output (dict): LLM output to validate
        schema (Dict): schema used for validation

    Returns:
        bool: True if valid output, False if not
    """
    try:
        jsonschema.validate(instance=llm_output, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        logging.warning(f"\t[DOC {doc_id}] Invalid LLM output : {e.message}")
        return False


def process_llm_output(entities: List, llm_output: Dict, labels_to_refine: Dict) -> List:
    """
    Process LLM output (convert French label to expected ENglish one)

    Args:
        entities (List): list of entities to process
        llm_output (Dict): LLM output
        labels_to_refine (Dict): dictionnary to convert French label to ENglish expected ones

    Returns:
        List: list of entities processed
    """
    if llm_output:
        for ent in entities:
            entity_text = ent.get("text")
            if entity_text.lower() in llm_output:
                tmp_label = llm_output[entity_text.lower()]
                ent["label"] = labels_to_refine[tmp_label.lower()]
    return entities


def fine_grained_classification_with_llm(json_input_filename: str, json_output_filename: str, csv_input_filename: str, csv_output_filename: str, simple_mapping: Dict, ollama_url: str, ollama_model: str, llm_attempts_for_ner: int):
    """
    Fine-grained classification with LLM

    Args:
        json_input_filename (str): JSON input to process
        json_output_filename (str): JSON output
        csv_input_filename (str): CSV input to process
        csv_output_filename (str): CSV output
        simple_mapping (Dict): dictionnary containing mapping for non-merged labels
        ollama_url (str): Ollama url
        ollama_model (str): Ollama model to use
        llm_attempts_for_ner (int): number of LLM attemps requests
    """
    # Load JSON file
    df = pd.read_json(json_input_filename)

    # Prepare an empty column for refined results
    data_to_export = []
    llm_entities = []
    doc_id = 1
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Get text and entities list
        text = row.get("text", [])
        entities = row.get("entities", [])

        # Apply simple mapping for unmerged labels
        new_entities = process_simple_mapping(entities, simple_mapping)

        # Get lists of Maladie, Période and Date
        disease_texts = [ent['text'].lower() for ent in new_entities if ent['label'] == 'Maladie']
        disease_texts = list(set(disease_texts))
        period_texts = [ent['text'].lower() for ent in new_entities if ent['label'] == 'Période']
        period_texts = list(set(period_texts))
        date_texts = [ent['text'].lower() for ent in new_entities if ent['label'] == 'Date']
        date_texts = list(set(date_texts))

        # Labels to refine
        labels_to_refine = {
            "disease": {
                "infectieuse": "INF_DISEASE",
                "non infectieuse": "NON_INF_DISEASE"
            },
            "period": {
                "absolue": "ABS_PERIOD",
                "relative": "REL_PERIOD",
                "fuzzy": "FUZZY_PERIOD"
            },
            "date": {
                "absolue": "ABS_DATE",
                "relative": "REL_DATE"
            }
        }

        if disease_texts:
            label_type_to_refine = "disease"
            logging.info(f"\tProcessing {label_type_to_refine}")
            prompt_disease, schema_disease = get_prompt_for_llm.french_prompt_for_ner__disease(disease_texts)
            parsed_response_disease = request_per_list(label_type_to_refine, doc_id, prompt_disease, ollama_model, ollama_url, schema_disease, llm_attempts_for_ner)
            new_entities = process_llm_output(new_entities, parsed_response_disease, labels_to_refine[label_type_to_refine])
        if date_texts:
            label_type_to_refine = "date"
            logging.info(f"\tProcessing {label_type_to_refine}")
            prompt_date, schema_date = get_prompt_for_llm.french_prompt_for_ner__date(date_texts)
            parsed_response_date = request_per_list(label_type_to_refine, doc_id, prompt_date, ollama_model, ollama_url, schema_date, llm_attempts_for_ner)
            new_entities = process_llm_output(new_entities, parsed_response_date, labels_to_refine[label_type_to_refine])
        if period_texts:
            label_type_to_refine = "period"
            logging.info(f"\tProcessing {label_type_to_refine}")
            prompt_period, schema_period = get_prompt_for_llm.french_prompt_for_ner__period(period_texts)
            parsed_response_period = request_per_list(label_type_to_refine, doc_id, prompt_period, ollama_model, ollama_url, schema_period, llm_attempts_for_ner)
            new_entities = process_llm_output(new_entities, parsed_response_period, labels_to_refine[label_type_to_refine])

        llm_entities.append(new_entities)
        data_to_export.append({
            "text": text,
            "entities": new_entities
        })

        doc_id += 1

    # Add refined entities to the DataFrame
    df["llm_entities"] = llm_entities

    # Save JSON
    with open(json_output_filename, "w", encoding="utf-8") as f:
        json.dump(data_to_export, f, ensure_ascii=False, indent=4)
    logging.info(f"\t✅ JSON file saved at: {json_output_filename}")

    # Save CSV
    csv_df = pd.read_csv(csv_input_filename)
    csv_df["llm_entities"] = llm_entities
    csv_df.to_csv(csv_output_filename, index=False, encoding="utf-8")
    logging.info(f"\t✅ CSV file saved at: {csv_output_filename}")


def extract_and_map_labels(true_entities: str, label_mapping: Dict) -> List:
    """
    Extract and map true labels on predicted ones

    Args:
        true_entities (str): list of true entities (with expected labels)
        label_mapping (Dict): dictionnary containing the conversion from expected labels to number of LLM attemps requests

    Returns:
        List: list of entoties with remapped labels
    """
    # Check if the value is not NaN
    if not pd.isna(true_entities):
        # Evaluate the string to a list of dicts
        entities = ast.literal_eval(true_entities)
        if isinstance(entities, list):
            # Iterate over each entity and update the label
            for entity in entities:
                # Get the label and map it
                if 'label' in entity:
                    original_label = entity['label']
                    # Update the label
                    entity['label'] = label_mapping.get(original_label, original_label)
            return entities


def remap_true_entities(predictions_vs_truth_filename: str, label_mapping: Dict, output_filename: str):
    """
    Remap true entities labels so expected results can easily be compared with predicted ones

    Args:
        predictions_vs_truth_filename (str): CSV input filename
        label_mapping (Dict): dictionnary containing the conversion from expected labels to number of LLM attemps requests
        output_filename (str): CSV output filename
    """
    # Load CSV into DataFrame
    df = pd.read_csv(predictions_vs_truth_filename)
    # Apply the mapping function to the 'true_entities' column
    df['true_entities_mapped'] = df['true_entities'].apply(lambda x: extract_and_map_labels(x, label_mapping))
    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_filename, index=False)
    logging.info(f"\t✅ CSV file saved at: {output_filename}")
