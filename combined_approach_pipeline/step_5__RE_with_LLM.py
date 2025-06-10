import json
import logging
import requests
import pandas as pd
from tqdm import tqdm
import get_prompt_for_llm


def request_ollama(prompt: str, model: str, ollama_url: str) -> str:
    """
    Send a prompt to the local Ollama API and retrieve the generated response

    Args:
        prompt (str): prompt to send
        model (str): Ollama model to use

    Raises:
        Exception: if the API request fails (i.e., the response status code is not 200)

    Returns:
        str: generated response
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(ollama_url, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data["response"]
    else:
        raise Exception(f"Erreur {response.status_code} : {response.text}")


def main(input_filename: str, output_filename: str, llm_attempts_requests: int, ollama_url: str, ollama_model: str):
    """
    Events / Relations Extraction (RE) with LLM

    Args:
        input_filename (str): JSON input to pocess
        output_filename (str): JSON output with RE results
        llm_attempts_requests (int): number of LLM attemps requests
        ollama_url (str): Ollama url
        ollama_model (str): Ollama model to use
    """

    # Load JSON file
    df = pd.read_json(input_filename)

    # Prepare an empty list for results
    data_to_export = []
    doc_id = 1
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Get text and entities
        text = row.get("text", "")
        entities = row.get("entities", [])
        events = []
        # logging.debug(f"\t{len(text)} caracters for text")
        # logging.debug(f"\t{len(str(entities))} caracters for entities")

        # Build prompt
        prompt = get_prompt_for_llm.french_prompt_for_re().replace("{text}", text).replace("{entities}", json.dumps(entities, ensure_ascii=False, indent=2))
        # logging.debug(f"\t{prompt}")
        if len(prompt) > 8000:
            logging.warning(f"\t[DOC {doc_id}] Prompt is too long ({len(prompt)} characters) and will be truncated.")

        for attempt in range(llm_attempts_requests):
            # logging.debug("\tprompt: ", prompt, "\n")
            # logging.info(f"\tLLM request attempt {attempt+1}/{llm_attempts_requests}")
            try:
                response = request_ollama(prompt, ollama_model, ollama_url)
                parsed = json.loads(response)
                events.append(parsed)
                break
            except json.JSONDecodeError:
                logging.info(f"\t[DOC {doc_id}] Invalid JSON from model: retry LLM request ({attempt+1}/{llm_attempts_requests})")
            except Exception as e:
                logging.warning(f"\t[DOC {doc_id}] Error from model:\n{e}")
                break
        else:
            # This runs if the for-loop is never broke = all attempts failed
            logging.warning(f"\t[DOC {doc_id}] All {llm_attempts_requests} attempts failed — no valid output.")
            events.append([])  # Fallback in case of total failure

        data_to_export.append({
            "text": text,
            "entities": entities,
            "events": events
        })

        # Save JSON after each document
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(data_to_export, f, ensure_ascii=False, indent=4)

        logging.info(f"\t✅ Progress saved after document {doc_id}")

        doc_id += 1

    logging.info(f"\t✅ JSON file saved at: {output_filename}")
