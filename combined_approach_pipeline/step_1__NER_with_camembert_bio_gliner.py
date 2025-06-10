import re
import json
# import torch
import logging
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from gliner import GLiNER
from typing import Tuple, List
from transformers import CamembertTokenizerFast

# Define a tokenizer to split text in smaller parts
tokenizer = CamembertTokenizerFast.from_pretrained("camembert-base")


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using basic punctuation rules
    Sentences are split on punctuation followed by whitespace (e.g., '.', '!', '?')

    Args:
        text (str): input text to split into sentences

    Returns:
        List[str]: list of sentences
    """
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in the given text using the tokenizer definedat at the beginning of the code

    Args:
        text (str): input text to count tokens from

    Returns:
        int: number of tokens
    """
    return len(tokenizer.encode(text, add_special_tokens=True))


def split_text(text: str, max_tokens: int = 384) -> List[Tuple[str, int]]:
    """
    Split a long text into smaller parts based on a maximum token limit, ensuring that each part ends at sentence boundaries to preserve semantic coherence

    Args:
        text (str): input text to split
        max_tokens (int, optional): maximum number of tokens allowed per text part, defaults to 384

    Returns:
        List[Tuple[str, int]]: list of tuples, where each tuple contains:
            - text part (str) that respects the token limit
            - The tokens length of that text part
    """
    sentences = split_sentences(text)

    text_parts = []
    current_text_part = ""

    for sentence in sentences:
        test_text = current_text_part + " " + sentence if current_text_part else sentence
        token_count = count_tokens(test_text)

        if token_count <= max_tokens:
            current_text_part = test_text
        else:
            # Finalize current part and store its token count
            if current_text_part:
                text_parts.append((current_text_part, count_tokens(current_text_part)))
            current_text_part = sentence

    if current_text_part:
        text_parts.append((current_text_part, count_tokens(current_text_part)))

    return text_parts


def predict_entities(text: str, labels: List[str], model, max_tokens: int = 384) -> List[dict]:
    """
    Predicts entities in a given text using with support for spliting text when the input exceeds the token limit.

    Args:
        text (str): input text for NER.
        labels (List[str]): custom labels.
        model (): model instance used for entity prediction.
        max_tokens (int, optional): maximum number of tokens allowed per text part. Defaults to 384 for camembert-bio-gliner.

    Returns:
        List[dict]: list of detected entities.
    """
    text_parts = split_text(text, max_tokens=max_tokens)
    all_entities = []

    for text_part, len_text_part in text_parts:
        entities = model.predict_entities(text_part, labels, threshold=0.5, flat_ner=True)
        for ent in entities:
            all_entities.append(ent)

    # Deduplicate entities by span and label
    unique_entities = {(e["start"], e["end"], e["label"]): e for e in all_entities}
    return list(unique_entities.values())


def main(input_filename: str, labels: List, predictions_path: str, output_json_filename: str, output_csv_filename: str):
    """
    Named Entity Recognition (NER) with CamemBERT-bio-GLiNER

    Args:
        input_filename (str): JSON input for NER processing
        labels (List): custom list of labels to pass to the model
        predictions_path (str): path to save predictions in
        output_json_filename (str): JSON output with NER
        output_csv_filename (str): CSV output with NER
    """
    logging.info(f"\tUsing following labels: {', '.join(labels)}")

    # Create folder if not exists to save predictions
    Path(predictions_path).mkdir(parents=True, exist_ok=True)

    # Display GPU device
    # logging.debug(f"\tGPU: {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}")  # GPU = True

    # Load JSON file
    with open(input_filename, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Load model
    model = GLiNER.from_pretrained("almanach/camembert-bio-gliner-v0.1")

    all_results = []
    all_results_for_comparison = []

    for example in tqdm(test_data, desc="Process documents"):
        text = example["text"]

        predicted_entities = predict_entities(text, labels, model, max_tokens=384)

        # for entity in predicted_entities:
        #     logging.debug(f"\t{entity["text"]} => {entity["label"]}")

        true_entities = example.get("entities", [])

        prediction = {
            "text": text,
            "entities": predicted_entities
        }
        all_results.append(prediction)

        prediction_for_truth_comparison = {
            "text": text,
            "true_entities": true_entities,
            "predicted_entities": predicted_entities
        }
        all_results_for_comparison.append(prediction_for_truth_comparison)

    # Save JSON
    with open(output_json_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    logging.info(f"\t✅ JSON file saved at: {output_json_filename}")

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "text": r["text"],
            "true_entities": json.dumps(r["true_entities"], ensure_ascii=False),
            "predicted_entities": json.dumps(r["predicted_entities"], ensure_ascii=False)
        }
        for r in all_results_for_comparison
    ])
    # Save prediction and truth values for comparison in CSV
    df.to_csv(output_csv_filename, index=False, encoding="utf-8")
    logging.info(f"\t✅ CSV file saved at: {output_csv_filename}")
