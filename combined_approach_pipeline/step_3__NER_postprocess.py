import re
import uuid
import json
import logging
import pandas as pd


def detect_overlapping(start1: int, end1: int, start2: int, end2: int) -> int:
    """
    Detect whether two entities spans overlap

    Args:
        start1 (int): start index of the first span
        end1 (int): end index of the first span
        start2 (int): start index of the second span
        end2 (int): end index of the second span

    Returns:
        int: 1 if the spans overlap, 0 otherwise
    """
    return int(max(start1, start2) < min(end1, end2))


def correct_positions(full_text, annotated_item, entity, entity_text, start, end, corrected_items, corrected_items_not_in_full_text):
    """
    Correct start and end positions of an entity within a given full text

    Args:
        full_text (str): complete text in which entity positions are to be validated
        annotated_item (dict): dictionary representing an annotated item, expected to contain an 'entities' list
        entity (dict): dictionary representing a single entity with 'text', 'label', 'start' and 'end'
        entity_text (str): text of the entity to be located in the full text
        start (int): original start position of the entity
        end (int): original end position of the entity
        corrected_items (int): number of corrected entities
        corrected_items_not_in_full_text (int): number of entities not found in the full text

    Returns:
        tuple:
            - annotated_item (dict): updated annotated item with possibly corrected entity positions
            - entity (dict): updated entity dictionary
            - corrected_items (int): number of corrected entities
            - corrected_items_not_in_full_text (int): number of entities not found in the full text
    """

    # Try to find all possible correct positions
    matches = [m.start() for m in re.finditer(re.escape(entity_text), full_text)]

    if matches:
        existing_spans = [
            (ent["start"][0], ent["end"][0])
            for ent in annotated_item["entities"]
            if isinstance(ent.get("start"), list) and isinstance(ent.get("end"), list)
        ]

        added = 0
        for match_start in matches:
            match_end = match_start + len(entity_text)
            # Check than new span doesn't overlap an existing one
            if not any(detect_overlapping(match_start, match_end, ex_start, ex_end) for ex_start, ex_end in existing_spans):
                if added == 0:
                    # Correct current entity
                    entity["start"] = [match_start]
                    entity["end"] = [match_end]
                else:
                    # Create new entity
                    new_entity = {
                        "text": entity_text,
                        "label": entity.get("label", ""),
                        "start": [match_start],
                        "end": [match_end]
                    }
                    annotated_item["entities"].append(new_entity)
                existing_spans.append((match_start, match_end))
                corrected_items += 1
                added += 1

        if added:
            logging.debug(f"\t\tPOSITIONS: '{entity_text}' [{start}-{end}] ({entity['label']}) corrected. {added} match(es) added (no overlap).")
        else:
            logging.debug(f"\t\tPOSITIONS: '{entity_text}' [{start}-{end}] ({entity['label']}) found, but all matches overlapped existing entities.")

    else:
        logging.debug(f"\t\tPOSITIONS: '{entity_text}' [{start}-{end}] ({entity['label']}) doesn't match and was not found in text.")
        new_entity = full_text[start:end]
        entity["text"] = new_entity
        corrected_items_not_in_full_text += 1

    return annotated_item, entity, corrected_items, corrected_items_not_in_full_text


def main(json_input_filename: str, json_output_filename: str, csv_input_filename: str, csv_output_filename: str):
    """Postprocess NER results

    Args:
        json_input_filename (str): JSON input to postprocess
        json_output_filename (str): JSON postprocessed output
        csv_input_filename (str): CSV input to postprocess
        csv_output_filename (str): CSV postprocessed output
    """

    # Load JSON file
    with open(json_input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Parse data
    i = 1
    corrected_items = 0
    corrected_items_not_in_full_text = 0
    duplicated_items = 0

    for item in data:
        logging.debug(f"\tDocument {i}:")
        full_text = item.get("text", "")
        for entity in item.get("entities", []):
            entity_text = entity.get("text", "")
            # 1. Convert start and end to lists of ints if not already done
            if isinstance(entity.get("start"), int):
                entity["start"] = [entity["start"]]
            if isinstance(entity.get("end"), int):
                entity["end"] = [entity["end"]]

            # 2. Validate/correct positions
            # Check if the span matches the entity text
            start = entity["start"][0]
            end = entity["end"][0]
            if full_text[start:end] != entity_text:
                item, entity, corrected_items, corrected_items_not_in_full_text = correct_positions(full_text, item, entity, entity_text, start, end, corrected_items, corrected_items_not_in_full_text)

            # 3. Add id
            if "id" not in entity:
                entity["id"] = str(uuid.uuid4())

            # 4. Remove score
            if "score" in entity:
                del entity["score"]

        # Remove duplicates
        entities = item.get("entities", [])
        original_count = len(entities)
        unique_entities = []
        seen = set()
        for entity in entities:
            start_val = entity["start"][0] if isinstance(entity["start"], list) else entity["start"]
            end_val = entity["end"][0] if isinstance(entity["end"], list) else entity["end"]
            label = entity.get("label")
            key = (start_val, end_val, label)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        item["entities"] = unique_entities
        new_count = len(unique_entities)

        if new_count < original_count:
            logging.debug(f"\t\t{original_count - new_count} duplicate(s) removed.")
            duplicated_items += original_count - new_count
        else:
            logging.debug("\t\tNo duplicates removed.")

        i += 1

    logging.info(f"\t{corrected_items} corrected entities + {corrected_items_not_in_full_text} (not in full text) = {corrected_items + corrected_items_not_in_full_text} corrected entities")
    logging.info(f"\t{duplicated_items} duplicated entities removed")

    # Save the JSON file
    with open(json_output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info(f"\t✅ JSON file saved at: {json_output_filename}")

    # Save the CSV file
    df = pd.read_csv(csv_input_filename)
    df["postprocessed_entities"] = [item["entities"] for item in data]
    df.to_csv(csv_output_filename, index=False)
    logging.info(f"\t✅ CSV saved at: {csv_output_filename}")
    # Check if texts match between CSV and JSON before merging
    for idx, (csv_text, json_item) in enumerate(zip(df["text"], data)):
        if str(csv_text).strip() != json_item.get("text", "").strip():
            logging.warning(f"Text mismatch at row {idx + 1}")
