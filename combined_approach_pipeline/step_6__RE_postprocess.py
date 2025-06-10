import json
import logging
from jsonschema import validate, ValidationError


def main(input_filename_to_postprocess: str, output_filename: str):
    """Postprocess RE results

    Args:
        input_filename_to_postprocess (str): JSON input to postprocess
        output_filename (str): JSON postprocessed output
    """
    # JSON Schema for validating the 'events' field
    events_schema = {
        "type": "array",
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attribute": {"type": "string"},
                    "occurrences": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["attribute", "occurrences"],
                "additionalProperties": False
            }
        }
    }

    # Load JSON file
    with open(input_filename_to_postprocess, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate events for each item
    doc_id = 1
    ok_events = 0
    empty_events = 0
    non_ok_events_invalid_format = 0
    non_ok_events_invalid_attribute = 0
    for item in data:
        events = item.get("events")
        try:
            if events != [[]]:
                # Check if valid format
                validate(instance=events, schema=events_schema)

                # Check central/associated counts
                valid_event_groups = True
                for group in events:
                    central_count = sum(1 for e in group if e.get("attribute") == "evt:central_element")
                    associated_count = sum(1 for e in group if e.get("attribute") == "evt:associated_element")
                    if central_count != 1 or associated_count < 1:
                        valid_event_groups = False
                        break

                if valid_event_groups:
                    ok_events += 1
                else:
                    logging.warning(f"\t[DOC {doc_id}] Invalid attribute: must have 1 central and ≥1 associated element per group. Replacing with empty list.")
                    item["events"] = []
                    non_ok_events_invalid_attribute += 1
            else:
                logging.debug(f"\t[DOC {doc_id}] Empty events.")
                item["events"] = []
                empty_events += 1
        except ValidationError:
            logging.warning(f"\t[DOC {doc_id}] Invalid events, replacing with empty list.")
            item["events"] = []
            non_ok_events_invalid_format += 1
        doc_id += 1

    logging.info(f"\t{ok_events}/{len(data)} ok events")
    logging.info(f"\t{empty_events}/{len(data)} empty events")
    logging.info(f"\t{non_ok_events_invalid_format}/{len(data)} non ok events (invalid format)")
    logging.info(f"\t{non_ok_events_invalid_attribute}/{len(data)} non ok events (invalid attribute)")

    # Save JSON
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logging.info(f"\t✅ JSON file saved at: {output_filename}")
