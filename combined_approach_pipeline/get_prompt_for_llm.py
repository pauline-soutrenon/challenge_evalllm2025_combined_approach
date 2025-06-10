import logging
from typing import Dict, Tuple, List


def french_prompt_for_ner__disease(disease_list: List) -> Tuple[str, Dict]:
    """Get French NER prompt for diseases

    Args:
        disease_list (List): list of disease entities

    Returns:
        Tuple[str, Dict]: prompt and schema
    """
    disease_text = "\n".join(disease_entity for disease_entity in disease_list)
    prompt_disease = (
        "Pour chaque ligne de la liste ci-dessous, indique si c'est une maladie 'infectieuse' ou 'non infectieuse'.\n"
        "Réponds uniquement avec une liste JSON au format :\n"
        "[{\"sida\": \"infectieuse\"}, {\"sclérose en plaques\": \"non infectieuse\"}]\n"
        "Ne rajoute pas d'élément et n'invente pas de nouveau type.\n\n"
        f"Liste à analyser :\n{disease_text}"
    )

    schema_disease = {
        "type": "object",
        "patternProperties": {
            "^.*$": {
                "type": "string",
                "enum": ["infectieuse", "non infectieuse"]
            }
        },
        "additionalProperties": False
    }

    if len(prompt_disease) > 8000:
        logging.warning(f"Prompt is too long ({len(prompt_disease)} caracters) and will be truncated.")

    return prompt_disease, schema_disease


def french_prompt_for_ner__period(period_list: List) -> Tuple[str, Dict]:
    """Get French NER prompt for periods

    Args:
        period_list (List): list of period entities

    Returns:
        Tuple[str, Dict]: prompt and schema
    """
    period_text = "\n".join(period_entity for period_entity in period_list)
    prompt_period = (
        "Pour chaque ligne de la liste ci-dessous, indique si c'est une période 'absolue', 'relative ou 'fuzzy'.\n"
        "Réponds uniquement avec une liste JSON au format :\n"
        "[{\"janvier 2018\": \"absolue\"}, {\"la semaine prochaine\": \"relative\"}, {\"ces dernières années\": \"fuzzy\"}]\n"
        "Ne rajoute pas d'élément et n'invente pas de nouveau type.\n\n"
        f"Liste à analyser :\n{period_text}"
    )

    schema_period = {
        "type": "object",
        "patternProperties": {
            "^.*$": {
                "type": "string",
                "enum": ["absolue", "relative", "fuzzy"]
            }
        },
        "additionalProperties": False
    }

    if len(prompt_period) > 8000:
        logging.warning(f"Prompt is too long ({len(prompt_period)} caracters) and will be truncated.")

    return prompt_period, schema_period


def french_prompt_for_ner__date(date_list: List) -> Tuple[str, Dict]:
    """Get French NER prompt for periods

    Args:
        date_list (List): list of date entities

    Returns:
        Tuple[str, Dict]: prompt and schema
    """
    date_text = "\n".join(date_entity for date_entity in date_list)
    prompt_date = (
        "Pour chaque ligne de la liste ci-dessous, indique si c'est une date 'absolue' ou 'relative'.\n"
        "Réponds uniquement avec une liste JSON au format :\n"
        "[{\"demain\": \"relative\"}, {\"lundi dernier\": \"relative\"}, {\"8 janvier 2015\": \"absolue\"}]\n"
        "Ne rajoute pas d'élément et n'invente pas de nouveau type.\n\n"
        f"Liste à analyser :\n{date_text}"
    )

    schema_date = {
        "type": "object",
        "patternProperties": {
            "^.*$": {
                "type": "string",
                "enum": ["absolue", "relative"]
            }
        },
        "additionalProperties": False
    }

    if len(prompt_date) > 8000:
        logging.warning(f"Prompt is too long ({len(prompt_date)} caracters) and will be truncated.")

    return prompt_date, schema_date


def french_prompt_for_re() -> str:
    """Get French RE prompt

    Returns:
        str: prompt
    """
    return """
### Tâche :
Tu es un modèle de langage chargé d'extraire des **événements épidémiologiques ou sanitaires** entre des entités.

Tu reçois un texte et une liste d'entités avec des identifiants.

### Contraintes :
- Un texte peut contenir **entre 0 et 4 événements maximum**.
- Un évènement est composé de :
    - une entité centrale unique (evt:central_element) pouvant UNIQUEMENT être : INF_DISEASE, NON_INF_DISEASE, PATHOGEN, BIO_TOXIN, TOXIC_C_AGENT, RADIOISOTOPE ou EXPLOSIVE ;
    - une ou plusieurs entités associées (evt:associated_element) pouvant UNIQUEMENT être : LOCATION, DATE ou PERIOD.
- Un événement ne peut impliquer **au maximum que 10 entités**.
- Les entités doivent être représentées par **leur identifiant unique (id)**, et non leur texte brut.

### Sortie attendue :
- Renvoie UNIQUEMENT la liste des événements (events) sous forme de **JSON valide**.
- Ne fournis AUCUNE explication, ni texte en plus, ni commentaires.
- N’ajoute PAS de préfixes comme "json" ou "```".

### Exemple de sortie :
[
    [
        {
            "attribute": "evt:central_element",
            "occurrences": [
                "euDoZG0Ang"
            ]
        },
        {
            "attribute": "evt:associated_element",
            "occurrences": [
                "dtPEACcPAJ"
            ]
        }
    ],
    [
        {
            "attribute": "evt:central_element",
            "occurrences": [
                "45DoZG0Ang",
                "oPjzhgd56"
            ]
        },
        {
            "attribute": "evt:associated_element",
            "occurrences": [
                "zedlhkhiu"
            ]
        }
    ]
]

---

Texte à traiter :
{text}

Entités à traiter :
{entities}
"""
