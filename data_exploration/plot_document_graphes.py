import os
import json
import pandas as pd
import networkx as nx
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt


def format_data(file_with_data_to_plot: str, path_to_save_plots: str, text_file_with_events_description: str) -> Dict:
    """Format data to plot network for each event

    Args:
        file_with_data_to_plot (str): file with data to plot
        path_to_save_plots (str): path to save images and txt file with events description

    Returns:
        Dict: dictionnary with all events
    """
    # Load data
    with open(file_with_data_to_plot, 'r') as file:
        data = json.load(file)

    # Init variables
    document_nb = 1
    for_txt_file = ""
    events_to_plot = {}
    events_per_doc = {0: 0}

    # Parse data
    for document in data:
        entities = document["entities"]
        entities_dict = get_element_text_from_id(entities)
        events = document["events"]
        # Check if documents has event(s)
        if len(events) > 0:
            for_txt_file += f"### Document {document_nb} - {len(events)} event(s) ###\n"
            if len(events) not in events_per_doc.keys():
                events_per_doc[len(events)] = 1
            else:
                events_per_doc[len(events)] += 1
            event_nb = 1
            for event in events:
                for_txt_file += f"\tEvent {event_nb} - {len(event)} entities\n"
                event_tmp = {}
                central_elements = []
                associated_elements = []
                for entity in event:
                    if entity["attribute"] == "evt:central_element":
                        entity_id = entity["occurrences"][0]
                        central_entity_dict = (entities_dict.get(entity_id, "error"))
                        if central_entity_dict != "error":
                            central_entity = central_entity_dict["text"] + "\n(" + central_entity_dict["label"] + ")"
                            central_elements.append(central_entity)
                        else:
                            print(f"ERROR: document_{document_nb}_event_{event_nb}_entities_{len(event)} - central element {entity_id} does not exists.")
                    elif entity["attribute"] == "evt:associated_element":
                        entity_id = entity["occurrences"][0]
                        entity_dict_associated = entities_dict.get(entity_id, "error")
                        if entity_dict_associated != "error":
                            associated_entity = entity_dict_associated["text"] + "\n(" + entity_dict_associated["label"] + ")"
                            associated_elements.append(associated_entity)
                        else:
                            print(f"ERROR: document_{document_nb}_event_{event_nb}_entities_{len(event)} - associated element {entity_id} does not exists.")
                central_elements = central_elements * len(associated_elements)
                associated_elements_without_line_return = [a.replace("\n", " ") for a in associated_elements]
                for_txt_file += f"\t\t{central_elements[0].replace("\n", " ")} : {", ".join(associated_elements_without_line_return)}\n"
                event_tmp = {"from": central_elements, "to": associated_elements}
                events_to_plot[f"document_{document_nb}_event_{event_nb}_entities_{len(event)}"] = event_tmp
                event_nb += 1
        else:
            for_txt_file += f"### Document {document_nb} - no event ###\n"
            events_per_doc[0] += 1
        document_nb += 1

    # Calculate some stats (chart already in data_exploration.ipynb)
    for_txt_file += "### Events per doc: ###\n"
    for k, v in events_per_doc.items():
        for_txt_file += f"{k:<2} event → {v:>2} documents\n"
    total = sum(events_per_doc.values())
    for_txt_file += f"Total sum: {total}"

    # Save events description in a .txt file
    with open(os.path.join(path_to_save_plots, text_file_with_events_description), "w+") as f:
        f.write(for_txt_file)
    f.close()

    return events_to_plot


def get_element_text_from_id(entities: Dict) -> Dict:
    """
    Extract entity information from a list of entities and maps it by unique entity ID

    Args:
        entities (Dict): a list of dictionaries, where each dictionary represents an entity with keys 'id', 'text', 'start', 'end', and 'label'

    Returns:
        Dict: a dictionary mapping each unique entity ID to its corresponding entity data (text, start, end, label). If duplicate IDs are found, an error message is printed
    """
    entities_dict = {}
    for entity in entities:
        if entity["id"] not in entities_dict.keys():
            entities_dict[entity["id"]] = {'text': entity["text"], 'start': entity["start"], 'end': entity["end"], 'label': entity["label"]}
        else:
            print("ERROR: id is not in entities_dict.")
    return entities_dict


def insert_linebreaks(text: str, max_len: int) -> str:
    """Insert linebreaks in text

    Args:
        text (str): text to insert linebreaks in
        max_len (int): maximum number of characters allowed per line before inserting a linebreak

    Returns:
        str: text with linebreaks
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) + 1 <= max_len:
            if current_line:
                current_line += ' ' + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return '\n'.join(lines)


def plot_network(df: pd.DataFrame, filename: str, path_to_save_plots: str):
    """Plot network and save it in file

    Args:
        df (pd.DataFrame): dataframe to plot
        filename (str): network filename
        path_to_save_plots (str): path to save images
    """
    G = nx.from_pandas_edgelist(df, 'from', 'to')

    pos = nx.kamada_kawai_layout(G)

    plt.figure(figsize=(7, 5))

    labels = {node: insert_linebreaks(node, 13) for node in G.nodes()}
    from_nodes = set(df['from'])
    node_colors = ["#99C1B9" if node in from_nodes else "#f2d0a9" for node in G.nodes()]

    nx.draw(
        G,
        pos=pos,
        labels=labels,
        with_labels=True,
        node_size=1200,
        node_color=node_colors,
        node_shape="o",
        alpha=0.9,
        linewidths=1.5,
        font_size=10,
        font_color="black",
        width=1.2,
        edge_color="gray"
    )

    output_path = os.path.join(path_to_save_plots, filename)
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def main(file_with_data_to_plot: str, path_to_save_plots: str, text_file_with_events_description: str):
    final_dict = format_data(file_with_data_to_plot, path_to_save_plots, text_file_with_events_description)
    for i, (event, values_for_network) in tqdm(enumerate(final_dict.items(), start=1), total=len(final_dict)):
        df = pd.DataFrame(values_for_network)
        filename = f"E{i}_{event}.png"
        plot_network(df, filename, path_to_save_plots)


if __name__ == "__main__":
    file_with_data_to_plot = "../data/training_data/20250428_NP_train-evalLLM.json"
    path_to_save_plots = "training_data__images/documents_graphes/"
    text_file_with_events_description = "training_data__documents_graphes.txt"
    Path(path_to_save_plots).mkdir(parents=True, exist_ok=True)
    main(file_with_data_to_plot, path_to_save_plots, text_file_with_events_description)
