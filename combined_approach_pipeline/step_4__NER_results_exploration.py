import ast
import logging
import openpyxl
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.io as pio
from typing import Tuple, List
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict, Counter


def save_xlsx_with_fixed_width(df: pd.DataFrame, output_filename: str):
    """Save DataFrame to XLSX file with fixed column width

    Args:
        df (pd.DataFrame): DataFrame to save
        output_filename (str): output filename to save DataFrame in
    """
    # Save DataFrame to an Excel file with fixed column width
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

        # Access the openpyxl workbook and sheet
        workbook = writer.book
        sheet = workbook['Sheet1']

        # Set the column width for each column (e.g., width of 20 for all columns)
        column_widths = [20] * len(df.columns)  # Adjust width for all columns
        for i, width in enumerate(column_widths, start=1):  # 1-based index for columns
            sheet.column_dimensions[openpyxl.utils.get_column_letter(i)].width = width


def compare_label_distributions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare the distribution of entity labels before and after postprocessing

    Args:
        df (pd.DataFrame): DataFrame with 'predicted_entities' and 'postprocessed_entities' columns

    Returns:
        pd.DataFrame: DataFrame summarizing per-label statistics
    """

    # List all predicted labels (before postprocessing)
    predicted_labels = Counter()
    postprocessed_labels = Counter()
    losses_by_label = Counter()

    for _, row in df.iterrows():
        predicted = row['predicted_entities']  # Assumes column exists
        postprocessed = row['postprocessed_entities']

        pred_text_label_set = {(e['text'], e['label']) for e in predicted}
        post_text_label_set = {(e['text'], e['label']) for e in postprocessed}

        # Fill counters
        predicted_labels.update([e['label'] for e in predicted])
        postprocessed_labels.update([e['label'] for e in postprocessed])

        # Detect missing predictions (i.e., lost in postprocessing)
        lost_entities = pred_text_label_set - post_text_label_set
        losses_by_label.update([label for _, label in lost_entities])

    # Union of all labels
    all_labels = set(predicted_labels) | set(postprocessed_labels)

    # Build DataFrame
    comparison = []
    for label in sorted(all_labels):
        count_pred = predicted_labels[label]
        count_post = postprocessed_labels[label]
        loss = losses_by_label[label]
        kept = count_post / count_pred * 100 if count_pred else 0
        comparison.append({
            "label": label,
            "predicted": count_pred,
            "postprocessed": count_post,
            "kept_%": round(kept, 2),
            "lost": loss
        })

    return pd.DataFrame(comparison)


def generate_bar_graph_for_train_set__html_format(df: pd.DataFrame, output_html: str):
    """
    Generate a bar graph for train dataset showing correct, incorrect, and expected (true) entities per label

    Args:
        df (pd.DataFrame): DataFrame with correct predictions, incorrect predictions and true entities
        output_filename (str): HTML output filename
    """
    label_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'examples_correct': [], 'examples_incorrect': []})
    # Except TOTAL line
    df = df[df[df.columns[0]] != 'TOTAL']
    for _, row in df.iterrows():
        for ent in row['correct_predictions']:
            label = ent['label']
            label_stats[label]['correct'] += 1
            if len(label_stats[label]['examples_correct']) < 10:
                label_stats[label]['examples_correct'].append(ent['text'])
        for ent in row['incorrect_predictions']:
            label = ent['label']
            label_stats[label]['incorrect'] += 1
            if len(label_stats[label]['examples_incorrect']) < 10:
                label_stats[label]['examples_incorrect'].append(ent['text'])
    # Prepare data
    labels = []
    correct_counts = []
    incorrect_counts = []
    hover_texts_correct = []
    hover_texts_incorrect = []
    for label, stats in sorted(label_stats.items()):
        labels.append(label)
        correct_counts.append(stats['correct'])
        incorrect_counts.append(stats['incorrect'])
        hover_correct = f"<b>{label}</b><br>{stats['correct']} correct entities<br>Ex. : {', '.join(stats['examples_correct'])}<br>"
        hover_incorrect = f"<b>{label}</b>{stats['incorrect']} incorrect entities<br>Ex. incorrects: {', '.join(stats['examples_incorrect'])}"
        hover_texts_correct.append(hover_correct)
        hover_texts_incorrect.append(hover_incorrect)

    fig = go.Figure()
    # Correct entities
    fig.add_trace(go.Bar(
        x=labels,
        y=correct_counts,
        name='Corrects',
        marker_color='#7e9e94',
        customdata=hover_texts_correct,
        hovertemplate='%{customdata}<extra></extra>',
    ))
    # Incorrect entities
    fig.add_trace(go.Bar(
        x=labels,
        y=incorrect_counts,
        name='Incorrects',
        marker_color='#a57b88',
        customdata=hover_texts_incorrect,
        hovertemplate='%{customdata}<extra></extra>',
    ))
    fig.update_layout(
        barmode='stack',
        title='Prédictions correctes et incorrectes par label',
        xaxis_title='Label',
        yaxis_title='Nombre de prédictions',
        xaxis_tickangle=-45,
        legend_title='Prédictions',
        template='plotly_white',
        title_font=dict(size=20),
        xaxis_title_font=dict(size=20),
        yaxis_title_font=dict(size=20),
        xaxis_tickfont=dict(size=16),
        yaxis_tickfont=dict(size=16)
    )
    pio.write_html(fig, file=output_html, auto_open=False)


def generate_bar_plot_for_train_set__png_format(df: pd.DataFrame, output_filename, all: True):
    """
    Generate a bar graph for train dataset showing correct, incorrect, and expected (true) entities per label

    Args:
        df (pd.DataFrame): DataFrame with correct predictions, incorrect predictions and true entities
        output_filename (str): PNG output filename
    """
    label_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'expected': 0})
    df = df[df[df.columns[0]] != 'TOTAL']

    # Count predictions and true entities
    for _, row in df.iterrows():
        for ent in row.get('correct_predictions', []):
            label_stats[ent['label']]['correct'] += 1
        for ent in row.get('incorrect_predictions', []):
            label_stats[ent['label']]['incorrect'] += 1
        for ent in row.get('true_entities', []):
            label_stats[ent['label']]['expected'] += 1

    if not all:
        labels_to_exclude = {"LOCATION", "ORGANIZATION", "Maladie", "Date", "Période"}
        excluded_labels = {label: stats for label, stats in label_stats.items() if label in labels_to_exclude}

        label_stats = {label: stats for label, stats in label_stats.items() if label not in labels_to_exclude}

        if excluded_labels:
            logging.warning("\tExcluded labels:")
            for label, stats in excluded_labels.items():
                total = stats['expected']
                logging.warning(f"\t\t{label}: {total} expected | {stats['correct']} correct | {stats['incorrect']} incorrect")

    # Sort labels by expected (true_entities) count descending
    sorted_items = sorted(label_stats.items(), key=lambda x: x[1]['expected'], reverse=True)

    labels = [item[0] for item in sorted_items]
    correct_counts = [item[1]['correct'] for item in sorted_items]
    incorrect_counts = [item[1]['incorrect'] for item in sorted_items]
    expected_counts = [item[1]['expected'] for item in sorted_items]

    sns.set_context("notebook")
    x = range(len(labels))
    bar_width = 0.25

    plt.figure(figsize=(13, 6))
    plt.bar(
        [i - bar_width for i in x],
        expected_counts,
        width=bar_width,
        label='Entités attendues',
        color="#6d5e99",
        alpha=0.8,
        hatch='...',
        edgecolor='#6d5e9920'
    )
    plt.bar(
        x,
        correct_counts,
        width=bar_width,
        label='Prédictions correctes',
        color="#7e9e94",
        alpha=0.8
    )
    plt.bar(
        [i + bar_width for i in x],
        incorrect_counts,
        width=bar_width,
        label='Prédictions incorrectes',
        color="#a57b88",
        alpha=0.8,
        hatch='///',
        edgecolor='#a57b88'
    )

    # plt.title('Distribution des entités attendues et identifiées par label', fontsize=16, fontweight='bold')
    plt.xlabel('Labels des entités', fontsize=14)
    plt.ylabel('Nombre d\'entités', fontsize=14)
    plt.xticks(x, labels, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Légende')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    sns.despine()

    plt.tight_layout()
    plt.savefig(output_filename.replace("html", "png"), dpi=300, bbox_inches='tight')


def generate_bar_graph_for_test_dataset(df: pd.DataFrame, df_column_name_to_process: str, output_html: str):
    """Generate a bar graph for test dataset

    Args:
        df (pd.DataFrame): DtaFrame of test dataset
        df_column_name_to_process (str): column name to process
        output_html (str): HTML output filename
    """
    label_counts = defaultdict(lambda: {'count': 0, 'examples': []})

    # Except TOTAL line
    df = df[df[df.columns[0]] != 'TOTAL']

    for _, row in df.iterrows():
        for ent in row[df_column_name_to_process]:
            label = ent['label']
            label_counts[label]['count'] += 1
            if len(label_counts[label]['examples']) < 10:
                label_counts[label]['examples'].append(ent['text'])

    labels = []
    counts = []
    hover_texts = []

    for label, stats in sorted(label_counts.items()):
        labels.append(label)
        counts.append(stats['count'])
        hover = f"<b>{label}</b><br>{stats['count']} entities<br>Ex. : {', '.join(stats['examples'])}"
        hover_texts.append(hover)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=counts,
        name='Prédictions',
        marker_color='#7e9e94',
        customdata=hover_texts,
        hovertemplate='%{customdata}<extra></extra>',
    ))

    fig.update_layout(
        title='Entités prédites par label (jeu de test)',
        xaxis_title='Label',
        yaxis_title='Nombre de prédictions',
        xaxis_tickangle=-45,
        template='plotly_white',
        title_font=dict(size=20),
        xaxis_title_font=dict(size=20),
        yaxis_title_font=dict(size=20),
        xaxis_tickfont=dict(size=16),
        yaxis_tickfont=dict(size=16)
    )

    pio.write_html(fig, file=output_html, auto_open=False)

    logging.info(f"\t✅ HTML graph saved at: {output_html}")


def add_summing_row(df: pd.DataFrame, fine_grained_classification: bool) -> pd.DataFrame:
    """Add a summing row to DataFrame

    Args:
        df (pd.DataFrame): DataFrame to process
        fine_grained_classification (bool): if it's a combined approach or not

    Returns:
        pd.DataFrame: DataFrame with summing row
    """
    # Columns to sum
    columns_to_sum = ['nb_true_entities', 'nb_postprocessed_entities', 'nb_correct_predictions', 'nb_incorrect_predictions']
    if not fine_grained_classification:
        columns_to_sum.append('nb_true_entities_mapped')
    # Create the summary row with dtype=object to accept mixed types
    summary_row = pd.Series(data=[np.nan] * len(df.columns), index=df.columns, dtype=object)
    # Fill in totals for selected columns
    for col in columns_to_sum:
        summary_row[col] = df[col].sum()
    # Set the label in the first column
    summary_row[df.columns[0]] = 'TOTAL'

    # Compute global matching percentage
    total_correct = summary_row['nb_correct_predictions']
    total_predicted = summary_row['nb_postprocessed_entities']
    summary_row['matching_percentage'] = round((total_correct / total_predicted) * 100, 2) if total_predicted else 0

    # Append the summary row to the DataFrame
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    return df


def format_entities_positions(entities: List) -> List:
    """Format entities positions by making sure that start and end fields are in a list

    Args:
        entities (List): list of entities to process

    Returns:
        List: list of entities with formatted positions
    """
    # S'assure que les champs 'start' et 'end' sont dans une liste,
    # pour uniformiser le format des entités.
    for ent in entities:
        if not isinstance(ent['start'], list):
            ent['start'] = [ent['start']]
        if not isinstance(ent['end'], list):
            ent['end'] = [ent['end']]
    return entities


def match_entities(true_entities: List, predicted_entities: List) -> Tuple[List, List]:
    """
    Compare predicted entities with truth entities and classify them into correctly or incorrectly predicted entities

    Args:
        true_entities (List): list of truth entities, where each entity is a dictionary containing at least the keys 'text' and 'label'
        predicted_entities (List): list of predicted entities in the same format

    Returns:
        Tuple[List, List]:
            - list of correctly predicted entities (present in true_entities).
            - list of incorrectly predicted entities (not present in true_entities).
    """
    true_set = {(e["text"], e["label"]) for e in true_entities}
    correct = []
    incorrect = []
    for pred in predicted_entities:
        key = (pred["text"], pred["label"])
        if key in true_set:
            correct.append(pred)
        else:
            incorrect.append(pred)
    return correct, incorrect


def main(input_filename: str, output_filename: str, graph_filename: str, fine_grained_classification: bool):
    """Explore NER results

    Args:
        input_filename (str): CSV input to process
        output_filename (str): XLSX output with NER exploration
        graph_filename (str): filename of the graph to generate
        fine_grained_classification (bool): if it's a combined approach or not
    """
    is_test_set = "test_data" in input_filename

    # Load CSV input file
    df = pd.read_csv(input_filename)

    if not is_test_set:
        # Convert stringified lists of dicts to Python objects
        if fine_grained_classification:
            column_name_for_reference_entities = "true_entities"
        else:
            column_name_for_reference_entities = "true_entities_mapped"
            df['true_entities_mapped'] = df['true_entities_mapped'].apply(ast.literal_eval)

        df['true_entities'] = df['true_entities'].apply(ast.literal_eval)
        df['postprocessed_entities'] = df['postprocessed_entities'].apply(ast.literal_eval)

        # Normalize predicted entities: map labels and wrap start/end in lists
        df['postprocessed_entities'] = df['postprocessed_entities'].apply(format_entities_positions)

        # Apply matching and extract correct + incorrect predictions
        df[['correct_predictions', 'incorrect_predictions']] = df.apply(
            lambda row: pd.Series(match_entities(row[column_name_for_reference_entities], row['postprocessed_entities'])),
            axis=1
        )

        df['nb_true_entities'] = df['true_entities'].apply(len)
        if not fine_grained_classification:
            df['nb_true_entities_mapped'] = df[column_name_for_reference_entities].apply(len)
        df['nb_postprocessed_entities'] = df['postprocessed_entities'].apply(len)
        df['nb_correct_predictions'] = df['correct_predictions'].apply(len)
        df['matching_percentage'] = df.apply(
            lambda row: round((row['nb_correct_predictions'] / row['nb_postprocessed_entities'] * 100), 2)
            if row['nb_postprocessed_entities'] else 0,
            axis=1
        )
        df['nb_incorrect_predictions'] = df['incorrect_predictions'].apply(len)

        # Define columns to sum
        df = add_summing_row(df, fine_grained_classification)

        # Save updated CSV with custom columns width
        save_xlsx_with_fixed_width(df, output_filename)
        logging.info(f"\t✅ XLSX file saved at: {output_filename}")

        # Generate and save graph
        generate_bar_graph_for_train_set__html_format(df, graph_filename)
        logging.info(f"\t✅ HTML graph saved at: {graph_filename}")
        generate_bar_plot_for_train_set__png_format(df, graph_filename.replace(".html", "_all.png"), True)
        logging.info(f"\t✅ PNG graph (all) saved at: {graph_filename}")
        generate_bar_plot_for_train_set__png_format(df, graph_filename.replace(".html", "_filtered.png"), False)
        logging.info(f"\t✅ PNG graph (filtered) saved at: {graph_filename}")

    else:
        # Normalize predicted entities: map labels and wrap start/end in lists
        df['predicted_entities'] = df['predicted_entities'].apply(ast.literal_eval)
        df['postprocessed_entities'] = df['postprocessed_entities'].apply(ast.literal_eval)

        label_comparison_df = compare_label_distributions(df)
        label_comparison_df.to_excel(output_filename, index=False)
        logging.info(f"\t✅ XLSX file saved at: {output_filename}")

        # Generate and save graph
        generate_bar_graph_for_test_dataset(df, "predicted_entities", graph_filename.replace(".html", "_predicted_entities.html"))
        generate_bar_graph_for_test_dataset(df, "postprocessed_entities", graph_filename.replace(".html", "_postprocessed_entities.html"))
