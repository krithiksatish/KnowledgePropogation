import argparse
import csv
import os
import random
import uuid
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from wikipedia_edit_scrape_tool.scrape_page_content import get_del_add_text, get_monthly_revision_comparison_links, get_text_differences
from wikipedia_edit_scrape_tool.sentence_simlarity import SentenceSimilarityCalculator

def process_page(page_title, mode, directory):
    if mode == 'wiki':
        handle_wiki_mode(page_title, directory)
    elif mode == 'similarity':
        handle_similarity_mode(page_title, directory)

def handle_wiki_mode(page_title, directory):
    sim = SentenceSimilarityCalculator()
    csv_file = os.path.join(directory, f"{page_title.replace('/', '_')}.csv")

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Edit ID', 'Old Text', 'New Text', 'Old Has Textual Change', 'New Has Textual Change', 
                   'Cleaned Old Text', 'Cleaned New Text', 'Difference', 'Is Significant Edit', 
                   'References', 'Editor Name', 'Editor ID', 'Edit Timestamp', 'Edit URL']
        writer.writerow(headers)

        revisions_data = get_monthly_revision_comparison_links(page_title)
        for url, meta in revisions_data:
            editor_name, editor_id, edit_timestamp = meta['editor'], meta['editor_id'], meta['edit_time']
            edit_timestamp = datetime.fromisoformat(edit_timestamp.rstrip('Z')).replace(tzinfo=timezone.utc)
            
            print(f"  - Processing revision comparison: {url}")
            context_pairs_list = get_del_add_text(url)
            print(f"    - Found {len(context_pairs_list)} context pairs")

            for old_text, new_text, cleaned_old_text, cleaned_new_text, refs, old_has_content, new_has_content in context_pairs_list:
                significant_edit = sim.is_significant_edit(cleaned_old_text, cleaned_new_text)
                differences = get_text_differences(cleaned_old_text, cleaned_new_text)
                edit_id = str(uuid.uuid4())
                writer.writerow([edit_id, old_text, new_text, old_has_content, new_has_content, cleaned_old_text, cleaned_new_text,
                                 differences, significant_edit, refs if refs else '', editor_name, editor_id, edit_timestamp, url])

def handle_similarity_mode(page_title, directory):
    sim = SentenceSimilarityCalculator()
    csv_file = os.path.join(directory, f"sim_{page_title.replace('/', '_')}.csv")

    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Cleaned Old Text', 'Cleaned New Text', 'Difference', 
                   'BERT Similarity', 'RoBERTa Similarity', 
                   'Subject and Verb (SM)', 'Subject and Verb (LG)', 
                   'Contradiction', 'Contradiction Confidence', 
                   'Edit Distance', 'Significant Edit']
        writer.writerow(headers)

        revisions_data = get_monthly_revision_comparison_links(page_title)
        for url, _ in revisions_data:
            print(f"  - Processing revision comparison: {url}")
            context_pairs_list = get_del_add_text(url)
            print(f"    - Found {len(context_pairs_list)} context pairs")

            for _, _, cleaned_old_text, cleaned_new_text, _, _, new_has_content in context_pairs_list:
                if new_has_content:
                    differences = get_text_differences(cleaned_old_text, cleaned_new_text)
                    # Additional data processing as needed
                    bert_similarity = sim.bert_cosine_similarity(cleaned_old_text, cleaned_new_text)
                    roberta_similarity = sim.roberta_cosine_similarity(cleaned_old_text, cleaned_new_text)
                    has_subject_and_verb_sm = sim.has_subject_and_verb(cleaned_new_text, model='sm')
                    has_subject_and_verb_lg = sim.has_subject_and_verb(cleaned_new_text, model='lg')
                    label, confidence = sim.check_contradiction(cleaned_old_text, cleaned_new_text)
                    is_contradiction = label == 'contradiction'
                    confidence = confidence if is_contradiction else 0
                    edit_distance = sim.edit_distance(cleaned_old_text, cleaned_new_text)
                    significant_edit = sim.is_significant_edit(cleaned_old_text, cleaned_new_text)

                    writer.writerow([cleaned_old_text, cleaned_new_text, differences, bert_similarity, 
                                     roberta_similarity, has_subject_and_verb_sm, has_subject_and_verb_lg, 
                                     is_contradiction, confidence, edit_distance, significant_edit])

def generate_annotation_set(all_lines, directory, target_count=300):
    sim = SentenceSimilarityCalculator()
    edits_collected = 0
    random.shuffle(all_lines)  # Shuffle to randomize the page selection

    csv_file = os.path.join(directory, 'random_annotations.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Raw Old Text', 'Raw New Text', 'Cleaned Old Text', 'Cleaned New Text', 'Difference', 
                   'BERT Similarity', 'Has Subject and Vert', 'Edit URL']
        writer.writerow(headers)

        # Iterate over the shuffled list of pages
        for line in all_lines:
            if edits_collected >= target_count:
                break
            page_title = line.strip()
            revisions_data = get_monthly_revision_comparison_links(page_title)

            # Gather context pairs from each revision
            for url, meta in revisions_data:
                context_pairs_list = get_del_add_text(url)
                random.shuffle(context_pairs_list)  # Shuffle edits to randomize selection

                for edit in context_pairs_list:
                    old_text, new_text, cleaned_old_text, cleaned_new_text, _, _, new_has_content = edit
                    if new_has_content:
                        differences = get_text_differences(cleaned_old_text, cleaned_new_text)
                        bert_similarity = sim.bert_cosine_similarity(cleaned_old_text, cleaned_new_text)
                        has_subject_and_verb = sim.has_subject_and_verb(cleaned_new_text, model='sm')

                        # Write to CSV including the URL
                        writer.writerow([old_text, new_text, cleaned_old_text, cleaned_new_text, differences, 
                                         bert_similarity, has_subject_and_verb, url])
                        edits_collected += 1
                        print(f"Collected {edits_collected} edits for annotation.")

                        if edits_collected >= target_count:
                            break

    print(f"Collected {edits_collected} edits for annotation.")

def main(mode, input_file):
    directory_mapping = {
        'wiki': 'wiki_csv_sets',
        'similarity': 'similarity_score_sets',
        'rand': 'data_annotation_sets'
    }
    directory = directory_mapping.get(mode, 'wiki_csv_sets')  # Default to 'test_wiki_csv' if mode not found
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

        if mode == 'rand':
            generate_annotation_set(lines, directory, target_count=300)
        else:
            for line in lines:
                page_title = line.strip()
                print(f"Processing page: {page_title}")
                if mode == 'wiki':
                    handle_wiki_mode(page_title, directory)
                else:
                    handle_similarity_mode(page_title, directory)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process Wikipedia revision data")
    parser.add_argument('-m', '--mode', choices=['wiki', 'similarity', 'rand'], default='wiki',
                        help='Mode to run the script: "wiki" for generating wiki datasets, "similarity" for similarity scores, "rand" to generate random set for annotation. Default is "wiki".')
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Input file containing Wikipedia page titles.')
    args = parser.parse_args()
    main(args.mode, args.input_file)