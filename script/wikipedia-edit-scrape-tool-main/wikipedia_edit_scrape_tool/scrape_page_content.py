from datetime import datetime
import bs4
import requests
import re
# import dataclasses
from nltk.tokenize import sent_tokenize
from typing import Dict, Union, List
import uuid

from typing import List, Tuple  # for get_prev_after_text
import difflib  # for find_text_difference_and_context
from sklearn.feature_extraction.text import TfidfVectorizer  # for find_closest_sentences
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

# given the url of a wiki-diff page, returns a tuple containing a list of sentence pairs 
# with each pair containing a text block, and boolean indicating whether the text contains real content
def get_del_add_text(diff_page_url: str) -> List[Tuple[str, str, str, str, set, bool, bool]]:
    
    # do try/except 3 times.
    for _ in range(3):
        try:
            html = requests.get(diff_page_url, timeout=10).text
            break
        except requests.exceptions.Timeout:
            print("timeout error")
            continue

    soup = bs4.BeautifulSoup(html, 'html.parser')
    tr_blocks = soup.find_all('tr')
    context_pairs_list = []

    # loop through tr blocks
    for tr_block in tr_blocks:
        moved_edits = False
        # Check for the moved paragraph left anchor (case where paragraph was edited but moved)
        move_left_anchor = tr_block.find('a', class_='mw-diff-movedpara-left')
        if move_left_anchor:
            # Find the parent 'td' element of the move_left_anchor which has class "diff-marker"
            diff_marker_td = move_left_anchor.find_parent('td', class_='diff-marker')
            # Now use find_next_sibling to find the 'td' with class "diff-deletedline diff-side-deleted"
            deletion_block = diff_marker_td.find_next_sibling('td', class_='diff-deletedline diff-side-deleted')

            move_right_name = move_left_anchor['href'].lstrip('#')
            move_right_anchor = soup.find('a', {'name': move_right_name})
            if move_right_anchor:
                # Find the parent 'td' element with class "diff-addedline diff-side-added"
                addition_block = move_right_anchor.find_parent('td', class_='diff-addedline diff-side-added')
                moved_edits = True

        # if tr block has a td element child with class "diff-marker"
        # and without a data-marker attribute, and no moved edits, this means 
        # it's simply shifted text. skip this block
        if (tr_block.find('td', class_='diff-marker') 
                and not tr_block.find('td', class_='diff-marker').get('data-marker') 
                and not moved_edits):
            continue
        
        # get the td elements within the tr block with class "diff-deletedline" or "diff-addedline"
        if not moved_edits:
            deletion_block = tr_block.find('td', class_='diff-deletedline')
            addition_block = tr_block.find('td', class_='diff-addedline')
        
        addition_text = ""
        deletion_text = ""
        add_refs = set()

        # add aggregated text from both before and after text to tuple
        if deletion_block:
            deletion_text, _ = get_aggregated_text(deletion_block) # we don't care about deleted references
        if addition_block:
            addition_text, add_refs = get_aggregated_text(addition_block)

        # if theres no deletion text and no addition text, skip this block
        if not deletion_text and not addition_text:
            continue

        # find differences and their contexts for each pair of deletion and addition texts
        context_pairs = find_differences_and_context(addition_text, deletion_text, add_refs)
        context_pairs_list.extend(context_pairs)

    return context_pairs_list

# takes in an addition or deletion block of html and returns both the aggregated text and a set of references
# returns tuple: [0] aggregated text, [1] set of references, [2] whether the block contained real text content
def get_aggregated_text(block: bs4.element.Tag) -> Tuple[str, set]:
    total_block_text = ""
    refs_set = set()

    # Aggregating all text content from the block's children
    for child in block.children:
        total_block_text += child.text

    # Regex patterns to find references and URLs (including optional square brackets)
    ref_regex = r'(<ref.*?>.*?</ref>|<ref.*?/>|\*\s*\{\{cite.*?\}\}|{{Citation.*?}}</ref>)'
    url_regex = r'\[?(https?://\S+)\]?'

    # Extract references
    while True:
        match = re.search(ref_regex, total_block_text, flags=re.DOTALL)
        if not match:
            break
        start, end = match.span()
        refs_set.add(total_block_text[start:end])
        total_block_text = total_block_text[:start] + total_block_text[end:]

    # Extract URLs, handling optional square brackets
    while True:
        url_match = re.search(url_regex, total_block_text)
        if not url_match:
            break
        url_start, url_end = url_match.span()
        url_text = url_match.group(1)  # Group 1 is the actual URL without square brackets
        refs_set.add(url_text)
        total_block_text = total_block_text[:url_start] + total_block_text[url_end:]

    return total_block_text, refs_set

# custom preprocessor to clean text before TF-IDF vectorization
def custom_preprocessor(text):
    # Replace special characters with space
    text = re.sub(r'[{}[\]()<>]', ' ', text)
    text = re.sub(r'[!?,;:.]', ' ', text)
    text = re.sub(r'[\'"@#\$%\^&\*\_\+=]', ' ', text)
    text = re.sub(r'[\\|/]', ' ', text)
    text = re.sub(r'[~-]', ' ', text)
    return text.strip()  # Strip leading and trailing spaces

def find_closest_sentences(sentences_before, sentences_after, threshold=0.4):
    # Handle cases where either sentences_before or sentences_after is empty
    if not sentences_before and not sentences_after:
        return [], [], []
    if not sentences_before:
        return [], [], list(range(len(sentences_after)))
    if not sentences_after:
        return [], list(range(len(sentences_before))), []
    
    sentences_before = [custom_preprocessor(sent) for sent in sentences_before]
    sentences_after = [custom_preprocessor(sent) for sent in sentences_after]

    if not any(sentences_before) and not any(sentences_after):
        return [([i for i in range(len(sentences_before))], [j for j in range(len(sentences_after))])], [], []
    if not any(sentences_before):
        return [], [], list(range(len(sentences_after)))
    if not any(sentences_after):
        return [], list(range(len(sentences_before))), []
    
    # Direct mapping if both lists have only one sentence each
    if len(sentences_before) == 1 and len(sentences_after) == 1:
        return [([0], [0])], [], []

    print ("sentences_before: ", sentences_before)
    print ("sentences_after: ", sentences_after)
    
    # Replace empty sentences with a placeholder to avoid errors in vectorization
    rand_id = str(uuid.uuid4())
    sentences_before = [sent if sent else "EMPTY" + rand_id for sent in sentences_before]
    sentences_after = [sent if sent else "EMPTY" + rand_id for sent in sentences_after]

    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer().fit(sentences_before + sentences_after)
    vectors_before = vectorizer.transform(sentences_before)
    vectors_after = vectorizer.transform(sentences_after)

    # Compute cosine similarity between each pair of sentences
    similarity_matrix = cosine_similarity(vectors_before, vectors_after)

    # Track the best matches from both before to after and after to before
    before_to_after = {}
    after_to_before = {}

    # Find the best matches for each sentence in 'before'
    for i, row in enumerate(similarity_matrix):
        best_match_index = row.argmax()
        best_match_score = row[best_match_index]
        if best_match_score > threshold:
            if best_match_index not in before_to_after:
                before_to_after[best_match_index] = []
            before_to_after[best_match_index].append(i)

    # Find the best matches for each sentence in 'after'
    for j, col in enumerate(similarity_matrix.T):
        best_match_index = col.argmax()
        best_match_score = col[best_match_index]
        if best_match_score > threshold:
            if best_match_index not in after_to_before:
                after_to_before[best_match_index] = []
            after_to_before[best_match_index].append(j)

    # Combine both mappings to ensure all sentences are considered
    combined_mappings = {}
    for j, i_list in before_to_after.items():
        combined_mappings[j] = combined_mappings.get(j, set()).union(i_list)
    for i, j_list in after_to_before.items():
        for j in j_list:
            combined_mappings[j] = combined_mappings.get(j, set()).union([i])

    # Merge mappings where multiple 'after' sentences map to the same 'before' sentence and vice versa
    final_mappings = {}
    for after, befores in combined_mappings.items():
        for before in befores:
            if before in final_mappings:
                final_mappings[before][1].add(after)
            else:
                final_mappings[before] = (set([before]), set([after]))

    # Further merge final_mappings if any before indices are shared
    merged_final_mappings = []
    for before, (befores, afters) in final_mappings.items():
        found = False
        for i, (existing_befores, existing_afters) in enumerate(merged_final_mappings):
            if not existing_befores.isdisjoint(befores) or not existing_afters.isdisjoint(afters):
                merged_final_mappings[i] = (existing_befores.union(befores), existing_afters.union(afters))
                found = True
                break
        if not found:
            merged_final_mappings.append((befores, afters))

    # Convert merged_final_mappings from sets to lists and combine consecutive afters
    combined_final_mappings = [(sorted(befores), sorted(afters)) for befores, afters in merged_final_mappings]

    # Add unmatched sentences
    used_before = set()
    used_after = set()
    for befores, afters in combined_final_mappings:
        used_before.update(befores)
        used_after.update(afters)

    unmatched_before = [i for i in range(len(sentences_before)) if i not in used_before]
    unmatched_after = [j for j in range(len(sentences_after)) if j not in used_after]

    return combined_final_mappings, unmatched_before, unmatched_after

# takes in the text block containing changes, the original text block, and the 
# start and end indices of the changes, and returns just the portion of the text 
# block that contains the changes and its corresponding portion of the original text block
# uses TF-IDF vectorization and cosine similarity
# returns a tuple containing:
# [0] the unmodified before text block, [1] the unmodified after text block, [2] the modified before text block,
# [3] the modified after text block, [4] a set of references corresponding to the text block,
# [5] whether the unmodified before text block contained real content, [6] whether the unmodified after text block contained real content
def find_differences_and_context(changed_text, og_text, references) -> List[Tuple[str, str, str, str, set, bool, bool]]:
    # Tokenize texts into sentences
    sentences_before = nltk.sent_tokenize(og_text)
    sentences_after = nltk.sent_tokenize(changed_text)

    # Get mappings
    mappings, unmatched_before, unmatched_after = find_closest_sentences(sentences_before, sentences_after)

    # Identify modified sentences
    modified = []

    for befores, afters in mappings:
        before_text = ' '.join([sentences_before[i] for i in befores])
        after_text = ' '.join([sentences_after[j] for j in afters])
        if before_text != after_text:
            modified.append((before_text, after_text))

    # Handle unmatched sentences as modified with empty counterparts
    for i in unmatched_before:
        modified.append((sentences_before[i], ""))
    for j in unmatched_after:
        modified.append(("", sentences_after[j]))

    # Process modified sentences
    cleaned_contexts = []
    for before_text, after_text in modified:
        process_and_append_context(before_text, after_text, references, cleaned_contexts)

    return cleaned_contexts

# helper function to process contexts and append them to cleaned_contexts list
def process_and_append_context(og_context, ch_context, references, cleaned_contexts) -> None:
    old_contains_content, new_contains_content = filter_non_content_edits(og_context, ch_context)
    cleaned_og_context = clean_wiki_text(og_context) if old_contains_content else ""
    cleaned_ch_context = clean_wiki_text(ch_context) if new_contains_content else ""
    cleaned_contexts.append((og_context, ch_context, cleaned_og_context, cleaned_ch_context, references, 
                             old_contains_content, new_contains_content))

# takes in a text block and returns the cleaned text
def clean_wiki_text(text: str) -> str:
    text = text.strip()

    # Remove wiki markup
    text = re.sub(r"''+", "", text)  # Remove italics and bold

    # Update the pattern to keep the text after the '|', if it exists, otherwise keep the text before the '|'
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)  # Modify internal links

    text = re.sub(r"\[([^\]]+)\]", r"\1", text)  # Remove single brackets

    # Remove complex templates and any nested templates
    # Continue removing nested templates until all are gone
    old_text = None
    while old_text != text:
        old_text = text
        text = re.sub(r"\{\{[^{}]*(?:\{\{[^{}]*\}\}[^{}]*)*\}\}", "", text)
    
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text)  # Remove references
    text = re.sub(r"<[^>]+>", "", text)  # Remove other HTML tags
    text = re.sub(r"\s+", " ", text)  # Normalize whitespace

    # Strip leading asterisks, hashes, and whitespace globally, including any space right after the asterisk
    text = re.sub(r'^\s*[\*#]+\s*', '', text, flags=re.MULTILINE)

    # Remove any stray closing curly braces, "|}", and stray brackets in one go
    text = re.sub(r"\}\}+|\|\}|[\[\]#]", "", text)

    return text.strip()  # Strip trailing and leading whitespace globally

# filter out non-content edits based on the context of the changes (eg. presence of certain characters
# that aren't related to actual content changes)
# return two boolean values: 
#   [0] whether old text contains actual textual content, [1] whether new text contains actual textual content
def filter_non_content_edits(old_text: str, new_text: str) -> Tuple[bool, bool]:
    old_contains_content = True
    new_contains_content = True
    cutout_set = ('|', '{|', '[[Category:', '[[Image:' '[[File:', '{{', '#REDIRECT', '#redirect', "==")

    # Strip leading asterisks and other whitespace characters from the new text
    strip_new_text = re.sub(r'^\s*\*+\s*', '', new_text.strip(), flags=re.MULTILINE)
    # Strip leading asterisks and other whitespace characters from the old text
    strip_old_text = re.sub(r'^\s*\*+\s*', '', old_text.strip(), flags=re.MULTILINE)

    # Patterns to check for non-content edits
    non_content_patterns = [
        r'^<math>.*</math>$',         # Entire content within <math> tags
        r'^<[^>]+>$',                 # Entire content within any other single HTML tag
    ]

    # Check if the new text starts with any non-content leading characters or matches any of the specified patterns
    if strip_new_text.startswith(cutout_set) or any(re.match(pattern, strip_new_text) for pattern in non_content_patterns):
        new_contains_content = False
    
    # Check if the old text contains any non-content patterns
    if strip_old_text.startswith(cutout_set) or any(re.match(pattern, strip_old_text) for pattern in non_content_patterns):
        old_contains_content = False

    return old_contains_content, new_contains_content

# takes in two texts and returns the differences between old and new text
def get_text_differences(old_text, new_text) -> str:
    # Tokenize the texts
    old_words = old_text.split()
    new_words = new_text.split()
    
    # Get the difference iterator
    diff = difflib.ndiff(old_words, new_words)
    
    # Extract additions
    added_words = [word[2:] for word in diff if word.startswith('+ ')]
    
    # Join words to form the resulting string
    result = ' '.join(added_words)
    return result

# takes in a wikipedia page title and returns a list of urls to the page's revisions at 
# pseudo-monthly intervals, as well as the metadata for each revision
# usage: get_monthly_revision_comparison_links("Attention_deficit_hyperactivity_disorder")
def get_monthly_revision_comparison_links(page_title: str) -> List[Tuple[str, Dict]]:
    # Start a requests session for HTTP requests
    S = requests.Session()
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the initial API request to get revisions
    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "titles": page_title,
        "rvlimit": "max",
        "rvdir": "newer",
        "rvprop": "ids|timestamp|user|userid",
        "formatversion": "2",
        "format": "json"
    }

    # Dictionary to store metadata for each revision by ID
    revision_metadata = {}
    # Dictionary to store the first revision ID of each month
    monthly_revisions = {}
    # Flag to indicate if we are done fetching revisions
    done = False

    while not done:
        # Make the API request
        response = S.get(url=URL, params=PARAMS)
        data = response.json()

        # Parse the revisions from the API response
        revisions = data['query']['pages'][0].get('revisions', [])
        for rev in revisions:
            rev_id = rev['revid']
            timestamp = rev['timestamp']
            if 'user' in rev:
                user = rev['user']
                userid = rev['userid']
            date = datetime.fromisoformat(timestamp.rstrip('Z'))
            month_key = f"{date.year}-{date.month:02d}"
            if month_key not in monthly_revisions:
                monthly_revisions[month_key] = rev_id
            revision_metadata[rev_id] = {
                "timestamp": timestamp,
                "user": user,
                "userid": userid
            }

        # Check if there's more data to fetch
        if 'continue' in data:
            PARAMS['rvcontinue'] = data['continue']['rvcontinue']
        else:
            done = True

    # Generate comparison URLs for the stored monthly revisions
    prev_rev_id = None
    result = []
    for month_key in sorted(monthly_revisions.keys()):
        rev_id = monthly_revisions[month_key]
        if prev_rev_id:
            comparison_url = f"https://en.wikipedia.org/w/index.php?title={page_title.replace(' ', '_')}&diff={rev_id}&oldid={prev_rev_id}"
            metadata = revision_metadata.get(rev_id, {})
            # Extract the editor and edit time from the metadata, and default to "Unknown" if not found
            result.append((comparison_url, {
                "editor": metadata.get('user', 'Unknown'),
                "edit_time": metadata.get('timestamp', 'Unknown'),
                "editor_id": metadata.get('userid', 'Unknown')
            }))
        prev_rev_id = rev_id

    return result

# takes in wikipedia catagory name and returns list of all article titles extracted from the page
# usage: get_article_titles_from_category("Mental_health")
def get_article_titles_from_category(category_name) -> List[str]:
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the API request
    PARAMS = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmlimit": "max",
        "format": "json"
    }

    # List to store article titles
    article_titles = []

    # Flag to indicate if more data needs to be fetched
    done = False

    while not done:
        response = S.get(url=URL, params=PARAMS)
        data = response.json()

        # Extract article titles from the response
        for member in data['query']['categorymembers']:
            if member['ns'] == 0:  # ns == 0 indicates an article page
                article_titles.append(member['title'])

        # Check if there's more data to fetch
        if 'continue' in data:
            PARAMS['cmcontinue'] = data['continue']['cmcontinue']
        else:
            done = True

    return article_titles

# get necessary page ids from footer of a wiki page (eg. related wiki articles to mental health)
def get_link_ids_from_footer(wiki_link: str):
    for _ in range(3):
        try:
            html = requests.get(wiki_link, timeout=10).text
            break
        except requests.exceptions.Timeout:
            print("Timeout error on attempt", _ + 1)
            continue
    else:  # If we never broke out of the loop, return an empty list
        print("Failed to retrieve content after 3 attempts")
        return []

    # Parse the HTML content with BeautifulSoup
    soup = bs4.BeautifulSoup(html, 'html.parser')

    # Find all the td elements with the specific classes
    td_elements = soup.find_all('td', class_=['navbox-list-with-group', 'navbox-list'])

    # Open a file named 'test.txt' in write mode
    with open('wiki_page_ids.txt', 'w') as file:
        # Find all the td elements with the specific classes
        td_elements = soup.find_all('td', class_=['navbox-list-with-group', 'navbox-list'])

        # Loop through each td element
        for td in td_elements:
            # Find all ul elements within the td element
            ul_elements = td.find_all('ul')
            for ul in ul_elements:
                # For each ul element, find all a elements and get their href attributes if they exist
                for a in ul.find_all('a'):
                    href = a.get('href')  # Use the .get() method to avoid KeyError

                    # Make sure the href attribute is not None and starts with '/wiki/' and does not contain a colon
                    # This is to filter out non-article links and links to other language versions of the page
                    # eg. /wiki/Template_talk:Mental_disorders
                    if href and href.startswith('/wiki/') and ':' not in href:
                        # Write each link to the file, followed by a newline
                        # Strip the leading '/wiki/' from the href before writing
                        file.write(href[6:] + '\n')
    