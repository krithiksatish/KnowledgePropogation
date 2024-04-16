from datetime import datetime
import ipdb
from urllib.parse import urlparse, parse_qs
import bs4
import requests
import re
# import dataclasses
from nltk.tokenize import sent_tokenize
from typing import Dict, Union, List
from dataclasses import dataclass
from html2text import HTML2Text
import os
from .wiki_regexes import cut_list, cut_sup, cut_note, cut_table, cut_first_table,\
    cut_references2, cut_references, cut_references_es, cut_references_fr, cut_references_fr2, cut_references_ko, cut_references_ru,\
          double_paren, emphasis, bold, second_heading, third_heading, fourth_heading, second_heading_separation, fourth_heading2, third_heading2, all_spaces, \
            punctuations, link, link_number, paren, paren_fr, paren_zh2

from typing import List, Tuple  # for get_prev_after_text
import difflib  # for find_text_difference_and_context
import nltk
nltk.download('punkt')

re_category_list = re.compile(r'<link rel="mw:PageProp\/Category" href=".\/(Category:.*?)"')

text_maker = HTML2Text()
text_maker.ignore_links = True
text_maker.ignore_images = True
text_maker.ignore_tables = True
text_maker.ignore_emphasis = False

def get_lang(person_link):
    try:
        json = requests.get("https://en.wikipedia.org/api/rest_v1/page/metadata/"+person_link).json()
    except:
        return [('en', person_link)]
    if "language_links" in json:
        lang_list = json["language_links"]
        return [('en', person_link)]+[(l['lang'], l['titles']['canonical']) for l in lang_list]
    else:
        return [('en', person_link)]

def cut_ref(text, lang_wiki):
    if lang_wiki == "enwiki":
        return cut_references.sub(r"\1",text)
    elif lang_wiki == "eswiki":
        return cut_references_es.sub(r"\1",text)
    elif lang_wiki == "frwiki":
        text = cut_references_fr.sub(r"\1",text)
        return cut_references_fr2.sub(r"\1",text)
    elif lang_wiki == "kowiki":
        return cut_references_ko.sub(r"\1",text)
    elif lang_wiki == "ruwiki":
        return cut_references_ru.sub(r"\1",text)
    else:
        raise ValueError("unsupported lang: "+lang_wiki)

def get_category(person_id, lang_wiki): 
    """_summary_

    Args:
        person_id (str): Wikipedia page ID.

    Returns:
        List[str]: List of categories that the person belongs to.
    """
    if lang_wiki == "enwiki":
        txt = requests.get("https://en.wikipedia.org/api/rest_v1/page/html/"+person_id).text.replace("\n","")
        categories = re_category_list.findall(txt)
        return categories
    else:   
        return [] # NOTE: haven't implemented this for other languages yet.

def get_text(person_link, lang_wiki):
    # txt = requests.get("https://en.wikipedia.org/api/rest_v1/page/html/"+person).text.replace("\n","")
    txt = requests.get(person_link).text.replace("\n","")
    txt = clean_html(txt, lang_wiki)
    return txt

@dataclass
class Paragraph:
    clean_text: str

@dataclass
class Header:
    text: str
    level: int



def clean_paragraph(paragraph_elem: bs4.element.Tag) -> Paragraph:
    # use the html2text library to convert the html to text.
    paragraph = text_maker.handle(str(paragraph_elem))

    paragraph = re.sub(r"\n", " ", paragraph)
    # remove the reference links.
    paragraph = re.sub(r"\[\d+\]", "", paragraph)
    # remove text that is in parentheses.
    paragraph = re.sub(r'\([^)]*\)', "", paragraph) # TODO: this leaves a blank space, which is not ideal.
    # remove text that is in brackets.
    paragraph = re.sub(r'\[[^)]*\]', "", paragraph) # TODO: this might also leave a blank space.

    # remove the markdown formatting for bold and italics.
    paragraph = re.sub(r"\*\*", "", paragraph)
    paragraph = re.sub(r"_", "", paragraph)
    paragraph = paragraph.strip()
    return Paragraph(paragraph)

def retrieve_all_sentences(content_blocks: List[Union[Paragraph, Header]]) -> List[str]:
    all_sentences = []
    for paragraph in filter(lambda x: isinstance(x, Paragraph), content_blocks):
        paragraph_text = paragraph.clean_text
        sentences = sent_tokenize(paragraph_text)
        all_sentences.extend(sentences)
    return all_sentences

def clean_header(header_elem: bs4.element.Tag) -> Header:
    # get the level of the header.
    level = int(header_elem.name[1])
    # get the text of the header.
    header_text = header_elem.text
    return Header(header_text, level)
    
def remove_non_sentences(content_div: bs4.element.Tag, wiki_lang: str) -> bs4.element.Tag:
    hatnotes = content_div.find_all('div', class_='hatnote')
    for hatnote in hatnotes:
        hatnote.decompose()
    
    # remove mw-editsection
    edit_sections = content_div.find_all('span', class_='mw-editsection')
    for edit_section in edit_sections:
        edit_section.decompose()

    # remove <p> that have navbar in their class.
    navbars = content_div.find_all('p', class_='navbar')
    for navbar in navbars:
        navbar.decompose()

    # remove the info box if it exists.
    info_box = content_div.find('table', class_='infobox')
    if info_box:
        info_box.decompose()
    # remove all figures.
    figures = content_div.find_all('figure')
    for figure in figures:
        figure.decompose()
    
    # remove all the mw-empty-elt paragraphs
    empty_paragraphs = content_div.find_all('p', class_='mw-empty-elt')
    for empty_para in empty_paragraphs:
        empty_para.decompose()
    # remove all the tables.
    tables = content_div.find_all('table')
    for table in tables:
        table.decompose()
    # remove all the lists.
    lists = content_div.find_all('ul')
    for list_ in lists:
        list_.decompose()
    # remove all the images.
    images = content_div.find_all('img')
    for image in images:
        image.decompose()
    # remove all the audio files.
    audio_files = content_div.find_all('audio')
    for audio_file in audio_files:
        audio_file.decompose()
    # remove all the video files.
    video_files = content_div.find_all('video')
    for video_file in video_files:
        video_file.decompose()
    # remove all the references.
    references = content_div.find_all('div', class_='reflist')
    for reference in references:
        reference.decompose()

# TODO: fill this in
def _filter_empty_sections(important_content_elems: List[Union[Paragraph, Header]]) -> List[Union[Paragraph, Header]]:
    # filter out headers where they don't enclose any paragraphs.
    filtered_important_content_elems = []


def get_text(page_link, wiki_lang) -> List[Union[Paragraph, Header]]:
    #### step 1: requesting the html
    # get the html through a request.

    # do try/except 3 times.
    for _ in range(3):
        try:
            html = requests.get(page_link, timeout=10).text
            break
        except requests.exceptions.Timeout:
            print("timeout error")
            continue

    soup = bs4.BeautifulSoup(html, 'html.parser')
    # keep only the #mw-content-text div.
    content_div = soup.find('div', id='mw-content-text')

    ### Step 2 removing large swathes of the page
    remove_non_sentences(content_div, wiki_lang) # NOTE: warning, this modifies the content_div in place.

    # iterate over the children of the content div. 
    important_content_elems = []
    print("looking for p, h2, h3")
    for element in soup.find_all(lambda tag: tag.name in ['p', 'h2', 'h3']):
        if element.name == 'p':
            important_content_elems.append(clean_paragraph(element))
        elif element.name == 'h2' or element.name == 'h3':
            important_content_elems.append(clean_header(element))
    # TODO: add call to filter headers for empty sections.
    return important_content_elems

# TODO: there might be an issue here when getting the segmented text.
def get_headings(t):
    def _get_headings(heading_list, heading="## ",second=True):
        if len(heading_list)==1:
            if second:
                return [["summary",heading_list[0]]]
            else:
                return []
        res = []
        _head = None
        for i, h in enumerate(heading_list):
            if i == 0 and not h.startswith("#") and second:
                res.append(["summary",h])
                continue
            if h.startswith(heading):
                _head = h.replace("#","").strip().lower()
            else:
                if _head is None:
                    continue
                if not second:
                    h = fourth_heading2.sub(" ",h)
                    h = all_spaces.sub(" ",h)
                res.append([_head, h])
                _head=None
        return res
    segmented = {}
    seconds = re.split(r"\s(#{2}\s{1}.*?)\s{2}",t)
    segmented["second"]=_get_headings(seconds, "## ", True)
    segmented["third"]= [_get_headings(re.split(r"\s(#{3}\s{1}.*?)\s{2}",h_text), "### ", False) for h, h_text in segmented["second"]]
    for i, (h, h_text) in enumerate(segmented["second"]):
        h_text = fourth_heading2.sub("  ",h_text)
        h_text = third_heading2.sub("  ",h_text)
        h_text = all_spaces.sub(" ",h_text)
        segmented["second"][i][1] = h_text
    return segmented

def _verify_previous_revision_info(revision_info_div: bs4.element.Tag, lang: str):
    if lang == "enwiki":
        assert "old revision" in revision_info_div.find('p').text 
    elif lang == "frwiki":
        assert "version archivée" in revision_info_div.text
    elif lang == "eswiki":
        assert "versión antigua" in revision_info_div.text
    elif lang == "ruwiki":
        assert "старая версия" in revision_info_div.text
    elif lang == "kowiki":
        # assert "이전 버전" in revision_info_div.text, ipdb.set_trace()
        pass
    else:
        raise ValueError(f"Language {lang} not supported.")

def cut_paren(text, lang_wiki):
    if lang_wiki == "enwiki":
        return paren.sub("",text)
    elif lang_wiki == "eswiki":
        return paren.sub("",text)
    elif lang_wiki == "frwiki":
        text = paren.sub("",text)
        return paren_fr.sub(r"\1",text)
    elif lang_wiki == "zh":
        text = paren.sub("",text)
        # text = paren_zh.sub("",text)
        return paren_zh2.sub(r"\1",text)
    elif lang_wiki == "kowiki":
        return paren.sub("",text)
    elif lang_wiki == "ruwiki":
        text = paren.sub("",text)
        return paren_fr.sub(r"\1",text)
    else:
        raise ValueError("unsupported lang: "+lang_wiki)
    
def _verify_current_revision_info(revision_info_div: bs4.element.Tag, lang: str):
    # TODO: ask Chan if we really need these.
    if lang == "enwiki":
        assert "current revision" in revision_info_div.find('p').text
    elif lang == "frwiki":
        assert "version actuelle" in revision_info_div.text
    elif lang == "eswiki":
        assert "versión actual" in revision_info_div.text
    elif lang == "ruwiki":
        assert "текущая версия" in revision_info_div.text
    elif lang == "kowiki":
        # assert "현재 버전" in revision_info_div.text, ipdb.set_trace()
        # NOTE: commented out, maybe it's needed 
        pass

    else:
        raise ValueError(f"Language {lang} not supported.")

def clean_html(text, lang_wiki: str):
    # if lang_wiki != "enwiki":
    #     ipdb.set_trace()
    text_soup = bs4.BeautifulSoup(text, 'html.parser')
    # remove the div with mw-content-subtitle element.
    content_revision_info_previous = text_soup.find('div', id='mw-revision-info')
    content_revision_info_current = text_soup.find('div', id='mw-revision-info-current')
    if content_revision_info_previous: 
        # assert that the content revision info has a <p> tag containing the text "old revision"
        _verify_previous_revision_info(content_revision_info_previous, lang_wiki)
        content_revision_info_previous.decompose()
    if content_revision_info_current:
        # assert "current revision" in content_revision_info_current.find('p').text
        _verify_current_revision_info(content_revision_info_current, lang_wiki)
        content_revision_info_current.decompose()
    text = str(text_soup)
    t = cut_table.sub("  ", text)
    t = cut_list.sub(" ",t)
    t = cut_sup.sub("",t)
    t = cut_note.sub("",t)
    t = text_maker.handle(t).replace("\n"," ")
    t = cut_first_table.sub(r"\2",t)
    t = cut_ref(t, lang_wiki)
    t = cut_references2.sub(r"\1",t)
    t = double_paren.sub("",t)
    t = link.sub(r"\1",t)
    t = link_number.sub("",t)
    t = emphasis.sub(r"\1",t)
    t = cut_paren(t, lang_wiki)
    t = bold.sub(r"\1",t)
    t = t.replace(" > "," ")
    if lang_wiki == "enwiki":
        t = t.replace("\'t", " \'t")
        t = t.replace("\'s", " \'s")
        t = t.replace("\'ve", " \'ve")
        t = t.replace("\'m", " \'m")
        t = t.replace("\'re", " \'re")
    
    segmented = get_headings(t)
    second_headings_separated = second_heading_separation.findall(t)
    second_headings = [h[0] for h in second_headings_separated]
    fourth_headings = fourth_heading.findall(t)
    t = fourth_heading.sub("  ",t)
    third_headings = third_heading.findall(t)
    t = third_heading.sub("  ",t)
    second_headings = second_heading.findall(t)
    t = second_heading.sub("  ",t)
    # t = punctuations.sub(r" \1 ",t)
    t = all_spaces.sub(" ",t)
    return t, (second_headings, third_headings, fourth_headings), segmented

re_he = re.compile("[^a-zA-Z0-9]+he ")
re_she = re.compile("[^a-zA-Z0-9]+she ")
re_his = re.compile("[^a-zA-Z0-9]+his ")
re_her = re.compile("[^a-zA-Z0-9]+her ")
re_him = re.compile("[^a-zA-Z0-9]+him ")

def get_gender_with_text(txt):
    txt = txt.clean_text if isinstance(txt, Paragraph) else txt.text if isinstance(txt, Header) else "Unsupported object type"
#     txt = txt.lower()
    freq_he = len(re_he.findall(txt))+len(re_his.findall(txt))+len(re_him.findall(txt))
    freq_she = len(re_she.findall(txt))+len(re_her.findall(txt))
    gender = None
    if freq_she > freq_he:
        gender = "F"
    elif freq_he > freq_she:
        gender = "M"
    return gender

def get_info(wiki_link: str, lang_wiki: str):
    """Gets the information from every section of a Wikipedia page.

    Args:
        wiki_link (str): Link to the person's wikipedia page and the person's name.
        lang_wiki (str): 

    Returns:
        Tuple[str, Dict[str, Any]]: _description_
    """
    wiki_link = wiki_link.replace("/wiki/","")
    # link is of the form 'https://en.wikipedia.org/w/index.php?title=Scottie_Barnes&oldid=910610546'
    # get the title 
    person_id = parse_qs(urlparse(wiki_link).query)['title'][0]

    person_info = {}
    person_info["langs"] = get_lang(wiki_link) # TODO: needs to be replaced.
    person_info["categories"] = get_category(person_id, lang_wiki) # 
    txt, section_names, section_text = get_text(wiki_link, lang_wiki)
    person_info["gender"] = get_gender_with_text(txt) # count of gendered pronouns 
    person_info["text"] = txt # unsplit text
    person_info["section-names"] = section_names # lengths of sections should be the same as section-text.
    person_info["section-text"] = section_text
    return wiki_link, person_info


# returns a tuple containing a list of sentence pairs, with each pair containing a text block  given the url of a wiki-diff page
def get_del_add_text(diff_page_url: str) -> List[Tuple[str, str]]:
    
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

        # add aggregated text from both before and after text to tuple
        if deletion_block:
            deletion_text, del_refs = get_aggregated_text(deletion_block)
        if addition_block:
            addition_text, add_refs = get_aggregated_text(addition_block)

        # if theres no deletion text and no addition text, skip this block
        if not deletion_text and not addition_text:
            continue

        # find differences and their contexts for each pair of deletion and addition texts
        context_pairs = find_differences_and_context(addition_text, deletion_text)
        context_pairs_list.extend(context_pairs)

    return context_pairs_list

# takes in an addition or deletion block of html and returns both the aggregated text and a set of references
def get_aggregated_text(block: bs4.element.Tag) -> Tuple[str, set]:
    total_block_text = ""
    refs_set = set()
    
    for child in block.children:
        total_block_text += child.text

    # remove refs from text and add to set
    ref_regex = r'(<ref.*?>.*?</ref>|<ref.*?/>|\*\s*\{\{cite.*?\}\})'
    
    while True:
        match = re.search(ref_regex, total_block_text, flags=re.DOTALL)
        if not match:
            break
        start, end = match.span()
        refs_set.add(total_block_text[start:end])
        total_block_text = total_block_text[:start] + total_block_text[end:]

    # if first character of block is | or =, return None, None
    if total_block_text and (total_block_text[0] in ['|', '='] or total_block_text.startswith('[[Category:') or total_block_text.startswith('{{')):
        return None, None
    
    return total_block_text, refs_set

# takes in the text block containing changes, the original text block, and the 
# start and end indices of the changes, and returns just the portion of the text 
# block that contains the changes and its corresponding portion of the original text block
# need import difflib
def find_differences_and_context(changed_text, og_text, context_sentences=1) -> List[Tuple[str, str]]:
    if changed_text is None:
        changed_text = ''
    
    if og_text is None:
        og_text = ''

    # Tokenize the texts into sentences
    changed_sentences = sent_tokenize(changed_text)
    og_sentences = sent_tokenize(og_text)

    # Use difflib to find the differences
    diff = list(difflib.ndiff(og_sentences, changed_sentences))

    # Prepare a list to store the context of changes
    changes_with_context = []

    # Iterate through the diff output to find indices of changes
    for i, s in enumerate(diff):
        if s.startswith('- ') or s.startswith('+ '):  # Change found
            # Determine context range
            start_context = max(0, i - context_sentences)
            end_context = min(len(diff), i + context_sentences + 1)

            # Extract the context range
            context = diff[start_context:end_context]

            # Append the context of this change to the list
            changes_with_context.append(context)

    # Now we will merge contexts if they are close to each other
    merged_contexts = []
    if changes_with_context:  # Check if the list is not empty
        current_merge = changes_with_context[0]

        for context in changes_with_context[1:]:
            # If the next context starts within the current merge's range, merge them
            if diff.index(context[0]) <= diff.index(current_merge[-1]):
                current_merge.extend(context)
                current_merge = list(dict.fromkeys(current_merge))  # Remove duplicates while preserving order
            else:
                merged_contexts.append(current_merge)
                current_merge = context

        # Don't forget to add the last merge
        if current_merge not in merged_contexts:
            merged_contexts.append(current_merge)

    # Convert the diff markers back to normal text and differentiate between original and changed contexts
    cleaned_contexts = []
    for context in merged_contexts:
        og_context = ' '.join([s[2:] for s in context if s.startswith('- ')])
        ch_context = ' '.join([s[2:] for s in context if s.startswith('+ ')])
        cleaned_contexts.append((og_context, ch_context))  # Create a tuple

    return cleaned_contexts

# takes in a wikipedia page title and returns a list of urls to the page's revisions at pseudo-monthly intervals
# usage: get_monthly_revision_comparison_links("Attention_deficit_hyperactivity_disorder")
def get_monthly_revision_comparison_links(page_title: str) -> List[str]:
    # Start a requests session for HTTP requests
    S = requests.Session()
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"

    # Parameters for the initial API request to get revisions
    PARAMS = {
        "action": "query",                   # Action is 'query' to retrieve data.
        "prop": "revisions",                 # We want to get revisions of the page.
        "titles": page_title,                # Page title to get revisions for.
        "rvlimit": "max",                    # Request as many revisions as allowed.
        "rvdir": "newer",                    # Order revisions from oldest to newest.
        "rvprop": "ids|timestamp",           # Get the revision IDs and timestamps.
        "formatversion": "2",                # Use the modern format for the response.
        "format": "json"                     # Response in JSON format.
    }

    # Initialize list to store the comparison URLs
    comparison_urls = []
    # Dictionary to store the first revision ID of each month
    monthly_revisions = {}
    # Flag to indicate if we are done fetching revisions
    done = False

    while not done:
        # Make the API request
        response = S.get(url=URL, params=PARAMS)
        data = response.json()

        # Parse the revisions from the API response
        revisions = data['query']['pages'][0]['revisions']
        for rev in revisions:
            rev_id = rev['revid']  # Revision ID
            timestamp = rev['timestamp']  # Timestamp of the revision
            # Convert timestamp to datetime object
            date = datetime.fromisoformat(timestamp.rstrip('Z'))
            # Create a key for the month
            month_key = f"{date.year}-{date.month:02d}"
            # Store the first revision ID encountered for each month
            if month_key not in monthly_revisions:
                monthly_revisions[month_key] = rev_id

        # Check if there's more data to fetch
        if 'continue' in data:
            # Set the rvcontinue parameter to the continuation value from the response
            PARAMS['rvcontinue'] = data['continue']['rvcontinue']
        else:
            # No more revisions to fetch, so we're done
            done = True

    # The previous revision ID, used to generate comparison URLs
    prev_rev_id = None
    # Generate comparison URLs for the stored monthly revisions
    for month_key in sorted(monthly_revisions.keys()):
        rev_id = monthly_revisions[month_key]
        if prev_rev_id:
            # Create the comparison URL using the current and previous revision IDs
            comparison_url = f"https://en.wikipedia.org/w/index.php?title={page_title.replace(' ', '_')}&diff={rev_id}&oldid={prev_rev_id}"
            comparison_urls.append(comparison_url)
        # Update the previous revision ID for the next loop iteration
        prev_rev_id = rev_id

    # Return the list of comparison URLs
    return comparison_urls

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
    