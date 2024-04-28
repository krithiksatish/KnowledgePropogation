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

# given the url of a wiki-diff page, returns a tuple containing a list of sentence pairs 
# with each pair containing a text block, and boolean indicating whether the text contains real content
def get_del_add_text(diff_page_url: str) -> List[Tuple[str, str, str, str, bool]]:
    
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

# takes in the text block containing changes, the original text block, and the 
# start and end indices of the changes, and returns just the portion of the text 
# block that contains the changes and its corresponding portion of the original text block
# need import difflib
# takes in the text block containing changes, the original text block, and the 
# start and end indices of the changes, and returns just the portion of the text 
# block that contains the changes and its corresponding portion of the original text block
# need import difflib
def find_differences_and_context(changed_text, og_text, references, context_sentences=1) -> List[Tuple[str, str, str, str, set, bool]]:
    if not changed_text:
        changed_text = ''
    if not og_text:
        og_text = ''

    changed_sentences = sent_tokenize(changed_text)
    og_sentences = sent_tokenize(og_text)
    diff = list(difflib.ndiff(og_sentences, changed_sentences))
    changes_with_context = []

    # Identifying changes and their context
    for i, s in enumerate(diff):
        if s.startswith('- ') or s.startswith('+ '):
            start_context = max(0, i - context_sentences)
            end_context = min(len(diff), i + 1)
            context = diff[start_context:end_context]
            changes_with_context.append(context)

    # Merging close contexts and processing them
    merged_contexts = []
    if changes_with_context:
        current_merge = changes_with_context[0]
        for context in changes_with_context[1:]:
            if diff.index(context[0]) <= diff.index(current_merge[-1]):
                current_merge.extend(context)
                current_merge = list(dict.fromkeys(current_merge))
            else:
                merged_contexts.append(current_merge)
                current_merge = context
        merged_contexts.append(current_merge)  # Add the last context

    cleaned_contexts = []
    for context in merged_contexts:
        og_context = ' '.join([s[2:] for s in context if s.startswith('- ')])
        ch_context = ' '.join([s[2:] for s in context if s.startswith('+ ')])
        
        # Split contexts into smaller segments if needed
        if len(og_context) > 250 or len(ch_context) > 250:
            smaller_contexts = split_into_smaller_contexts(og_context, ch_context)
            for small_og, small_ch in smaller_contexts:
                process_and_append_context(small_og, small_ch, references, cleaned_contexts)
        else:
            process_and_append_context(og_context, ch_context, references, cleaned_contexts)
    
    return cleaned_contexts

def process_and_append_context(og_context, ch_context, references, cleaned_contexts) -> None:
    contains_content = filter_non_content_edits(og_context, ch_context)
    cleaned_og_context = clean_wiki_text(og_context) if contains_content else ""
    cleaned_ch_context = clean_wiki_text(ch_context) if contains_content else ""
    cleaned_contexts.append((og_context, ch_context, cleaned_og_context, cleaned_ch_context, references, contains_content))

def split_into_smaller_contexts(og_context, ch_context, max_length=250) -> List[Tuple[str, str]]:
    og_sentences = sent_tokenize(og_context)
    ch_sentences = sent_tokenize(ch_context)
    smaller_contexts = []
    
    current_og, current_ch = "", ""
    current_length = 0
    
    for og_sent, ch_sent in zip(og_sentences, ch_sentences):
        # Check the length with the new sentence added
        if current_length + len(og_sent) + len(ch_sent) <= max_length:
            current_og += (og_sent + " ")
            current_ch += (ch_sent + " ")
            current_length += len(og_sent) + len(ch_sent)
        else:
            if current_og and current_ch:  # Ensure not to add empty strings
                smaller_contexts.append((current_og.strip(), current_ch.strip()))
            current_og, current_ch = og_sent + " ", ch_sent + " "
            current_length = len(og_sent) + len(ch_sent)
    
    # Add the last segment if it has content
    if current_og and current_ch:
        smaller_contexts.append((current_og.strip(), current_ch.strip()))
    
    return smaller_contexts

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
def filter_non_content_edits(old_text: str, new_text: str) -> bool:
    contains_content = True
    cutout_set = ('|', '{|', '=', '[[Category:', '[[File:', '{{', '#REDIRECT', '#redirect')

    # Strip leading asterisks and other whitespace characters from the new text
    strip_new_text = re.sub(r'^\s*\*+\s*', '', new_text.strip(), flags=re.MULTILINE)

    # Patterns to check for non-content edits
    non_content_patterns = [
        r'^<math>.*</math>$',         # Entire content within <math> tags
        r'^<[^>]+>$',                 # Entire content within any other single HTML tag
    ]

    # Check if the new text starts with any non-content leading characters or matches any of the specified patterns
    if strip_new_text.startswith(cutout_set) or any(re.match(pattern, strip_new_text) for pattern in non_content_patterns):
        contains_content = False

    # Apply the same checks to the old text if the new text is empty
    if not new_text:
        strip_old_text = old_text.strip()
        if strip_old_text.startswith(cutout_set) or any(re.match(pattern, strip_old_text) for pattern in non_content_patterns):
            contains_content = False

    return contains_content

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
    