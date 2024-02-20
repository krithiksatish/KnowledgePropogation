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


def get_diff_text(diff_page_url: str, lang_wiki: str) -> List[str]:
    # Goal of function: given a url of the diff page, return a set of strings (ideally sentences)
    # that contains all the yellow highlights (the added sentences)

    #### step 1: requesting the html
    # get the html through a request.

    # do try/except 3 times.
    for _ in range(3):
        try:
            html = requests.get(diff_page_url, timeout=10).text
            break
        except requests.exceptions.Timeout:
            print("timeout error")
            continue
    
    # TODO: 
    #   We should consider abbreviation edge cases during sentence extraction

    sentence_set = []
    refs_set = set()
    soup = bs4.BeautifulSoup(html, 'html.parser')
    text_addition_blocks = soup.find_all('td', class_='diff-addedline diff-side-added')

    # Loop through each block
    for addition_block in text_addition_blocks:
        total_block_text = ""
        edits_range_list = []

        # Grab div within it (each addition_block should only have one div)
        addition_div = addition_block.find('div')

        if addition_div:
            # if addition div only has one child of type string , this means the entire 
            # div is added text, so add entire range to list
            if (len(addition_div.contents) == 1 and
                isinstance(addition_div.contents[0], bs4.element.NavigableString)):

                ins_text = addition_div.text
                ins_range = {'start': 0, 'end': len(ins_text)}
                edits_range_list.append(ins_range)
                total_block_text += ins_text
            
            # else, addition div has multiple children, iterate through them and find the 
            # exact ranges of added text
            else:
                for child in addition_div.children:                
                    # if the child is a string, add it to the total_block_text
                    if isinstance(child, bs4.element.NavigableString):
                        total_block_text += child
                    
                    # if the child is an ins element with classes diffchange and diffchange-inline, 
                    # get text and add range to list
                    elif (child.name == 'ins' and 'diffchange' in child.get('class', []) 
                                            and 'diffchange-inline' in child.get('class', [])):
                        ins_text = child.text
                        
                        # if text is space or punctuation only, skip
                        punctuation_regex = re.compile(r"^[?!,.\":;—]+$")
                        if not ins_text.strip() or punctuation_regex.match(ins_text.strip()):
                            total_block_text += ins_text
                            continue

                        # store range in dict - start index (inclusive), end index (exclusive)
                        ins_range = {'start': len(total_block_text), 'end': len(total_block_text) + len(ins_text)}
                        edits_range_list.append(ins_range)
                        total_block_text += ins_text

        # first check: if edits_range_list is empty, skip this block and go to next block
        if (not edits_range_list):
            continue

        # secondary check: if there is a deletion block sibling (should be a previous sibling), compare
        # full text of both blocks convert to lowercase and remove punctuation and spaces for comparison 
        # - if they are the same, skip this block
        deletion_block = addition_block.find_previous_sibling('td', class_='diff-deletedline diff-side-deleted')
        deletion_block_text = ""
        deletion_block_text_no_deletions = ""
        if deletion_block:
            # get div within deletion block
            deletion_div = deletion_block.find('div')
    
            if deletion_div:
                for child in deletion_div.children:
                    if isinstance(child, bs4.element.NavigableString):
                        deletion_block_text += child
                        deletion_block_text_no_deletions += child
                    elif (child.name == 'del' and 'diffchange' in child.get('class', []) 
                                            and 'diffchange-inline' in child.get('class', [])):
                        del_text = child.text
                        deletion_block_text += del_text
                
            # strip all spaces, punctuation, and convert to lowercase for both texts
            add_text_stripped = re.sub(r'[?!,.\":;—\s]+', '', total_block_text).lower()
            del_text_stripped = re.sub(r'[?!,.\":;—\s]+', '', deletion_block_text).lower()
            del_text_no_deletions_stripped = re.sub(r'[?!,.\":;—\s]+', '', deletion_block_text_no_deletions).lower()

            # if texts are the same without spaces and punctuation, or without deletions, skip this block
            if (add_text_stripped == del_text_stripped or add_text_stripped == del_text_no_deletions_stripped):
                continue

        while True:
            # match <ref...>...</ref> and * {{cite...}}
            ref_regex = r'(<ref.*?>.*?</ref>|\*\s*\{\{cite.*?\}\})'
            match = re.search(ref_regex, total_block_text, flags=re.DOTALL)
            
            # break if no more matches
            if not match:
                break
            
            # get start and end indices of ref tag
            start, end = match.span()
            removal_length = end - start
            
            # update edit ranges
            for i in range(len(edits_range_list)):
                # vars for start and end of edit range (for readability)
                edit_range = edits_range_list[i]
                edit_start, edit_end = edit_range['start'], edit_range['end']

                # case 1: if ref tag removed completely before edit range
                if end <= edit_start:
                    edits_range_list[i] = ({'start': edit_start - removal_length, 
                                            'end': edit_end - removal_length})
                
                # case 2: if ref tag overlaps only the beginning of edit range
                elif start < edit_start < end < edit_end:
                    edits_range_list[i] = ({'start': start, 
                                            'end': edit_end - removal_length})
                     # add ref tag to set of updated refs
                    refs_set.add(total_block_text[start:end])
                
                # case 3: if ref tag overlaps only the end of edit range
                elif end > edit_end > start > edit_start:
                    edits_range_list[i] = ({'start': edit_start, 'end': start})
                    # add ref tag to set of updated refs
                    refs_set.add(total_block_text[start:end])
                
                # case 4: if ref tag encompasses the entire edit range
                elif start <= edit_start and edit_end <= end:
                    edits_range_list[i] = None
                    # add ref tag to set of updated refs
                    refs_set.add(total_block_text[start:end])
                
                # case 5: if ref tag is within the edit range
                elif start >= edit_start and edit_end >= end:
                    edits_range_list[i] = ({'start': edit_start, 'end': start})
                    # add ref tag to set of updated refs
                    refs_set.add(total_block_text[start:end])
            
            # remove the ref tag from the text
            total_block_text = total_block_text[:start] + total_block_text[end:]
            
            # remove None values from edits_range_list
            edits_range_list = [rg for rg in edits_range_list if rg is not None]

        # check again if edits_range_list is empty, skip this block and go to next block
        if (not edits_range_list):
            continue

        # split text into sentences (preserve space before sentence start if not first sentence)
        # sentence_ranges = [(m.start(0), m.end(0)) for m in re.finditer(r'\S.*?[.!?]', total_block_text)]
        sentence_ranges = [(m.start(0) if m.start(0) == 0 else m.start(0) - 1, m.end(0)) 
                           for m in re.finditer(r'\S.*?[.!?]', total_block_text)]
        
        # if end of last range is not end of text, add the rest of the text as a sentence
        if sentence_ranges and sentence_ranges[-1][1] != len(total_block_text):
            sentence_ranges.append((sentence_ranges[-1][1], len(total_block_text)))
        
        # if no sentence ranges, add entire text as a sentence
        if not sentence_ranges:
            sentence_ranges.append((0, len(total_block_text)))

        # make a list to keep track of which sentences contain the edit ranges
        sentence_indices_state = [False] * len(sentence_ranges)

        for edit_range in edits_range_list:
            e_start, e_end = edit_range['start'], edit_range['end']
            
            # init vars for edit sentences range
            start_sentence_idx, end_sentence_idx = None, None
           
            # iterate thru sentence ranges to find the sentence range that the edit range is in
            for i, sentence_range in enumerate(sentence_ranges):
                s_start, s_end = sentence_range[0], sentence_range[1]
                
                # if start index is in current sentence range, set edit_sent_start to s_start
                if (s_start <= e_start < s_end):
                    start_sentence_idx = i
                
                # if end index is in current sentence range, set edit_sent_end to s_end
                if (s_start < e_end <= s_end):
                    end_sentence_idx = i
                    break

            # if both start and end sentence indices are found, set the state of the sentences
            # to True (and the sentences between to them)
            if (start_sentence_idx is not None and end_sentence_idx is not None):
                for i in range(start_sentence_idx, end_sentence_idx+1):
                    sentence_indices_state[i] = True

        # add the sentences that contain the edit ranges to the set (append sentences next to each other)
        sentences_to_add = ""
        for i, sentence_range in enumerate(sentence_ranges):
            if sentence_indices_state[i]:
                sentences_to_add += total_block_text[sentence_range[0]:sentence_range[1]]
            else:
                if sentences_to_add:
                    # strip potential spaces at front of text and add sentence to set
                    sentence_set.append(sentences_to_add.lstrip())
                    sentences_to_add = ""
        
        # if there are sentences left in sentences_to_add, add them to the set
        if sentences_to_add:
            sentence_set.append(sentences_to_add.lstrip())
    
    return sentence_set, refs_set

# returns a tuple containing a list of before and after sentence pairs, given the url of a wiki-diff page
def get_before_after_text(diff_page_url: str, lang_wiki: str) -> List[Tuple[str, str]]:
    
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
    prev_after_list = []

    # loop through tr blocks
    for tr_block in tr_blocks:
        # if tr block has a td element child with class "diff-marker"
        # and without a data-marker attribute, this means it's simply
        # shifted text. skip this block
        if tr_block.find('td', class_='diff-marker') and not tr_block.find('td', class_='diff-marker').get('data-marker'):
            continue
        
        # get the td elements within the tr block with class "diff-deletedline" or "diff-addedline"
        deletion_block = tr_block.find('td', class_='diff-deletedline')
        addition_block = tr_block.find('td', class_='diff-addedline')
        addition_text = ""
        deletion_text = ""

        # add aggregated text from both before and after text to tuple
        if deletion_block:
            deletion_text, del_refs = get_aggregated_text(deletion_block)
        if addition_block:
            addition_text, del_refs = get_aggregated_text(addition_block)

        # if theres no deletion or addition text, skip this block
        if not deletion_text and not addition_text:
            continue

        before_after_pair = (deletion_text, addition_text)
        prev_after_list.append(before_after_pair)

    return prev_after_list

# takes in an addition or deletion block of html and returns both the aggregated text and a set of references
def get_aggregated_text(block: bs4.element.Tag) -> Tuple[str, set]:
    total_block_text = ""
    refs_set = set()
    
    for child in block.children:
        total_block_text += child.text

    # remove refs from text and add to set
    ref_regex = r'(<ref.*?>.*?</ref>|\*\s*\{\{cite.*?\}\})'
    
    while True:
        match = re.search(ref_regex, total_block_text, flags=re.DOTALL)
        if not match:
            break
        start, end = match.span()
        refs_set.add(total_block_text[start:end])
        total_block_text = total_block_text[:start] + total_block_text[end:]
    
    return total_block_text, refs_set