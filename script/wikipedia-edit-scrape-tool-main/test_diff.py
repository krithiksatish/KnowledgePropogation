import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import difflib
from nltk.tokenize import sent_tokenize
import re


# Download punkt tokenizer if not already downloaded
nltk.download('punkt')

def get_aggregated_text(block: str):
    total_block_text = block
    refs_set = set()

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

    return total_block_text


def find_closest_sentences(sentences_before, sentences_after, threshold=0.4):
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
    merged_keys = set()
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

    print(combined_final_mappings)

    return combined_final_mappings, unmatched_before, unmatched_after

def find_differences_and_context(changed_text, og_text):
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

    # Print modified sentences
    print("Modified sentences:")
    for before, after in modified:
        print(f"Before: {before}")
        print(f"After: {after}")
        print()

def find_closest_sentences_difflib(sentences_before, sentences_after, threshold=0.4):
    # Compute similarity between each pair of sentences using difflib
    before_to_after = {}
    after_to_before = {}

    for i, before_sentence in enumerate(sentences_before):
        best_match_index = None
        best_match_score = 0
        for j, after_sentence in enumerate(sentences_after):
            score = difflib.SequenceMatcher(None, before_sentence, after_sentence).ratio()
            if score > best_match_score:
                best_match_score = score
                best_match_index = j
        if best_match_score > threshold:
            if best_match_index not in before_to_after:
                before_to_after[best_match_index] = []
            before_to_after[best_match_index].append(i)

    for j, after_sentence in enumerate(sentences_after):
        best_match_index = None
        best_match_score = 0
        for i, before_sentence in enumerate(sentences_before):
            score = difflib.SequenceMatcher(None, after_sentence, before_sentence).ratio()
            if score > best_match_score:
                best_match_score = score
                best_match_index = i
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

    print(combined_final_mappings)

    return combined_final_mappings, unmatched_before, unmatched_after

def find_differences_and_context_difflib(changed_text, og_text):
    # Tokenize texts into sentences
    sentences_before = nltk.sent_tokenize(og_text)
    sentences_after = nltk.sent_tokenize(changed_text)

    # Get mappings
    mappings, unmatched_before, unmatched_after = find_closest_sentences_difflib(sentences_before, sentences_after)

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

    # Print modified sentences
    print("Modified sentences:")
    for before, after in modified:
        print(f"Before: {before}")
        print(f"After: {after}")
        print()

def find_differences_and_context_ndiff(changed_text: str, og_text: str, context_sentences: int = 1):
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
            end_context = min(len(diff), i + context_sentences + 1)
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

    # Extracting before and after modified sentences
    modified_sentences = []
    for context in merged_contexts:
        before = []
        after = []
        for line in context:
            if line.startswith('- '):
                before.append(line[2:])
            elif line.startswith('+ '):
                after.append(line[2:])
        if before or after:
            before_text = ' '.join(before)
            after_text = ' '.join(after)
            modified_sentences.append((before_text, after_text))

    # Print modified sentences
    print("Modified sentences:")
    for before, after in modified_sentences:
        print(f"Before: {before}")
        print(f"After: {after}")
        print()


# # Example texts: https://en.wikipedia.org/w/index.php?title=Cognition&diff=5967701&oldid=4418685
# text_before = "Cognition is a diffuse term and is used in radically different ways by different disciplines. In psychology, it refers to an information processing view of an individual's psychological functions. Wider interpretations of the meaning of cognition link it to the development of concepts. Individual minds, groups, organizations, and even larger [[coalition]]s can be modelled as [[society of mind theory|societies]] which cooperate to form concepts. The autonomous elements of each 'society' would have the opportunity to demonstrate [[emergence|emergent behavior]]."

# text_after = "''Cognition'' is a diffuse term and is used in radically different ways by different disciplines. In psychology, it refers to an [[information processing]] view of an individual's psychological functions. Wider interpretations of the meaning of ''cognition'' link it to the development of ''concepts''; individual minds, groups, organizations, and even larger [[coalition]]s of [[entity|entities]] can be modelled as ''[[society of mind theory|societies]]'' which [[cooperation|cooperate]] to form [[concepts]]. The autonomous elements of each '[[society]]' would have the opportunity to demonstrate [[emergence|emergent behavior]] in the face of some crisis or opportunity."

# # Example texts: https://en.wikipedia.org/w/index.php?title=Cognition&diff=5967701&oldid=4418685
# text_before = "One famous image taken during the first Apollo mission to the Moon, [[Earthrise]], which shows planet Earth in a single photograph, is now the icon for [[Earth Day]], which did not arise until after the image became widespread.  At this level, an example of an 'emergent behavior' might be concern for 'Spaceship Earth', as encouraged by the development of orbiting [[space observatory|space observatories]] etc. "

# text_after = "One famous image, ''[[Earthrise]]'', taken during [[Apollo 8]], the first Apollo mission to the Moon, shows planet Earth in a single photograph. ''Earthrise'' is now the icon for [[Earth Day]], which did not arise until after the image became widespread.  At this level, an example of an 'emergent behavior' might be ''concern for [[Spaceship Earth]]'', as encouraged by the development of orbiting [[space observatory|space observatories]] etc. "


# # Examples tests: https://en.wikipedia.org/w/index.php?title=Stimulant_psychosis&diff=1160922483&oldid=1132916094
# text_before = r"""'''Stimulant psychosis''' is a [[mental disorder]] characterized by [[psychotic]] symptoms (such as [[hallucination]]s, paranoid ideation, [[delusion]]s, [[Thought disorder|disorganized thinking]], grossly disorganized behaviour) which involves and typically occurs following an overdose or several day 'binge' on [[psychostimulant]]s;<ref name= "ICD-11-web"/> however, it has also been reported to occur in approximately 0.1% of individuals, within the first several weeks after starting [[amphetamine]] or [[methylphenidate]] therapy.<ref name= "Adderall XR .1%">{{cite web | title = Adderall XR Prescribing Information | url = http://www.accessdata.fda.gov/drugsatfda_docs/label/2013/021303s026lbl.pdf | publisher = [[US Food and Drug Administration]]| work= FDA.gov |date=December 2013 | access-date = 30 December 2013 | quote = Treatment-emergent psychotic or manic symptoms, e.g. hallucinations, delusional thinking, or mania in children and adolescents without prior history of psychotic illness or mania can be caused by stimulants at usual doses.&nbsp;... In a pooled analysis of multiple short-term, placebo controlled studies, such symptoms occurred in about 0.1% (4 patients with events out of 3482 exposed to methylphenidate or amphetamine for several weeks at usual doses) of stimulant-treated patients compared to 0 in placebo-treated patients.}}</ref><ref name="Cochrane recreational amph psychosis">{{cite journal |last1=Shoptaw |first1=Steven J |last2=Kao |first2=Uyen |last3=Ling |first3=Walter |title=Treatment for amphetamine psychosis |journal= Cochrane Database of Systematic Reviews |issue=1 |pages=CD003026 |date=21 January 2009 |doi=10.1002/14651858.CD003026.pub3 |pmid=19160215 |pmc=7004251 }}</ref><ref name="pmid19171629">{{cite journal |vauthors= Mosholder AD, Gelperin K, Hammad TA, Phelan K, Johann-Liang R | title = Hallucinations and other psychotic symptoms associated with the use of attention-deficit/hyperactivity disorder drugs in children | journal = Pediatrics | volume = 123 | issue = 2 | pages = 611–616 | date = February 2009 | pmid = 19171629 | doi = 10.1542/peds.2008-0185 | s2cid = 22391693 }}</ref> Methamphetamine psychosis, or long-term effects of stimulant use in the brain (at the molecular level), depend upon genetics and may persist for some time.<ref>{{cite journal |last1= Greening |first1=David W. |last2=Notaras |first2=Michael |last3=Chen |first3=Maoshan |last4=Xu |first4=Rong |last5=Smith |first5=Joel D. |last6=Cheng |first6=Lesley |last7=Simpson |first7=Richard J. |last8=Hill |first8=Andrew F. |last9=van den Buuse |first9=Maarten |title=Chronic methamphetamine interacts with BDNF Val66Met to remodel psychosis pathways in the mesocorticolimbic proteome |journal=Molecular Psychiatry |date=10 December 2019 |volume=26 |issue=8 |pages=4431–4447 |doi= 10.1038/s41380-019-0617-8 |pmid=31822818 |s2cid=209169489 }}</ref>"""

# text_after = r"""'''Stimulant psychosis''' is a [[mental disorder]] characterized by [[psychotic]] symptoms (such as [[hallucination]]s, paranoid ideation, [[delusion]]s, [[Thought disorder|disorganized thinking]], grossly disorganized behaviour). It involves and typically occurs following an overdose or several day 'binge' on [[psychostimulant]]s;<ref name= "ICD-11-web"/> however, one study reported occurrences at regularly prescribed doses in approximately 0.1% of individuals within the first several weeks after starting [[amphetamine]] or [[methylphenidate]] therapy.<ref name= "Adderall XR .1%">{{cite web | title = Adderall XR Prescribing Information | url = http://www.accessdata.fda.gov/drugsatfda_docs/label/2013/021303s026lbl.pdf | publisher = [[US Food and Drug Administration]]| work= FDA.gov |date=December 2013 | access-date = 30 December 2013 | quote = Treatment-emergent psychotic or manic symptoms, e.g. hallucinations, delusional thinking, or mania in children and adolescents without prior history of psychotic illness or mania can be caused by stimulants at usual doses.&nbsp;... In a pooled analysis of multiple short-term, placebo controlled studies, such symptoms occurred in about 0.1% (4 patients with events out of 3482 exposed to methylphenidate or amphetamine for several weeks at usual doses) of stimulant-treated patients compared to 0 in placebo-treated patients.}}</ref><ref name="Cochrane recreational amph psychosis">{{cite journal |last1=Shoptaw |first1=Steven J |last2=Kao |first2=Uyen |last3=Ling |first3=Walter |title=Treatment for amphetamine psychosis |journal= Cochrane Database of Systematic Reviews |issue=1 |pages=CD003026 |date=21 January 2009 |volume=2009 |doi=10.1002/14651858.CD003026.pub3 |pmid=19160215 |pmc=7004251 }}</ref><ref name="pmid19171629">{{cite journal |vauthors= Mosholder AD, Gelperin K, Hammad TA, Phelan K, Johann-Liang R | title = Hallucinations and other psychotic symptoms associated with the use of attention-deficit/hyperactivity disorder drugs in children | journal = Pediatrics | volume = 123 | issue = 2 | pages = 611–616 | date = February 2009 | pmid = 19171629 | doi = 10.1542/peds.2008-0185 | s2cid = 22391693 }}</ref> Methamphetamine psychosis, or long-term effects of stimulant use in the brain (at the molecular level), depend upon genetics and may persist for some time.<ref>{{cite journal |last1= Greening |first1=David W. |last2=Notaras |first2=Michael |last3=Chen |first3=Maoshan |last4=Xu |first4=Rong |last5=Smith |first5=Joel D. |last6=Cheng |first6=Lesley |last7=Simpson |first7=Richard J. |last8=Hill |first8=Andrew F. |last9=van den Buuse |first9=Maarten |title=Chronic methamphetamine interacts with BDNF Val66Met to remodel psychosis pathways in the mesocorticolimbic proteome |journal=Molecular Psychiatry |date=10 December 2019 |volume=26 |issue=8 |pages=4431–4447 |doi= 10.1038/s41380-019-0617-8 |pmid=31822818 |s2cid=209169489 }}</ref>"""

# Example texts: https://en.wikipedia.org/w/index.php?title=Perception&diff=490557073&oldid=485187543
# text_before = """'''Perception''' (from the Latin ''perceptio, percipio'') is the process of attaining [[awareness]] or [[understanding]] of the [[natural environment|environment]] by organizing and interpreting [[sense|sensory]] [[information]].<ref name="pomerantz" /><ref>Defined as "receiving, collecting, action of taking possession, apprehension with the mind or senses." in [http://www.OED.com Oxford English Dictionary: The way in which things are understood; ]</ref> All perception involves signals in the [[nervous system]], which in turn result from physical stimulation of the sense organs.<ref name="Goldstein5">Goldstein (2009) pp. 5–7</ref> For example, vision involves [[photon|light]] striking the [[retina]]s of the eyes, smell is mediated by odor [[molecules]] and hearing involves [[sound wave|pressure waves]]. Perception is not the passive receipt of these signals, but can be shaped by [[Perceptual learning|learning]], [[memory]] and [[expectation (epistemic)|expectation]].<ref name="mind_perception" /><ref name="Bernstein2010" /> Perception involves these "top-down" effects as well as the "bottom-up" process of processing sensory input.<ref name="Bernstein2010">{{cite book|last=Bernstein|first=Douglas A.|title=Essentials of Psychology|url=http://books.google.com/books?id=rd77N0KsLVkC&pg=PA123|accessdate=25 March 2011|date=5 March 2010|publisher=Cengage Learning|isbn=9780495906933|pages=123–124}}</ref> The "bottom-up" processing is basically low-level information that's used to build up higher-level information (i.e - shapes for object recognition). The "top-down" processing refers to a person's concept and expectations (knowledge) that influence perception. Perception depends on complex functions of the nervous system, but subjectively seems mostly effortless because this processing happens outside conscious awareness.<ref name="Goldstein5" />"""
# text_after = """'''Perception''' (from the Latin ''perceptio, percipio'') is the organization, identification, and interpretation of sensory information in order to fabricate a mental representation through the process of transduction, which sensors in the body transform signals from the environment into encoded neural signals.<ref>{{cite book|last=Schacter|first=Daniel|title=Psychology|year=2011|publisher=Worth Publishers}}</ref> All perception involves signals in the [[nervous system]], which in turn result from physical stimulation of the sense organs.<ref name="Goldstein5">Goldstein (2009) pp. 5–7</ref> For example, vision involves [[photon|light]] striking the [[retina]]s of the eyes, smell is mediated by odor [[molecules]] and hearing involves [[sound wave|pressure waves]]. Perception is not the passive receipt of these signals, but can be shaped by [[Perceptual learning|learning]], [[memory]] and [[expectation (epistemic)|expectation]].<ref name="mind_perception" /><ref name="Bernstein2010" /> Perception involves these "top-down" effects as well as the "bottom-up" process of processing sensory input.<ref name="Bernstein2010">{{cite book|last=Bernstein|first=Douglas A.|title=Essentials of Psychology|url=http://books.google.com/books?id=rd77N0KsLVkC&pg=PA123|accessdate=25 March 2011|date=5 March 2010|publisher=Cengage Learning|isbn=9780495906933|pages=123–124}}</ref> The "bottom-up" processing is basically low-level information that's used to build up higher-level information (i.e. - shapes for object recognition). The "top-down" processing refers to a person's concept and expectations (knowledge) that influence perception. Perception depends on complex functions of the nervous system, but subjectively seems mostly effortless because this processing happens outside conscious awareness.<ref name="Goldstein5" />"""

# text_before = """The ''principles of grouping'' (or ''Gestalt laws of grouping'') are a set of principles in [[psychology]], first proposed by [[Gestalt psychology|Gestalt psychologists]] to explain how humans naturally perceive objects as organized patterns and objects. Gestalt psychologists argued that these principles exist because the mind has an innate disposition to perceive patterns in the stimulus based on certain rules. These principles are organized into five categories. The principle of ''proximity'' states that, all else being equal, perception tends to group stimuli that are close together as part of the same object, and stimuli that are far apart as two separate objects. The principle of ''similarity'' states that, all else being equal, perception lends itself to seeing stimuli that physically resemble each other as part of the same object, and stimuli that are different as part of a different object. This allows for people to distinguish between adjacent and overlapping objects based on their visual texture and resemblance. The principle of ''closure'' refers to the mind’s tendency to see complete figures or forms even if a picture is incomplete, partially hidden by other objects, or if part of the information needed to make a complete picture in our minds is missing. For example, if part of a shape’s border is missing people still tend to see the shape as completely enclosed by the border and ignore the gaps. The principle of ''good continuation'' makes sense of stimuli that overlap: when there is an intersection between two or more objects, people tend to perceive each as a single uninterrupted object. The principle of ''common fate'' groups stimuli together on the basis of their movement. When visual elements are seen moving in the same direction at the same rate, perception associates the movement as part of the same stimulus. This allows people to make out moving objects even when other details, such as color or outline, are obscured. The principle of ''good form'' refers to the tendency to group together forms of similar shape, pattern, color, etc.<ref>Gray, Peter O. (2006): ''Psychology'', 5th ed.,  New York: Worth, p. 281. ISBN 978-0716706175</ref><ref>{{cite book|ref=harv|chapterurl=http://www.sinauer.com./wolfe/chap4/gestaltF.htm|chapter=Gestalt Grouping Principles|publisher=Sinauer Associates|year=2008|edition=2nd|title=Sensation and Perception|first1=Jeremy M.|last1=Wolfe|first3=Dennis M.|last3=Levi|first2=Keith R.|last2=Kluender|first5=Rachel S.|last5=Herz|first6=Roberta L.|last6=Klatzky|first4=Linda M.|last4=Bartoshuk|first7=Susan J.|last7=Lederman|isbn=9780878939381|pages=78, 80}}</ref><ref>Goldstein (2009). pp. 105–107</ref><ref>{{cite encyclopaedia|ref=harv|encyclopedia=Encyclopaedic Dictionary of Psychological Terms|first=J. C.|last=Banerjee|publisher=M.D. Publications Pvt. Ltd|year=1994|isbn10=818588028X|isbn=9788185880280|article=Gestalt Theory of Perception|pages=107–108}}</ref> Later research has identified additional grouping principles.<ref>{{cite book|ref=harv|title=Psychology: themes and variations|first=Wayne|last=Weiten|edition=4th|publisher=Brooks/Cole Pub. Co.|year=1998|isbn10=0534340148|isbn=9780534340148|page=144}}</ref>"""
# text_after = """The ''principles of grouping'' (or ''Gestalt laws of grouping'') are a set of principles in [[psychology]], first proposed by [[Gestalt psychology|Gestalt psychologists]] to explain how humans naturally perceive objects as organized patterns and objects. Gestalt psychologists argued that these principles exist because the mind has an innate disposition to perceive patterns in the stimulus based on certain rules. These principles are organized into six categories. The principle of ''proximity'' states that, all else being equal, perception tends to group stimuli that are close together as part of the same object, and stimuli that are far apart as two separate objects. The principle of ''similarity'' states that, all else being equal, perception lends itself to seeing stimuli that physically resemble each other as part of the same object, and stimuli that are different as part of a different object. This allows for people to distinguish between adjacent and overlapping objects based on their visual texture and resemblance. The principle of ''closure'' refers to the mind’s tendency to see complete figures or forms even if a picture is incomplete, partially hidden by other objects, or if part of the information needed to make a complete picture in our minds is missing. For example, if part of a shape’s border is missing people still tend to see the shape as completely enclosed by the border and ignore the gaps. The principle of ''good continuation'' makes sense of stimuli that overlap: when there is an intersection between two or more objects, people tend to perceive each as a single uninterrupted object. The principle of ''common fate'' groups stimuli together on the basis of their movement. When visual elements are seen moving in the same direction at the same rate, perception associates the movement as part of the same stimulus. This allows people to make out moving objects even when other details, such as color or outline, are obscured. The principle of ''good form'' refers to the tendency to group together forms of similar shape, pattern, color, etc.<ref>Gray, Peter O. (2006): ''Psychology'', 5th ed.,  New York: Worth, p. 281. ISBN 978-0716706175</ref><ref>{{cite book|ref=harv|chapterurl=http://www.sinauer.com./wolfe/chap4/gestaltF.htm|chapter=Gestalt Grouping Principles|publisher=Sinauer Associates|year=2008|edition=2nd|title=Sensation and Perception|first1=Jeremy M.|last1=Wolfe|first3=Dennis M.|last3=Levi|first2=Keith R.|last2=Kluender|first5=Rachel S.|last5=Herz|first6=Roberta L.|last6=Klatzky|first4=Linda M.|last4=Bartoshuk|first7=Susan J.|last7=Lederman|isbn=9780878939381|pages=78, 80}}</ref><ref>Goldstein (2009). pp. 105–107</ref><ref>{{cite encyclopaedia|ref=harv|encyclopedia=Encyclopaedic Dictionary of Psychological Terms|first=J. C.|last=Banerjee|publisher=M.D. Publications Pvt. Ltd|year=1994|isbn10=818588028X|isbn=9788185880280|article=Gestalt Theory of Perception|pages=107–108}}</ref> Later research has identified additional grouping principles.<ref>{{cite book|ref=harv|title=Psychology: themes and variations|first=Wayne|last=Weiten|edition=4th|publisher=Brooks/Cole Pub. Co.|year=1998|isbn10=0534340148|isbn=9780534340148|page=144}}</ref>"""

# Example texts: https://en.wikipedia.org/w/index.php?title=Traffic_psychology&diff=4339096&oldid=3196196
# text_before = """[[Traffic]] psychology is a young expanding field in [[psychology]]. Whereas traffic psychology is primarily related to “the study of the behaviour of road users and the psychological processes underlying that behaviour” (Rothengatter, 1997, 223) as well as to the relation between [[behaviour]] and accidents, transportation psychology, sometimes referred to as mobility psychology, has its focus on mobility issues, individual and social factors in the movement of people and goods, and travel demand management. """
# text_after = """'''Traffic psychology''' is a young expanding field in [[psychology]]. Whereas ''[[traffic]]'' psychology is primarily related to ''"the study of the behaviour of road users and the psychological processes underlying that behaviour"'' (Rothengatter, 1997, 223) as well as to the relation between [[behaviour]] and [[accidents]], ''[[transportation]]'' psychology, sometimes referred to as ''mobility'' psychology, has its focus on mobility issues, individual and social factors in the movement of people and goods, and ''[[travel]] demand management (TDM)''. """

text_before = """There is no single theoretical framework in traffic psychology, but many specific models explaining, e.g., perceptual, attentional, cognitive, social, motivational and emotional determinants of mobility and traffic behaviour. One of the most prominent behavioural models divides the various tasks involved in traffic participation into three hierarchical levels, i.e. the strategic, the tactical and the operational level. The model demonstrates the diversity of decision and control tasks which have to be accomplished when driving a vehicle. However, until now, most of the psychological models have a rather heuristic nature, e.g. risk theories like Wilde´s risk-homeostasis, Fuller´s task capability model, and thus are not sufficiently precise to allow for concrete behavioural prediction and control. This is partly due to the importance of individual differences, a major topic of psychology which in traffic and transportation has not yet been sufficiently accounted for. On the other hand, social-psychological attitude-behaviour models, such as Ajzen´s theory of planned behaviour, have been helpful in identifying determinants of mobility decisions."""
text_after = """There is no single theoretical framework in traffic psychology, but many specific models explaining, e.g., perceptual, attentional, [[cognitive]], [[social]], [[motivational]] and [[emotional]] determinants of mobility and traffic behaviour. One of the most prominent behavioural models divides the various tasks involved in traffic participation into three hierarchical levels, i.e. the strategic, the tactical and the operational level. The model demonstrates the diversity of decision and control tasks which have to be accomplished when driving a vehicle. However, until now, most of the psychological models have a rather heuristic nature, e.g. risk theories like Wilde's [[risk-homeostasis]], [[Fuller's task capability model]], and thus are not sufficiently precise to allow for concrete behavioural prediction and control. This is partly due to the importance of individual differences, a major topic of psychology which in traffic and transportation has not yet been sufficiently accounted for. On the other hand, social-psychological attitude-behaviour models, such as [[Ajzen's theory of planned behaviour]], have been helpful in identifying determinants of mobility decisions."""

# Find differences and context
text_before = get_aggregated_text(text_before)
text_after = get_aggregated_text(text_after)
print ("--------------tfidf---------------")
find_differences_and_context(text_after, text_before)
print ("--------------difflib---------------")
find_differences_and_context_difflib(text_after, text_before)
print ("--------------ndiff---------------")
find_differences_and_context_ndiff(text_after, text_before)