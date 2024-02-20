import wikipedia_edit_scrape_tool
from wikipedia_edit_scrape_tool.scrape_page_content import get_diff_text, get_before_after_text
from wikipedia_edit_scrape_tool.scrape_edit_history import get_diff_between_two_revisions
from wikipedia_edit_scrape_tool.sentence_simlarity import SentenceSimilarityCalculator

wiki_id = "Attention_deficit_hyperactivity_disorder"
# wiki_id = "COVID-19_vaccine"
# wiki_id = "Tim_Cook"
# wikipedia_edit_scrape_tool.get_edit_history(wiki_id, target_languages=['enwiki'])
# wikipedia_edit_scrape_tool.get_diff_between_two_revisions(wiki_id, "1194639363", "1189697434")

# need more test cases (works well on this one now though)
diff_url_test_set = [
    # "https://en.wikipedia.org/w/index.php?title=Attention_deficit_hyperactivity_disorder&diff=1198388681&oldid=1195199126",
    # "https://en.wikipedia.org/w/index.php?title=Attention_deficit_hyperactivity_disorder&diff=1198388681&oldid=1198147118",
    "https://en.wikipedia.org/w/index.php?title=Deep_learning&diff=1198810161&oldid=1185260881"
]

for diff_url in diff_url_test_set:
    print("========================")
    print("----------DIFF----------")
    print("========================")
    before_after_list = get_before_after_text(diff_url, "enwiki")
    for before, after in before_after_list:
        print("----------BEFORE----------")
        print(before)
        print("----------AFTER----------")
        print(after)

        sim = SentenceSimilarityCalculator()
        similarity_score = sim.cosine_similarity(before, after)

        print("----------SIMILARITY----------")
        print(similarity_score)
        print("----------SIGNIFICANT EDIT----------")
        print(sim.is_signifcant_edit(before, after))
        print("========================")
        print ("\n")


# get_diff_between_two_revisions uses get_diff_text(), but just takes in simpler input
#diff_text_set = get_diff_between_two_revisions(wiki_id, '1194639363','1189697434')
