import wikipedia_edit_scrape_tool
from wikipedia_edit_scrape_tool.scrape_page_content import get_diff_text
from wikipedia_edit_scrape_tool.scrape_edit_history import get_diff_between_two_revisions

wiki_id = "Attention_deficit_hyperactivity_disorder"
# wiki_id = "COVID-19_vaccine"
# wiki_id = "Tim_Cook"
# wikipedia_edit_scrape_tool.get_edit_history(wiki_id, target_languages=['enwiki'])
# wikipedia_edit_scrape_tool.get_diff_between_two_revisions(wiki_id, "1194639363", "1189697434")

#Example that doesn't work great
diff_url = "https://en.wikipedia.org/w/index.php?title=Attention_deficit_hyperactivity_disorder&diff=1198388681&oldid=1195199126"

# Example that works well, result is "ADHD represents the extreme lower end of the continuous (bell curve) dimensional trait of executive functioning and self-regulation, which is supported by twin, brain imaging and molecular genetic studies."
diff_url = "https://en.wikipedia.org/w/index.php?title=Attention_deficit_hyperactivity_disorder&diff=1198388681&oldid=1198147118"

diff_text_set = get_diff_text(diff_url, 'enwiki')

# get_diff_between_two_revisions uses get_diff_text(), but just takes in simpler input
#diff_text_set = get_diff_between_two_revisions(wiki_id, '1194639363','1189697434')
