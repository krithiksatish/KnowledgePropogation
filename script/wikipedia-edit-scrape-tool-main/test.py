import wikipedia_edit_scrape_tool

wiki_id = "Attention_deficit_hyperactivity_disorder"
wiki_id = "COVID-19_vaccine"
wiki_id = "Tim_Cook"
wikipedia_edit_scrape_tool.get_edit_history(wiki_id, target_languages=['enwiki'])
wikipedia_edit_scrape_tool.get_diff_between_two_revisions(wiki_id, "1194639363", "1189697434")