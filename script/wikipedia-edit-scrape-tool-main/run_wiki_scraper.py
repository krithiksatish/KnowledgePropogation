import wikipedia_edit_scrape_tool
from wikipedia_edit_scrape_tool.scrape_page_content import get_del_add_text, get_monthly_revision_comparison_links
from wikipedia_edit_scrape_tool.sentence_simlarity import SentenceSimilarityCalculator

def main():
    with open('wiki_page_ids.txt', 'r') as file:
        for line in file:
            page_title = line.strip()
            print(f"Processing page: {page_title}")

            # Get the monthly revision comparison links for the page
            comparison_urls = get_monthly_revision_comparison_links(page_title)

            # Iterate through the comparison URLs and get text differences
            for url in comparison_urls:
                print(f"  - Processing revision comparison: {url}")
                context_pairs_list = get_del_add_text(url)
                print(f"    - Found {len(context_pairs_list)} context pairs")

if __name__ == '__main__':
    main()
