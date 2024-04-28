from wikipedia_edit_scrape_tool.scrape_page_content import get_monthly_revision_comparison_links

def main():
    with open('wiki_page_ids.txt', 'r', encoding='utf-8') as file:
        for line in file:
            page_title = line.strip()
            try:
                links = get_monthly_revision_comparison_links(page_title)
                if not links:
                    print(f"Testing page: {page_title}")
                    print(f"No links found for {page_title}")
            except Exception as e:
                print(f"Testing page: {page_title}")
                print(f"Failed to process page {page_title}: {str(e)}")
        print("Done testing all pages")

if __name__ == '__main__':
    main()
