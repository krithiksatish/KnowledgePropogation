import csv
import requests
from urllib.parse import urlparse
import threading

from individual_sites_scraper import *
    
def get_actual_domain(url, result_dict, lock):
    """
    Fetches the actual domain from a given URL.
    """
    try:
        response = requests.head(url, allow_redirects=True)
        hosting_domain = urlparse(response.url).netloc
        with lock:
            result_dict[url] = (hosting_domain, response.url)
            #print(hosting_domain)
    except Exception as e:
        print(f"Error accessing URL: {e}")

def replace_doi_domain_parallel(csv_file):
    """
    Replaces the 'doi domain' key-value pair with the actual domain fetched from the URL using parallel processing.
    """
    result_dict = {}
    lock = threading.Lock()

    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        rows = []
        threads = []

        for row in reader:
            raw_text_dict = eval(row['raw_text'])
            for key in raw_text_dict.keys():
                if key == 'doi domain':
                    url = raw_text_dict[key]
                    thread = threading.Thread(target=get_actual_domain, args=(url, result_dict, lock))
                    thread.start()
                    threads.append(thread)
                    
            rows.append(row)

        for thread in threads:
            thread.join()

    # Update the raw_text column with the fetched domains
    for row in rows:
        raw_text_dict = eval(row['raw_text'])
        new_raw_text_dict = {}

        for domain, url in raw_text_dict.items():
            if domain == 'doi domain' and url in result_dict.keys():
                new_raw_text_dict[result_dict[url][0]] = result_dict[url][1]
            else:
                new_raw_text_dict[domain] = url

        row['raw_text'] = str(new_raw_text_dict)

    # Write the modified rows to a new CSV file
    fieldnames = ['', 'title', 'abstract', 'affiliations', 'authors', 'journal', 'date', 'keywords', 'url', 'raw_text']
    
    with open('replaced_doi_urls.csv', 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# TODO: use this method to call individual_sites_scraper's scraping logic
def extract_raw_article_text(href_dict):
    # Temporarily just returns the site name, will replace with returning full-text of article
    if 'www.mdpi.com' in href_dict.keys(): # currently implementing
        return 'mdpi'
    elif 'www.ncbi.nlm.nih.gov' in href_dict.keys(): #PMC
        return 'PMC'
        return extract_full_text_PMC(href_dict['www.ncbi.nlm.nih.gov'])
    elif 'linkinghub.elsevier.com' in href_dict.keys(): #403 Error
        return 'elsevier'
    elif 'link.springer.com' in href_dict.keys(): #Implemented
        return 'springer'
    elif 'journals.sagepub.com' in href_dict.keys():
        return 'sagepub'
    elif 'onlinelibrary.wiley.com' in href_dict.keys(): #403 Error
        return 'wiley'
    elif 'www.tandfonline.com' in href_dict.keys():
        return 'tandfonline'
    elif 'jamanetwork.com' in href_dict.keys(): #Implemented
        return 'jamanetwork'
    elif 'www.nature.com' in href_dict.keys(): #Implemented (not pushed yet)
        return 'nature'
    elif 'journals.lww.com' in href_dict.keys(): #Implemented
        return 'journals.lww'
    
    return 'NO TEXT (not hosted on top 3 sites)'


if __name__ == "__main__":
    # Replaces all the doi domains/url of articles.csv with the real domain/url, updated csv in 'replaced_doi_urls.csv'
    csv_file = 'articles.csv'  # Specify the path to your CSV file
    replace_doi_domain_parallel(csv_file)

    # With this new CSV file, run the invidiual sites scrapers for each row, replace the 
    #'full_text' column (currently contaning a dict with {domain name, url} with the full_text


    # Remove unnecessary rows (such as affiliations, authors, journal, keywords)

    # Clean CSV file

    # Resulting CSV is what we put into database

