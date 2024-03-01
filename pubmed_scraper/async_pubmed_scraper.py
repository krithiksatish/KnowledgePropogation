# Script used from https://github.com/IliaZenkov/async-pubmed-scraper
"""
Author: Ilia Zenkov
Date: 9/26/2020

This script asynchronously scrapes Pubmed - an open-access database of scholarly research articles -
and saves the data to a DataFrame which is then written to a CSV intended for further processing
This script is capable of scraping a list of keywords asynchronously

Contains the following functions:
    make_header:        Makes an HTTP request header using a random user agent
    get_num_pages:      Finds number of pubmed results pages returned by a keyword search
    extract_by_article: Extracts data from a single pubmed article to a DataFrame
    get_pmids:          Gets PMIDs of all article URLs from a single page and builds URLs to pubmed articles specified by those PMIDs
    build_article_urls: Async wrapper for get_pmids, creates asyncio tasks for each page of results, page by page,
                        and stores article urls in urls: List[string]
    get_article_data:   Async wrapper for extract_by_article, creates asyncio tasks to scrape data from each article specified by urls[]

requires:
    BeautifulSoup4 (bs4)
    PANDAS
    requests
    asyncio
    aiohttp
    nest_asyncio (OPTIONAL: Solves nested async calls in jupyter notebooks)
"""

import argparse
from collections import OrderedDict
import os
import time
from bs4 import BeautifulSoup
import bs4
import pandas as pd
import random
import requests
import asyncio
import aiohttp
import socket
import warnings; warnings.filterwarnings('ignore') # aiohttp produces deprecation warnings that don't concern us
#import nest_asyncio; nest_asyncio.apply() # necessary to run nested async loops in jupyter notebooks

from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from lxml import html

sites_hosting_pubmed_article_count = {}

# Use a variety of agents for our ClientSession to reduce traffic per agent
# This (attempts to) avoid a ban for high traffic from any single agent
# We should really use proxybroker or similar to ensure no ban
user_agents = [
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) Gecko/20100101 Firefox/55.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
        "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
        "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
        "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24"
        ]

def make_header():
    '''
    Chooses a random agent from user_agents with which to construct headers
    :return headers: dict: HTTP headers to use to get HTML from article URL
    '''
    # Make a header for the ClientSession to use with one of our agents chosen at random
    headers = {
            'User-Agent':random.choice(user_agents),
            }
    return headers

async def extract_by_article(url):
    '''
    Extracts all data from a single article
    :param url: string: URL to a single article (i.e. root pubmed URL + PMID)
    :return article_data: Dict: Contains all data from a single article
    '''
    conn = aiohttp.TCPConnector(family=socket.AF_INET)
    headers = make_header()
    # Reference our articles DataFrame containing accumulated data for ALL scraped articles
    global articles_data
    async with aiohttp.ClientSession(headers=headers, connector=conn) as session:
        async with semaphore, session.get(url) as response:
            data = await response.text()
            soup = BeautifulSoup(data, "lxml")
            # Get article abstract if exists - sometimes abstracts are not available (not an error)
            try:
                abstract_raw = soup.find('div', {'class': 'abstract-content selected'}).find_all('p')
                # Some articles are in a split background/objectives/method/results style, we need to join these paragraphs
                abstract = ' '.join([paragraph.text.strip() for paragraph in abstract_raw])
            except:
                abstract = 'NO_ABSTRACT'
            # Get author affiliations - sometimes affiliations are not available (not an error)
            affiliations = [] # list because it would be difficult to split since ',' exists within an affiliation
            try:
                all_affiliations = soup.find('ul', {'class':'item-list'}).find_all('li')
                for affiliation in all_affiliations:
                    affiliations.append(affiliation.get_text().strip())
            except:
                affiliations = 'NO_AFFILIATIONS'
            # Get article keywords - sometimes keywords are not available (not an error)
            try:
                # We need to check if the abstract section includes keywords or else we may get abstract text
                has_keywords = soup.find_all('strong',{'class':'sub-title'})[-1].text.strip()
                if has_keywords == 'Keywords:':
                    # Taking last element in following line because occasionally this section includes text from abstract
                    keywords = soup.find('div', {'class':'abstract' }).find_all('p')[-1].get_text()
                    keywords = keywords.replace('Keywords:','\n').strip() # Clean it up
                else:
                    keywords = 'NO_KEYWORDS'
            except:
                keywords = 'NO_KEYWORDS'
            try:
                title = soup.find('meta',{'name':'citation_title'})['content'].strip('[]')
            except:
                title = 'NO_TITLE'
            authors = ''    # string because it's easy to split a string on ','
            try:
                for author in soup.find('div',{'class':'authors-list'}).find_all('a',{'class':'full-name'}):
                    authors += author.text + ', '
                # alternative to get citation style authors (no first name e.g. I. Zenkov)
                # all_authors = soup.find('meta', {'name': 'citation_authors'})['content']
                # [authors.append(author) for author in all_authors.split(';')]
            except:
                authors = ('NO_AUTHOR')
            try:
                journal = soup.find('meta',{'name':'citation_journal_title'})['content']
            except:
                journal = 'NO_JOURNAL'
            try:
                #date = soup.find('time', {'class': 'citation-year'}).text
                date_text = soup.find('span', class_='cit').text
                parts = date_text.split(';') if ';' in date_text else date_text.split(':')
                date = parts[0]
                print(date)
            except:
                date = 'NO_DATE'

            try:
                raw_text = extract_raw_article_text(soup)
            except:
                raw_text = 'NO_TEXT'

            # Format data as a dict to insert into a DataFrame
            article_data = {
                'url': url,
                'title': title,
                'authors': authors,
                'abstract': abstract,
                'affiliations': affiliations,
                'journal': journal,
                'keywords': keywords,
                'date': date,
                'raw_text': raw_text
            }
            # Add dict containing one article's data to list of article dicts
            articles_data.append(article_data)

# NOTE: Method currently returns a link to the article text, not the raw text yet
def extract_raw_article_text(soup):
    '''
    Extracts raw text from a single pubmed article if it's hosted on PMC, elsevier, or springer.
    :param soup: BeautifulSoup object representing the parsed HTML of the PubMed page
    :return raw_article_text
    '''
    div_full_text_links_list = soup.find('div', class_='full-text-links-list')

    # Dict to store all href attributes, containing (domain, url)
    href_dict = {}

    if div_full_text_links_list:
        # All URL links to articles are within an <a> element
        a_elements = div_full_text_links_list.find_all('a')

        # Some pages haev multiple full-text links (i.e. https://pubmed.ncbi.nlm.nih.gov/38114130/)
        for a_element in a_elements:
            # Extract the href attribute
            href = a_element.get('href')

            # REMOVE: temporarily counting the number of articles a site hosts
            full_domain = get_domain(href)
            sites_hosting_pubmed_article_count[full_domain] = sites_hosting_pubmed_article_count.setdefault(full_domain, 0) + 1

            href_dict[full_domain] = href

        return extract_raw_article_text_helper(href_dict)
    else:
        print("Div with class 'full-text-links-list' not found.")
        return 'NO TEXT'

def extract_raw_article_text_helper(href_dict):
    # Temporarily just returns the site name, will replace with returning full-text of article
    if 'www.ncbi.nlm.nih.gov' in href_dict.keys(): #PMC
        return 'PMC'
        return extract_full_text_PMC(href_dict['www.ncbi.nlm.nih.gov'])
    elif 'linkinghub.elsevier.com' in href_dict.keys(): 
        return 'elsevier'
        #return extract_full_text_elsevier(href_dict['www.ncbi.nlm.nih.gov'])
    elif 'link.springer.com' in href_dict.keys():
        return 'springer'
        #return extract_full_text_springer(href_dict['www.ncbi.nlm.nih.gov'])
    elif 'journals.sagepub.com' in href_dict.keys():
        return 'sagepub'
    elif 'onlinelibrary.wiley.com' in href_dict.keys():
        return 'wiley'
    elif 'www.tandfonline.com' in href_dict.keys():
        return 'tandfonline'
    elif 'jamanetwork.com' in href_dict.keys():
        return 'jamanetwork'
    elif 'www.nature.com' in href_dict.keys():
        return 'nature'
    elif 'journals.lww.com' in href_dict.keys():
        return 'journals.lww'
    elif 'www.mdpi.com' in href_dict.keys():
        return 'mdpi'
    
    return 'NO TEXT (not hosted on top 3 sites)'

def extract_full_text_PMC(url):
    raise NotImplementedError
    
def extract_full_text_elsevier(url):
    # raise NotImplementedError




    #TODO: pass this stuff in the method so it doesn't install everytime lol
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)

    driver.get(url)

    # Wait for a few seconds for the page to load
    time.sleep(5)

    # Get the page source after JavaScript execution
    page_source = driver.page_source

    # Close the browser
    driver.quit()

    print(page_source)

    # Parse the content with BeautifulSoup
    # soup = BeautifulSoup(page_source, 'html.parser')
    # direct_url = soup.find('link', rel='canonical').get('href')
    # extract_full_text_elsevier_directurl(direct_url)
    
    
    # soup.find('article', class_='col-lg-12 col-md-16 pad-left pad-right u-padding-s-top')
    # return temp.text

def extract_full_text_elsevier_directurl(url):
    response = requests.get(url, allow_redirects=True)
    # assert response.status_code == 200

    print(response.text)
    soup = bs4.BeautifulSoup(response.content, 'html.parser')
    temp = soup.find('article', class_='col-lg-12 col-md-16 pad-left pad-right u-padding-s-top')
    return temp

def extract_full_text_springer(url):
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'html.parser')

    sections = soup.find('div', class_='main-content').findAll('div', class_='c-article-section__content')
    
    full_raw_text = ''

    for section in sections:
        paragraphs = section.find_all('p')
    
        # Filter out <p> elements that do not have any attributes or classes
        filtered_paragraphs = [p for p in paragraphs if not p.attrs]

        # Iterate over each paragraph within the section
        for paragraph in filtered_paragraphs:
            full_raw_text += paragraph.text + '\n'

    return full_raw_text

def get_domain(url):
    '''
    Extracts the domain name from a given URL
    :param url: str: The URL to extract the domain name from
    :return domain: str: The domain name of the URL
    '''
    parsed_url = urlparse(url)
    domain = parsed_url.netloc

    if domain == 'doi.org' or domain == 'dx.doi.org':
        # Access the URL to see what site is hosting
        try:
            response = requests.head(url, allow_redirects=True)
            hosting_domain = urlparse(response.url).netloc
            return hosting_domain
        except Exception as e:
            print(f"Error accessing URL: {e}")
            return None
    else:
        return domain

# TODO: make methods all asynchrnous (asyncronous version of get_domain doesn't work)
# async def get_domain(url):
#     '''
#     Extracts the domain name from a given URL
#     :param url: str: The URL to extract the domain name from
#     :return domain: str: The domain name of the URL
#     '''
#     parsed_url = urlparse(url)
#     domain = parsed_url.netloc

#     if domain == 'doi.org' or domain == 'dx.doi.org':
#         return await get_domain_doi(url)
#     else:
#         return domain
    
# async def get_domain_doi(url):
#     # Access the URL to see what site is hosting
#     try:
#         async with aiohttp.ClientSession() as session:
#             async with session.head(url, allow_redirects=True) as response:
#                     hosting_domain = urlparse(response.url).netloc
#                     return hosting_domain
#     except Exception as e:
#         print(f"Error accessing URL: {e}")
#     return None

async def get_pmids(page, keyword):
    """
    Extracts PMIDs of all articles from a pubmed search result, page by page,
    builds a url to each article, and stores all article URLs in urls: List[string]
    :param page: int: value of current page of a search result for keyword
    :param keyword: string: current search keyword
    :return: None
    """
    # URL to one unique page of results for a keyword search
    page_url = f'{pubmed_url}+{keyword}+&page={page}'
    headers = make_header()
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(page_url) as response:
            data = await response.text()
            # Parse the current page of search results from the response
            soup = BeautifulSoup(data, "lxml")
            # Find section which holds the PMIDs for all articles on a single page of search results
            pmids = soup.find('meta',{'name':'log_displayeduids'})['content']
            # alternative to get pmids: page_content = soup.find_all('div', {'class': 'docsum-content'}) + for line in page_content: line.find('a').get('href')
            # Extract URLs by getting PMIDs for all pubmed articles on the results page (default 10 articles/page)
            for pmid in pmids.split(','):
                url = root_pubmed_url + '/' + pmid
                urls.append(url)

def get_num_pages(keyword):
    '''
    Gets total number of pages returned by search results for keyword
    :param keyword: string: search word used to search for results
    :return: num_pages: int: number of pages returned by search results for keyword
    '''
    # Return user specified number of pages if option was supplied
    if args.pages != None: return args.pages

    # Get search result page and wait a second for it to load
    # URL to the first page of results for a keyword search
    headers=make_header()
    search_url = f'{pubmed_url}+{keyword}'
    with requests.get(search_url,headers=headers) as response:
        data = response.text
        soup = BeautifulSoup(data, "lxml")
        num_pages = int((soup.find('span', {'class': 'total-pages'}).get_text()).replace(',',''))
        return num_pages # Can hardcode this value (e.g. 10 pages) to limit # of articles scraped per keyword

async def build_article_urls(keywords):
    """
    PubMed uniquely identifies articles using a PMID
    e.g. https://pubmed.ncbi.nlm.nih.gov/32023415/ #shameless self plug :)
    Any and all articles can be identified with a single PMID

    Async wrapper for get_article_urls, page by page of results, for a single search keyword
    Creates an asyncio task for each page of search result for each keyword
    :param keyword: string: search word used to search for results
    :return: None
    """
    tasks = []
    for keyword in keywords:
        num_pages = get_num_pages(keyword)
        for page in range(1,num_pages+1):
            task = asyncio.create_task(get_pmids(page, keyword))
            tasks.append(task)

    await asyncio.gather(*tasks)

async def get_article_data(urls):
    """
    Async wrapper for extract_by_article to scrape data from each article (url)
    :param urls: List[string]: list of all pubmed urls returned by the search keyword
    :return: None
    """
    tasks = []
    for url in urls:
        if url not in scraped_urls:
            task = asyncio.create_task(extract_by_article(url))
            tasks.append(task)
            scraped_urls.append(url)

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # Set options so user can choose number of pages and publication date range to scrape, and output file name
    parser = argparse.ArgumentParser(description='Asynchronous PubMed Scraper')
    parser.add_argument('--pages', type=int, default=None, help='Specify number of pages to scrape for EACH keyword. Each page of PubMed results contains 10 articles. \n Default = all pages returned for all keywords.')
    parser.add_argument('--start', type=int, default=2019, help='Specify start year for publication date range to scrape. Default = 2019')
    parser.add_argument('--stop', type=int, default=2020, help='Specify stop year for publication date range to scrape. Default = 2020')
    parser.add_argument('--output', type=str, default='articles.csv',help='Choose output file name. Default = "articles.csv".')
    args = parser.parse_args()
    if args.output[-4:] != '.csv': args.output += '.csv' # ensure we save a CSV if user forgot to include format in --output option
    start = time.time()
    # This pubmed link is hardcoded to search for articles from user specified date range, defaults to 2019-2020
    pubmed_url = f'https://pubmed.ncbi.nlm.nih.gov/?term={args.start}%3A{args.stop}%5Bdp%5D'
    # The root pubmed link is used to construct URLs to scrape after PMIDs are retrieved from user specified date range
    root_pubmed_url = 'https://pubmed.ncbi.nlm.nih.gov'
    # Construct our list of keywords from a user input file to search for and extract articles from
    search_keywords = []

    # Get full path to keywords.txt
    current_directory = os.path.dirname(os.path.realpath(__file__))
    path_to_keywords = os.path.join(current_directory, 'keywords.txt')

    with open(path_to_keywords) as file:
        keywords = file.readlines()
        [search_keywords.append(keyword.strip()) for keyword in keywords]
    print(f'\nFinding PubMed article URLs for {len(keywords)} keywords found in keywords.txt\n')
    # Empty list to store all article data as List[dict]; each dict represents data from one article
    # This approach is considerably faster than appending article data article-by-article to a DataFrame
    articles_data = []
    # Empty list to store all article URLs
    urls = []
    # Empty list to store URLs already scraped
    scraped_urls = []

    # We use asyncio's BoundedSemaphore method to limit the number of asynchronous requests
    #    we make to PubMed at a time to avoid a ban (and to be nice to PubMed servers)
    # Higher value for BoundedSemaphore yields faster scraping, and a higher chance of ban. 100-500 seems to be OK.
    semaphore = asyncio.BoundedSemaphore(100)

    # Get and run the loop to build a list of all URLs
    loop = asyncio.get_event_loop()
    loop.run_until_complete(build_article_urls(search_keywords))
    print(f'Scraping initiated for {len(urls)} article URLs found from {args.start} to {args.stop}\n')
    # Get and run the loop to get article data into a DataFrame from a list of all URLs
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_article_data(urls))

    # Create DataFrame to store data from all articles
    articles_df = pd.DataFrame(articles_data, columns=['title','abstract','affiliations','authors','journal','date','keywords','url','raw_text'])
    print('Preview of scraped article data:\n')
    print(articles_df.head(5))
    # Save all extracted article data to CSV for further processing
    filename = args.output
    articles_df.to_csv(filename)
    print(f'It took {time.time() - start} seconds to find {len(urls)} articles; {len(scraped_urls)} unique articles were saved to {filename}')


    # REMOVE BELOW
    print()
    sorted_site_ranks = sorted(sites_hosting_pubmed_article_count.items(), key=lambda x: x[1], reverse=True)
    #print(f"total articles: {len(sorted_site_ranks)}")
    total_articles = 0
    for key, value in sorted_site_ranks:
        print(f"{key}: {value}")
        total_articles += value
    print(f"total articles: {total_articles}")