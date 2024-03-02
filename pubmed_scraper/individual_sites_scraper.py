from collections import OrderedDict
import time
from bs4 import BeautifulSoup, NavigableString
import bs4
import pandas as pd
import requests
import warnings

from async_pubmed_scraper import make_header; warnings.filterwarnings('ignore') # aiohttp produces deprecation warnings that don't concern us
#import nest_asyncio; nest_asyncio.apply() # necessary to run nested async loops in jupyter notebooks

from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from lxml import html

def extract_full_text_PMC(url):
    raise NotImplementedError

def extract_full_text_springer(url):
    response = requests.get(url)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'html.parser')

    full_raw_text = ''

    sections = soup.findAll('div', class_='c-article-section__content')

    for section in sections:
        paragraphs = section.find_all('p')
    
        # Filter out <p> elements that do not have any attributes or classes
        filtered_paragraphs = [p for p in paragraphs if not p.attrs]

        # Iterate over each paragraph within the section
        for paragraph in filtered_paragraphs:
            full_raw_text += paragraph.text + '\n'

    return full_raw_text

def extract_full_text_jama(url):
    headers = make_header()
    response = requests.get(url, headers=headers)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'html.parser')

    full_raw_text = ''

    paragraphs = soup.find('div', class_='article-full-text').findAll('p', class_='para')
    full_raw_text = ' '.join([paragraph.text for paragraph in paragraphs])

    return full_raw_text

def extract_summary_full_text_journalslww(url):
    # Full_text behind paywall, only grabs the summary
    headers = make_header()
    response = requests.get(url, headers=headers)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'html.parser')

    paragraphs = soup.find('div', class_='ejp-article-text-abstract').findAll('p')

    full_raw_text = ' '.join([paragraph.text for paragraph in paragraphs])

    return full_raw_text

def extract_full_text_mdpi(url):
    headers = make_header()
    response = requests.get(url, headers=headers)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'html.parser')

    # Scrape abstract
    full_raw_text = soup.find('section', class_='html-abstract').find('div', class_='html-p').text + '\n'

    # Scrape main body
    all_paragraphs = soup.find('div', class_='html-body').findAll('div', class_='html-p')

    for paragraph in all_paragraphs:
        for element in paragraph.contents:
            if element.name != 'div' or element.attrs['class'] != ['html-disp-formula-info']: # Avoids scraping the equations
                full_raw_text += element.text + ' '

    return full_raw_text

# THESE SITES WERE ATTEMPTED TO BE SCRAPPED BUT DIDN'T WORK (all 403 forbidden errors)
def extract_full_text_elsevier(url):
    raise NotImplementedError #403 error

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

def extract_full_text_wiley(url):
    raise NotImplementedError #403 error
    headers = make_header()
    #headers = {"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1"}
    response = requests.get(url, headers=headers)
    assert response.status_code == 200
    soup = bs4.BeautifulSoup(response.content, 'html.parser')

    full_raw_text = ''

    sections = soup.find('div', class_="article-section article-section__full").findAll('div', class_="article-section__content")

    for section in sections:
        # Grab title of section first
        # Find the <h2> tag with specific attributes
        h2_tag = section.find('h2', class_='article-section__title section__title section1')

        # Extract the title from the <h2> tag
        if h2_tag:
            section_title = h2_tag.text  # Use strip() to remove leading/trailing whitespace
            full_raw_text += section_title

        paragraphs = section.find_all('p')

        for para in paragraphs:
            full_raw_text += para.get_text(separator=' ', strip=False) + '\n'

    return full_raw_text