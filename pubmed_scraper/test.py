from async_pubmed_scraper import extract_full_text_springer, extract_full_text_elsevier, extract_full_text_PMC

res1 = extract_full_text_springer('https://link.springer.com/article/10.1007/s40596-020-01307-9')
print(res1)
print()
res2 = extract_full_text_springer('https://link.springer.com/article/10.1007/s00787-019-01441-2')
print(res2)

# Cannot get elsevier url's for some reason
# res3 = extract_full_text_elsevier('https://www.sciencedirect.com/science/article/pii/S2215036622000360')
# print(res3)
# print()
# res4 = extract_full_text_elsevier('https://www.sciencedirect.com/science/article/pii/S2352250X21000245')
# print(res4)