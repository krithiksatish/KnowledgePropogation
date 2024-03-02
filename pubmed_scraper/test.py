from async_pubmed_scraper import *

# SPRINGER TESTS - works!
# res1 = extract_full_text_springer('https://link.springer.com/article/10.1007/s40596-020-01307-9')
# print(res1)
# print()
# res2 = extract_full_text_springer('https://link.springer.com/article/10.1007/s00787-019-01441-2')
# print(res2)

# WILEY TESTS
#print(extract_full_text_wiley('https://doi.org/10.1111/tct.12504'))
#print(extract_full_text_wiley('https://doi.org/10.1111/ped.14137'))

# ELSEVIER TESTS - doesn't work
# print(extract_full_text_elsevier('https://linkinghub.elsevier.com/retrieve/pii/S2215-0366(22)00036-0'))
# print()
# print(extract_full_text_elsevier('https://www.sciencedirect.com/science/article/pii/S2215036622000360'))
# print(extract_full_text_elsevier('https://linkinghub.elsevier.com/retrieve/pii/S2352-250X(21)00024-5'))
# print(res4)

# JOURNAL.LWW TESTS - works!
# print(extract_full_text_journalslww('https://doi.org/10.1097/ANS.0000000000000221'))
# print()
# print(extract_full_text_journalslww('https://doi.org/10.1097/CEH.0000000000000197'))

# MDPI TESTS - currently implementing
extract_full_text_mdpi('https://www.mdpi.com/resolver?pii=ijerph17249338')