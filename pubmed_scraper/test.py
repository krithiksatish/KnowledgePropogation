from individual_sites_scraper import *

# # SPRINGER TESTS - works!
# print(extract_full_text_springer('https://link.springer.com/article/10.1007/s40596-020-01307-9'))
# print()
# print(extract_full_text_springer('https://link.springer.com/article/10.1007/s00787-019-01441-2'))

# # JOURNAL.LWW TESTS - works (kinda, full text locked behind paywall)
# print(extract_summary_full_text_journalslww('https://doi.org/10.1097/ANS.0000000000000221'))
# print(extract_summary_full_text_journalslww('https://doi.org/10.1097/YCO.0000000000000695'))
# print(extract_summary_full_text_journalslww('https://doi.org/10.1097/CEH.0000000000000197'))

# # Jama.network TESTS - works!
# print(extract_full_text_jama('https://jamanetwork.com/journals/jama/fullarticle/10.1001/jama.2023.12900'))
# print(extract_full_text_jama('https://jamanetwork.com/journals/jamapsychiatry/fullarticle/10.1001/jamapsychiatry.2022.3391')) # This site only hosts abstract, scraper still works though but doesn't get raw text

# # MDPI TESTS - works!
# print(extract_full_text_mdpi('https://www.mdpi.com/resolver?pii=ijerph17249338'))
# print(extract_full_text_mdpi('https://www.mdpi.com/resolver?pii=ijerph17249332'))

# THESE DON"T WORK

# WILEY TESTS - doesn't work
#print(extract_full_text_wiley('https://doi.org/10.1111/tct.12504'))
#print(extract_full_text_wiley('https://doi.org/10.1111/ped.14137'))

# ELSEVIER TESTS - doesn't work
# print(extract_full_text_elsevier('https://linkinghub.elsevier.com/retrieve/pii/S2215-0366(22)00036-0'))
# print()
# print(extract_full_text_elsevier('https://www.sciencedirect.com/science/article/pii/S2215036622000360'))
# print(extract_full_text_elsevier('https://linkinghub.elsevier.com/retrieve/pii/S2352-250X(21)00024-5'))
# print(res4)