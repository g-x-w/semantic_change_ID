# Identifying Sources of Semantic Change
This is a repository for code to be used in the research project into a computational method for identifying sources of semantic change carried out during the Summer of 2020 by Gary Wei with the Cognitive Lexicon Lab (CLL) in University of Toronto.

Contribution and push access will be granted to Professor Yang Xu and John Xu of the CLL.

## Dataset:
Looking at using a dataset of primary sources to track frequency of target words across different sources and timeframes during the SARS-CoV-2 pandemic. The dataset is available from: 

https://blog.aylien.com/free-coronavirus-news-dataset/

and is formatted as a JSON lines file. It consists of 528848 articles from ~400 global sources dating from November 1st 2019 to April 5th 2020. An example of one of the unwrapped source lines:

    {
    'author': {'avatar_url': None, 'id': 973106, 'name': 'Gavin Evans'}, 

    'body': 'On Sunday, British Prime Minister Boris Johnson was hospitalized "for tests" because of "persistent" COVID-19 symptoms\xa010 days\xa0after he tested positive, CNN reports.\xa0\nJohnson reportedly went to the unspecified London hospital after his doctor advised him to do so. A press release from his office called the\xa0move\xa0"precautionary."\xa0\nOn March 26, Johnson revealed he had tested positive and that he had been dealing with symptoms since that date. Britain had gone into lockdown two days earlier.\nSince the 26th, Johnson has been quarantined at his Downing Street residence. He is the first known world leader to have contracted the virus.\xa0\nRoughly a month ago, right around the time the U.K. started dealing with an outbreak, Johnson garnered media coverage for saying he\'d shook hands with coronavirus patients during a hospital visit. \xa0\n"I shook hands with everybody, you will be pleased to know, and I continue to shake hands," Johnson said during a press conference that took place on March 3. His positive test was registered 23 days later.\xa0\nOn Saturday, Johnson\'s fiancée, Carrie Symonds, tweeted out that she\'d spent a week in bed with coronavirus symptoms. She had not officially been tested for the disease, but said she felt "stronger" and "on the mend" following the week of rest:', 

    'categories': [{'confident': True, 'id': 'IAB7-3', 'level': 2, 'links': {'_self': 'https://api.aylien.com/api/v1/classify/taxonomy/iab-qag/IAB7-3', 'parent': 'https://api.aylien.com/api/v1/classify/taxonomy/iab-qag/IAB7'}, 'score': 0.11, 'taxonomy': 'iab-qag'}, {'confident': True, 'id': 'IAB7', 'level': 1, 'links': {'_self': 'https://api.aylien.com/api/v1/classify/taxonomy/iab-qag/IAB7', 'parent': None}, 'score': 0.09, 'taxonomy': 'iab-qag'}, {'confident': True, 'id': '07003004', 'level': 3, 'links': {'_self': 'https://api.aylien.com/api/v1/classify/taxonomy/iptc-subjectcode/07003004', 'parent': 'https://api.aylien.com/api/v1/classify/taxonomy/iptc-subjectcode/07003000'}, 'score': 0.13, 'taxonomy': 'iptc-subjectcode'}], 

    'characters_count': 1288, 

    'entities': {'body': [{'indices': [[34, 46]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Boris_Johnson'}, 'score': 1.0, 'text': 'Boris Johnson', 'types': ['Agent', 'OfficeHolder', 'Person', 'Politician']}, {'indices': [[1092, 1098]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Twitter'}, 'score': 0.9999861717224121, 'text': 'tweeted', 'types': ['Company', 'Agent', 'Organisation', 'Work', 'Product', 'Website', 'Service']}, {'indices': [[153, 155]], 'links': {'dbpedia': 'http://dbpedia.org/resource/CNN'}, 'score': 0.9998425841331482, 'text': 'CNN', 'types': ['Agent', 'TelevisionStation', 'Channel', 'Cable', 'Organisation', 'Broadcaster']}, {'indices': [[1076, 1081]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Carrie_Mathison'}, 'score': 0.5159270167350769, 'text': 'Carrie', 'types': ['Agent', 'FictionalCharacter', 'Person', 'Character']}, {'indices': [[466, 473]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Lockdown'}, 'score': 0.9477183222770691, 'text': 'lockdown', 'types': ['Book', 'Product', 'Definitions']}, {'indices': [[629, 633]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Virus'}, 'score': 0.993719756603241, 'text': 'virus', 'types': ['Animal', 'Agent', 'Species']}, {'indices': [[217, 224], [818, 825]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Hospital'}, 'score': 0.9999241232872009, 'text': 'hospital', 'types': ['University', 'PersonFunction', 'Institution']}, {'indices': [[11, 32]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Prime_Minister_of_the_United_Kingdom'}, 'score': 0.9976106286048889, 'text': 'British Prime Minister', 'types': ['OfficeHolder', 'Head', 'Person']}, {'indices': [[210, 224]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Royal_London_Hospital'}, 'score': 0.9988574981689453, 'text': 'London hospital', 'types': ['University', 'Hospital', 'Building', 'ArchitecturalStructure', 'Location', 'Place']}, {'indices': [], 'links': {'dbpedia': 'http://dbpedia.org/resource/United_Kingdom'}, 'score': 0.9997988939285278, 'text': 'U.K.', 'types': ['Country', 'State', 'Location', 'Place', 'PopulatedPlace']}, {'indices': [[788, 798], [1140, 1150]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Coronavirus'}, 'score': 1.0, 'text': 'coronavirus', 'types': ['Eukaryote', 'Species']}, {'indices': [[210, 215]], 'links': None, 'score': None, 'text': 'London', 'types': ['Place']}, {'indices': [[444, 450]], 'links': None, 'score': None, 'text': 'Britain', 'types': ['Place']}, {'indices': [[684, 686]], 'links': None, 'score': None, 'text': 'U.K', 'types': ['Place']}, {'indices': [[40, 46], [167, 173], [344, 350], [509, 515], [723, 729], [928, 934], [1057, 1063]], 'links': None, 'score': None, 'text': 'Johnson', 'types': ['Person']}, {'indices': [[1076, 1089]], 'links': None, 'score': None, 'text': 'Carrie Symonds', 'types': ['Person']}, {'indices': [[119, 125]], 'links': None, 'score': None, 'text': '10 days', 'types': ['Organisation']}], 'title': [{'indices': [[0, 21]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Prime_Minister_of_the_United_Kingdom'}, 'score': 0.9976551532745361, 'text': 'British Prime Minister', 'types': ['OfficeHolder', 'Head', 'Person']}, {'indices': [[23, 35]], 'links': {'dbpedia': 'http://dbpedia.org/resource/Boris_Johnson'}, 'score': 1.0, 'text': 'Boris Johnson', 'types': ['Agent', 'OfficeHolder', 'Person', 'Politician']}, {'indices': [[64, 81]], 'links': None, 'score': None, 'text': 'COVID-19 Diagnosis', 'types': ['Person']}]}, 
    
    'hashtags': ['#PrimeMinisterOfTheUnitedKingdom', '#BorisJohnson', '#Coronavirus', '#CNN', '#CNN', '#RoyalLondonHospital', '#Lockdown', '#DowningStreet', '#Virus', '#UnitedKingdom', '#Hospital', '#CarrieMathison', '#Twitter'], 

    'id': 74199025, 
    
    'keywords': ['Johnson', 'Hospitalized', 'Boris', 'Minister', 'Days', 'Prime', 'COVID-19', 'British', 'Diagnosis', 'British Prime Minister', 'Boris Johnson', 'coronavirus symptoms', 'symptoms', 'days', 'positive', 'press', 'tests', 'March', 'coronavirus', 'hospital', 'hands', 'time', 'virus', 'month', 'residence', 'leader', 'tweeted', 'CNN', 'Carrie', 'lockdown', 'London hospital', 'U.K.', 'Downing Street'], 

    'language': 'en', 

    'links': {'canonical': None, 'coverages': '/coverages?story_id=74199025', 'permalink': 'https://www.complex.com/life/2020/04/boris-johnson-hospitalized-coronavirus', 'related_stories': '/related_stories?story_id=74199025'}, 

    'media': [{'content_length': 520367, 'format': 'GIF', 'height': 675, 'type': 'image', 'url': 'https://images.complex.com/complex/images/c_fill,f_auto,g_center,w_1200/fl_lossy,q_70/jnnmrqnl64mxiljyc1l3/boris-johnson', 'width': 1200}, {'content_length': 1129577, 'format': 'GIF', 'height': 1080, 'type': 'image', 'url': 'https://images.complex.com/complex/images/c_limit,h_1080,w_1920/jnnmrqnl64mxiljyc1l3/boris-johnson', 'width': 1920}], 

    'paragraphs_count': 7, 

    'published_at': '2020-04-05 23:59:42+00:00', 

    'sentences_count': 12, 

    'sentiment': {'body': {'polarity': 'positive', 'score': 0.962804}, 'title': {'polarity': 'neutral', 'score': 0.86303}}, 

    'social_shares_count': {'facebook': [{'count': 97, 'fetched_at': '2020-04-06 18:25:41+00:00'}, {'count': 92, 'fetched_at': '2020-04-06 12:26:08+00:00'}, {'count': 68, 'fetched_at': '2020-04-06 06:26:23+00:00'}, {'count': 15, 'fetched_at': '2020-04-06 00:26:22+00:00'}], 'google_plus': [], 'linkedin': [], 'reddit': [{'count': 0, 'fetched_at': '2020-04-06 18:14:08+00:00'}, {'count': 0, 'fetched_at': '2020-04-06 09:17:25+00:00'}, {'count': 0, 'fetched_at': '2020-04-06 00:19:00+00:00'}]}, 

    'source': {'description': None, 'domain': 'complex.com', 'home_page_url': 'http://www.complex.com/', 'id': 1737, 'links_in_count': None, 'locations': [{'city': 'New York', 'country': 'US', 'state': 'New York'}], 'logo_url': 'https://images.complex.com/complex/image/upload/c_pad,g_west,h_40,w_125/v1464104389/COMPLEX_2015_RGB.png', 'name': 'Complex', 'rankings': {'alexa': [{'country': None, 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 4137}, {'country': 'AO', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 6499}, {'country': 'IN', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 10579}, {'country': 'PH', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 2797}, {'country': 'ES', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 14605}, {'country': 'US', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 1177}, {'country': 'IE', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 2584}, {'country': 'AU', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 2148}, {'country': 'NO', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 1836}, {'country': 'NG', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 3365}, {'country': 'SE', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 3524}, {'country': 'GB', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 2423}, {'country': 'FR', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 7755}, {'country': 'BR', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 17651}, {'country': 'ZA', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 2683}, {'country': 'DK', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 1581}, {'country': 'NL', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 5918}, {'country': 'CA', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 1424}, {'country': 'JP', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 21410}, {'country': 'CN', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 20214}, {'country': 'DE', 'fetched_at': '2019-06-06 16:06:34+00:00', 'rank': 8293}]}, 'scopes': [{'city': None, 'country': None, 'level': 'international', 'state': None}], 'title': None}, 
    
    'summary': {'sentences': ['On Sunday, British Prime Minister Boris Johnson was hospitalized "for tests" because of "persistent" COVID-19 symptoms\xa010 days\xa0after he tested positive, CNN reports.', 'On March 26, Johnson revealed he had tested positive and that he had been dealing with symptoms since that date.', "Roughly a month ago, right around the time the U.K. started dealing with an outbreak, Johnson garnered media coverage for saying he'd shook hands with coronavirus patients during a hospital visit.", '"I shook hands with everybody, you will be pleased to know, and I continue to shake hands," Johnson said during a press conference that took place on March 3.', "On Saturday, Johnson's fiancée, Carrie Symonds, tweeted out that she'd spent a week in bed with coronavirus symptoms."]}, 

    'title': 'British Prime Minister Boris Johnson Hospitalized 10 Days After COVID-19 Diagnosis', 

    'words_count': 218
    }

## Code Structure
All script functions are annotated with docstrings for ease of following; `jsonl_nav.py` pulls specific values from a given key in the dataset, facilitating manual navigation of the set for understanding. The main script `dataset_processing.py` processes the full dataset and outputs a dictionary in the format:

    {
    date1: {
        source1: {
            total count: {word1: count, word2: count, ..., wordn: count},
            article1: {time: '16:00:00', word1: count, word2: count, ..., wordn: count},
            article2: {time: '16:00:00', word1: count, word2: count, ..., wordn: count},
            ...
            articlen: {time: '16:00:00', word1: count, word2: count, ..., wordn: count}
        source2: {time: '16:00:00', word1: count, word2: count, ..., wordn: count},
        ...
        sourcen: {time: '16:00:00', word1: count, word2: count, ..., wordn: count}
        }
    date2: {
        source1: {time: '16:00:00', word1: count, word2: count, ..., wordn: count}, 
        source2: {time: '16:00:00', word1: count, word2: count, ..., wordn: count},
        ...
        sourcen: {time: '16:00:00', word1: count, word2: count, ..., wordn: count}
        }
    ...
    daten: {
        source1: {time: '16:00:00', word1: count, word2: count, ..., wordn: count}, 
        source2: {time: '16:00:00', word1: count, word2: count, ..., wordn: count},
        ...
        sourcen: {time: '16:00:00', word1: count, word2: count, ..., wordn: count}
        }    
    }

in both a text file and a JSON file. With the size of the aylien dataset being used (~7.6 GB), this output file works out to ~240MB or roughly 2 minute of full HD video.

## Current Stage
Having processed the data for this purpose, currently working on graphical and visual representations of the data in as high resolution as is reasonable. Next steps will be improve usability and parameterization of search in the direction of a utility search tool.