import re
import json
import canrevan.utils as utils
from bs4 import BeautifulSoup, SoupStrainer
from typing import List


def extract_article_urls(document: str) -> List[str]:
    document = document[document.find('<ul class="type06_headline">'):]

    # Extract article url containers.
    list1 = document[:document.find('</ul>')]
    list2 = document[document.find('</ul>') + 5:]
    list2 = list2[:list2.find('</ul>')]

    document = list1 + list2

    # Extract all article urls from the containers.
    article_urls = []
    while '<dt>' in document:
        document = document[document.find('<dt>'):]
        container = document[:document.find('</dt>')]

        if not container.strip():
            continue

        article_urls.append(
            re.search(r'<a href="(.*?)"', container).group(1))
        document = document[document.find('</dt>'):]

    return article_urls


def parse_article_content(document: str) -> str:
    titlestrainer = SoupStrainer('h3', attrs={'id': 'articleTitle'})
    strainer = SoupStrainer('div', attrs={'id': 'articleBodyContents'})
    titleDocument = BeautifulSoup(document, 'lxml', parse_only=titlestrainer)
    contentDocument = BeautifulSoup(document, 'lxml', parse_only=strainer)
    title = titleDocument.find('h3')
    content = contentDocument.find('div')
    # Skip invalid articles which do not contain news contents.
    if content is None:
        raise ValueError('there is no any news article content.')
    # Remove unnecessary tags except `<br>` elements for preserving line-break
    # characters.
    for child in title.find_all():
        if child.name != 'br':
            child.clear()
    for child in content.find_all():
        if child.name != 'br':
            child.clear()
    title = title.get_text(separator='\n').strip()

    content = content.get_text(separator='\n').strip()
    content = '\n'.join([line.strip() for line in content.splitlines()
                         if line.strip()])

    # Skip the contents which contain too many non-Korean characters.
    if utils.korean_character_ratio(content) < 0.5:
        raise ValueError('there are too few Korean characters in the content.')

    # Normalize the contents by removing abnormal sentences.
    content = ' '.join([
        line for line in content.splitlines()
        if utils.is_normal_character(line[0]) and line[-1] == '.'])
    if len(title) == 0 or len(content) == 0:
        raise ValueError('there is no any news article content.')
    return [title, content]
