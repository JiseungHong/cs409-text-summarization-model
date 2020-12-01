"""from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sent = ("휴일인 오늘도 서쪽을 중심으로 폭염이 이어졌는데요, 내일은 반가운 비 소식이 있습니다.", 
    "폭염을 피해서 휴일에 놀러왔다가 갑작스런 비로 인해 망연자실하고 있습니다.")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sent) #문장 벡터화 진행
print(tfidf_vectorizer.vocabulary_)
print(tfidf_matrix.shape[0])
print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]))"""
'''
from konlpy.tag import Okt
okt = Okt()

sent = '이것(가십거리]도 되나요?'
morphemes = okt.pos(sent, norm=True, stem=False)
print(morphemes)

list = [x[0] for x in morphemes if x[1] not in ['Josa', 'Punctuation']]
target = " ".join(list)
print(target)
'''
'''
import re

preprocessed_title = '안녕 내 이름은      ***이다.'
preprocessed_title = re.compile(r"\s+").sub(" ", preprocessed_title)
print(preprocessed_title)
'''