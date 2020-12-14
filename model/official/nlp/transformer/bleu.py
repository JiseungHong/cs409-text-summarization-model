# -*- coding: UTF-8 -*-
"""
bleu.py
~~~~~~
이 스크립트는 BLEU 점수를 계산하기 위한 평가 함수들을 정의하고 있다.

작성자: 이호창 (hochang@ncsoft.com)
작성일자: 2020.04.20 월요일
"""

# Python compatible
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard library
import collections
import math
import os
import re
import sys
import subprocess
import unicodedata

# Third-party libraries
# pylint: disable=redefined-builtin
from six.moves import range
from six.moves import zip
# pylint: enable=redefined-builtin

import numpy as np
import six

# 현재 소스 디렉토리 경로
here = os.path.dirname(__file__)


# BLEU 점수 관련 함수들
# =========================================================

def compute_bleu(reference_corpus,
                 translation_corpus,
                 max_order=4,
                 use_bp=True):
    """하나 또는 그 이상의 정답들에 대한 번역 세그먼트의 BLEU 점수를 계산한다.

    :param reference_corpus: 각 번역을 위한 정답들 목록 (list)
    :param translation_corpus: 점수를 계산할 번역 목록 (list)
    :param max_order: BLEU 점수를 계산할 때 사용할 최대 ngram 오더 (int)
    :param use_bp: brevity penalty를 적용 (bool)
    :returns: BLEU 점수 (float)
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []

    for (references, translations) in zip(reference_corpus, translation_corpus):
        reference_length += len(references)
        translation_length += len(translations)
        ref_ngram_counts = _get_ngrams(references, max_order)
        translation_ngram_counts = _get_ngrams(translations, max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram)-1] += translation_ngram_counts[ngram]

    precisions = [0] * max_order
    smooth = 1.0

    for i in range(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            else:
                smooth *= 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum/max_order)

    if use_bp:
        if not reference_length:
            bp = 1.0
        else:
            ratio = translation_length / reference_length
            if ratio <= 0.0:
                bp = 0.0
            elif ratio >= 1.0:
                bp = 1.0
            else:
                bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    return np.float32(bleu)


def _get_ngrams(segment, max_order):
    """입력 세그먼트로부터 주어진 최대 오더의 모든 ngram을 추출한다.

    :param segment: ngram으로 추출된 텍스트 세그먼트 (list)
    :param max_order: 토큰 내 최대 길이 (int)
    :returns: 세그먼트 내 최대 오더까지 모든 ngram을 포함하는 카운터 (collections.Counter)
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts



# 실제 BLEU 점수 측정 함수들
# =========================================================

def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
    """(정답과 예측) 두 파일을 위한 BLEU 점수를 계산한다."""
    ref_lines = native_to_unicode(
        open(ref_filename, "r").read()).split("\n")
    hyp_lines = native_to_unicode(
        open(hyp_filename, "r").read()).split("\n")
    assert len(ref_lines) == len(hyp_lines), ("{} != {}".format(
        len(ref_lines), len(hyp_lines)))
    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]
    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]
    return compute_bleu(ref_tokens, hyp_tokens)


class UnicodeRegex(object):
    """모든 특수문자와 심볼들을 인식하기 위한 Ad-hoc 해킹"""

    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
        return "".join(
            six.unichr(x) for x in range(sys.maxunicode)
            if unicodedata.category(six.unichr(x)).startswith(prefix))

uregex = UnicodeRegex()


def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.
    See https://github.com/moses-smt/mosesdecoder/"
            "blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).
    Note that a number (e.g. a year) followed by a dot at the end of sentence
    is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.

    :param string: 입력 문자열 (str)
    :returns: 토큰 목록 (list[str])
    """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()


def is_unicode(s):
    return isinstance(s, six.text_type)


def to_unicode(s, ignore_errors=False):
    if is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("UTF-8", errors=error_mode)


def native_to_unicode(s):
    if is_unicode(s):
        return s
    try:
        return to_unicode(s)
    except UnicodeDecodeError:
        res = to_unicode(s, ignore_errors=True)
        logging.info("Ignoring Unicode error, outputting: %s" % res)
        return res


def unicode_to_native(s):
    if six.PY2:
        return s.encode("UTF-8") if is_unicode(s) else s
    else:
        return s



# 메인함수
# =========================================================

def main(argv=sys.argv):
    """Main Function."""
    if len(argv) != 3:
        print("usage: python3 {} <정답> <예측>".format(argv[0]))
        return -1

    bleu = bleu_wrapper(argv[1], argv[2])
    print("BLEU score: {}".format(bleu))

if __name__ == '__main__':
    sys.exit(main())

