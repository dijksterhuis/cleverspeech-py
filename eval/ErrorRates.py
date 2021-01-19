def sentence_error_rate(decoding, target):
    if decoding == target:
        return 1.0
    else:
        return 0.0


def word_error_rate(decoding, target):
    """
    Calculates the WER between a and b.
    This is taken directly form `DeepSpeech.util.text.wer`.

    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER ist calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:
    target = target.split("=")
    decoding = decoding.split("=")
    return levenshtein(target, decoding) / float(len(target))


def character_error_rate(decoding, target):
    return levenshtein(decoding, target)


def levenshtein(a, b):

    """
    Calculates the Levenshtein distance between a and b.
    This is taken directly form `DeepSpeech.util.text.levenshtein`.

    # The following code is from: http://hetland.org/coding/python/levenshtein.py

    # This is a straightforward implementation of a well-known algorithm, and thus
    # probably shouldn't be covered by copyright to begin with. But in case it is,
    # the author (Magnus Lie Hetland) has, to the extent possible under law,
    # dedicated all copyright and related and neighboring rights to this software
    # to the public domain worldwide, by distributing it under the CC0 license,
    # version 1.0. This software is distributed without any warranty. For more
    # information, see <http://creativecommons.org/publicdomain/zero/1.0>

    :param a: string A
    :param b: string B
    :return: levenshtein distance
    """

    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = list(range(n+1))
    for i in range(1, m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1, n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]



