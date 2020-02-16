from math import log
from numpy import array
from numpy import argmax

def Beam_search_decoder(data,k):
    sequences = [[list(),1.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score * -log(row[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key = lambda tup:tup[1])
        sequences = ordered[:k]
    return sequences