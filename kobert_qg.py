from math import log
from numpy import array
from numpy import argmax
from mymodel import BertQuestionGenerator
from load_model import get_kobert_model, get_tokenizer

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

model = BertQuestionGenerator.from_pretrained('monologg/kobert')
# Pytorch가 이 모델을 GPU에서 돌리도록 지정함
model.cuda()

tokenizer = get_tokenizer()
