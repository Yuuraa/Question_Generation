from transformers import DistilBertModel, BertModel
from tokenization import KoBertTokenizer

# Get pre-trained models
def get_distilkobert_model():
    model = DistilBertModel.from_pretrained('monologg/distilkobert')
    return model


def get_kobert_model():
    model = BertModel.from_pretrained('monologg/kobert')
    return model

def get_mlmkobert_model():
    model = BertForMaskedLM.from_pretrained('monologg/kobert')
    return model

def get_tokenizer():
    return KoBertTokenizer.from_pretrained('monologg/kobert')
