# Question_Generation (in progress)
Code from hugging_face transformers 
In this code I will use KoBERT model, not DistilBERT model 

** Code will be modified to make it possible to choose between both models from command line

## Functions included

## load_model.py

**Included functions**

get_distilkobert_model()
get_kobert_model() 

**Imported modules**

transformer - BERT, DistilBERT
tokenization.py

## tokenization.py

**Included functions**

**Imported modules**

## preprocess.py

**Included functions**

**Imported modules**

## kobert_implement.py

**Included functions**

**Imported modules**


### Reference
- [Huggingface Transformers document]("https://huggingface.co/transformers/model_doc/bert.html#transformers.BertModel")
- [DistilKoBERT]("https://github.com/monologg/DistilKoBERT/tree/b9838f4d0fd1a73204dabf20e1bd4cc560ad9555")
- [KorQuAD-beginner]("https://github.com/graykode/KorQuAD-beginner")
- *A Recurrent BERT-based Model for Question Generation*, *2019*, Chan et al.
- [BERT has a Mouth, and It Must Speak: BERT as a Markov Random Field Language Model, Wang et al., 2019]("https://arxiv.org/abs/1902.04094") 