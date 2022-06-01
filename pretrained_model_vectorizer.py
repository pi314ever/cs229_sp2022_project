import transformers
import torch

#*** pretrained_model_vectorizer.py
# Library which includes all code that involves the use of pretrained
# models for contextualization
# Functions:
#    vectorize_with_pretrained_embeddings(sentences, labels): Given a list of text examples,
#    produces embeddings of dim 768 for each example
#***

def vectorize_with_pretrained_embeddings(sentences):
  """
  Produces a tensor containing a BERT embedding for each sentence in the dataset or in a
  batch
  Args:
    sentences: List of sentences of length n
  Returns:
    embeddings: A 2D torch array containing embeddings for each of the n sentences (n x d)
                where d = 768
  """

  tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')
  pretrained_model = transformers.BertModel.from_pretrained('bert-base-cased', output_hidden_states=False)
  pretrained_model.eval()
  embeddings = []
  for sentence in sentences:
    with_tags = "[CLS]" + sentence + "[SEP]"
    tokenized_sentence = tokenizer.tokenize(with_tags)
    indices_from_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    segments_ids = [1] * len(indices_from_tokens)
    tokens_tensor = torch.tensor([indices_from_tokens])
    segments_tensors = torch.tensor([segments_ids])
    with torch.no_grad():
      outputs = pretrained_model(tokens_tensor, segments_tensors)[0] #The output is the
      #last hidden state of the pretrained model of shape 1 x sentence_length x BERT embedding_length
      embeddings.append(torch.mean(outputs, dim = 1))# we average across the embedding length
      #dimension to produce constant sized tensors
  print(embeddings[0].shape)
  embeddings = torch.cat(embeddings, dim = 0)
  print('Shape of embeddings tensor (n x d = 768): ', embeddings.shape)
  return embeddings

if __name__=="__main__":

  #Test sentences. To test, run 'python pretrained_model_vectorizer.py' at the command line
  sentences = ["The boy is running", "The dog has been barking for the whole evening"]
 
  #To use with real data, import pretrained_model_vectorizer, then call
  #'vectorize_with_pretrained_embeddings' on your list of sentences to embed
  vectorize_with_pretrained_embeddings(sentences)
