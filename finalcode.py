import nltk
import scipy
import string
import numpy as np
import pandas as pd
import pickle
nltk.download('words')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

word_embeddings_loc = '#ENTER PRETRAINED WORD EMBEDDING#' #LOCATION OF THE GLOVE WORD EMBEDDING eg. F:/glove.6B.50d.txt

#NOW WE WIL PREPROCESS THE DATA
def preprocessing(document):
  # LOADING THE DATA FROM GOOGLE DRIVE
  with open(document, "r",encoding="utf-8") as file:  #"/gdrive/My Drive/ML stuff/sample.txt"
    text = file.read()
  text = text.replace("\n"," ")
  text = text.replace("\t"," ")
  #words = set(nltk.corpus.words.words())
  #text = " ".join(w for w in nltk.wordpunct_tokenize(text) if w.lower() in words or not w.isalpha())

  text = unidecode(text)
  punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~1234567890'''
  for x in text:
    if x in punctuations: 
      text = text.replace(x, " ") 

  corpus = text.lower().split(" ")
  stop_words = stopwords.words('english')

  # REMOVING THE PUNCTUATIONS

  cleantext = []
  for i in corpus:
    i = i.replace("\n"," ")
    for x in i: 
        if x in punctuations: 
            i = i.replace(x, "") 
    cleantext.append(i)

  # LEMMETIZATION OF TEXT

  morecleantext = []
  lem = WordNetLemmatizer()

  for w in cleantext:
    if w not in stop_words:
      w = lem.lemmatize(w)
      morecleantext.append(w)
  
  # EXPORTING WORD EMBEDDINGS FROM A TRAINED GloVe MODEL (this is from the function below)
  word_to_vec_map,words = read_glove_vecs(word_embeddings_loc) # alloting the index and words and vectors
 
  # CREATING A OUT OF VOCABULARY (OOV) LIST OF WORDS 
  oov = ['']
  while len(oov)>0:
    oov = []
    for word in morecleantext:
      if word not in words:
        oov.append(word)
    for word in morecleantext:
      if word in oov:
        morecleantext.remove(word)
    if len(oov) == 0:
      break  
  morecleantext = list(dict.fromkeys(morecleantext))
  return morecleantext
  # FINALLY WE HAVE OUR DOCUMENT TEXT CLEANED
  
  
# EXPORTING WORD EMBEDDINGS FROM A TRAINED GloVe MODEL
def read_glove_vecs(glove_file):
      with open(glove_file, 'r',encoding="utf-8") as f:
          words = set()         # ensures unique values
          word_to_vec_map = {}  # this will be a dictionary mapping words to their vectors
          for line in f:
              line = line.strip().split()
              curr_word = line[0]
              words.add(curr_word)
              word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
          
          i = 1
          words_to_index = {}   # dictionary mapping words to their index in the dictionary
          index_to_words = {}   # dictionary mapping index to the word in the dictionary
          for w in sorted(words):
              words_to_index[w] = i
              index_to_words[i] = w
              i = i + 1
      return word_to_vec_map,words
word_to_vec_map,words = read_glove_vecs(word_embeddings_loc) # alloting the index and words and vectors

 #creating wordvectors
def create_wordvectors(morecleantext, words):
  
  word_vectors = []
  for word in morecleantext:
    if word in words:
      word_vectors.append(word_to_vec_map[word])
  return word_vectors
 
#main program compiler

def return_topic_words(document,search_word):
  morecleantext = preprocessing(document)
  word_vectors = create_wordvectors(morecleantext,words)
  distances = []
  for word in morecleantext:
    cosine = scipy.spatial.distance.cosine(word_to_vec_map[search_word], word_to_vec_map[word])
    distances.append(cosine)
  df = pd.DataFrame(word_vectors, morecleantext)
  df = df.transpose()
  df = pd.DataFrame(np.c_[distances,morecleantext], columns=['DISTANCES','TOP_WORDS'])
  df = df.sort_values(by=['DISTANCES'])
  df_fin = pd.DataFrame(df['TOP_WORDS'][0:10])
  print(df_fin.to_string(index=False))


topic = input("Test Topic: ").lower()
return_topic_words("#ENTER DOCUMENT LOCTATION#",topic) 

#pickle.dump(names,open('searchmodel.pkl','wb'))

#model = pickle.load(open('searchmodel.pkl','rb'))
