#%%
import os 
import re
os.chdir(r'D:\MyDrive\10. MS in Data Science UofWisconsin\09. Data Visualization\Fourth Project')

with open('RomeAndJuliet.txt','r', encoding='utf-8') as f:
  book_RomeoAndJuliet = f.read()

import nltk 
# download the punkt tokenizer for sentence splitting
nltk.download('punkt')

# split the book string into sentences using nltk.sent_tokenize
sentences_RJ = nltk.sent_tokenize(book_RomeoAndJuliet)

#%%

# %%

# define the pattern to search for

# %%
sentences_RJ = sentences_RJ[85:3048]


chunks_RJ = []

for i in range(0,len(sentences_RJ),10):
  chunk = " ".join(sentences_RJ[i:i+10])
  chunks_RJ.append(chunk)


