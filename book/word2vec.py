import gensim
from gensim.models import Word2Vec, word2vec
from nltk.tokenize import word_tokenize,sent_tokenize
import nltk
# nltk.download('punkt')

sentences = 'Isabella ceased speaking, and took a drink of tea; then she rose, and \
bidding me put on her bonnet, and a great shawl I had brought, and \
turning a deaf ear to my entreaties for her to remain another hour . \
She stepped on to a chair, kissed Edgar’s and Catherine’s portraits , \
bestowed a similar salute on me, and descended to the carriage , \
accompanied by Fanny, who yelped wild with joy at recovering her \
mistress. '

data = []

for i in sent_tokenize(sentences):
    temp = []

    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

print(data)

# Create CBOW Model
model1 = gensim.models.Word2Vec(data,
                          min_count=1,
                          vector_size=100,
                          window=5,
                                sg=0)


model1.save("word2vec.model")

print("CBOW: ", model1.wv.most_similar('tea', topn=1))

# Create SkipGram Model
model1 = gensim.models.Word2Vec(data,
                          min_count=1,
                          vector_size=100,
                          window=5,
                                sg=1)


model1.save("word2vec.model")

print("Skip-gram: ", model1.wv.most_similar('tea', topn=1))
