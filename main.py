import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
#nltk.download('punkt')
#nltk.download('stopwords')
import string
import re
import demoji
#demoji.download_codes
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models import CoherenceModel

dataset = pd.read_csv('dataset.csv', low_memory=False)

length=len(dataset)-12
#length=100

df = dataset[['hashtags', 'username','content']]


#Aufgabe 1
# Sortiere den Datensatz nach der Spalte 'username' absteigend
user_frequency = dataset['username'].value_counts()

# Extrahieren der fünf häufigsten Nutzer
top_five_user = user_frequency.head(5).to_string()
top_five_user = top_five_user.replace("username","\n")


#Aufgabe 2
# Sortiere den Datensatz nach der Spalte 'hashtags' absteigend
hastags = dataset['hashtags'].value_counts()

# Extrahieren der fünf häufigsten Hashtags
top_five_hastags = hastags.head(5).to_string()
top_five_hastags = top_five_hastags.replace("hashtags","\n")

#Limitierung auf X Einträge
dataset = dataset.iloc[0:length-1]


# Aufgabe 3 - NLP
    #Methoden Definitionen
def stem_words(stopwords_removed):
    lemm = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()

    lemmatized_tokens_noun = [lemm.lemmatize(token,pos="n") for token in stopwords_removed]
    lemmatized_tokens_adjective = [lemm.lemmatize(token,pos="a") for token in lemmatized_tokens_noun]
    lemmatized_tokens_all = [lemm.lemmatize(token,pos="v") for token in lemmatized_tokens_adjective]
    stemmed_tokens= [porter_stemmer.stem(token) for token in lemmatized_tokens_all]
    return stemmed_tokens

    #Methode zum exportieren der Ergbnisse
def save_results(top_five_user, top_five_hashtags, top_topics):
    with open("save_result.txt", "w") as file:
        result=""
        for topic in top_topics:
            result+="Thema " + str(topic[0])+": \n\t"

            for entry in topic[1].split("+"):
                result+=str(entry)+"\n\t"

        file.write("Die fünf aktivsten Nutzer sind: " + str(top_five_user) +
                "\n \nDie fünf am häufigsten verwendeten Hashtags sind:" + str(top_five_hashtags) +
                "\n \nDie 5 häufigsten Tweet-Themen sind:\n \n" + str(result))
        
    # Methode zum filtern der eigentlichen Tweets
def extract_colum(dataset, column): 
    vector = dataset[column]
    return vector

    # Methode zum entfernen von Links in den Tweets
def remove_links(link):
    link_pattern = r'https\S+'
    text_without_links = re.sub(link_pattern, '', link)
    return text_without_links

    # Methode für Preprocessing, Tokenisierung und LDA
def lda_with_word_tokinsation(tweet, topic_num):
    #Stoppwörter entfernen #1
    cleaned_text= remove_stopwords(tweet)
    #Tokenisieren und Umwandlung in Kleinbuchstaben
    token_entries=word_tokenize(cleaned_text.lower())
     #Stoppwörter entfernen #2
    cleaned_text = [x for x in token_entries if x not in stopwords.words('english')]
    #Entfernen der Interpuntion
    list_without_punctation=[x for x in cleaned_text if x not in string.punctuation]
    #erneutes Tokenisieren und hinzufügebn der Token zu einer Liste
    cleaned_text=word_tokenize(" ".join(list_without_punctation))
    #Stemming und Lemmatizing
    list_without_punctation=stem_words(cleaned_text)

    #Definition Blacklist
    blacklist = ["’","“","”","``","'s","n't","s","a", "--", "m", "ç", "⬣","•"]
    #weiterführende Bereinigung durch eine Stoppwörter
    cleaned_text = [x for x in list_without_punctation if x not in blacklist]
    cleaned_text=" ".join(cleaned_text)
    #erneutes Tokenisieren, Umwandlung in Kleinschreibung und speichern in einer Variablen der Token
    tokens = [word_tokenize(sentence.lower()) for sentence in cleaned_text.split(".")]
    #LDA-Model
    dictionary = Dictionary(tokens)

    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    lda_model = LdaModel(corpus, num_topics=topic_num, id2word=dictionary, random_state=42)
    
    return lda_model.show_topics()



#Aufruf Methode - Filtern von Tweets
column_text = extract_colum(dataset,'content')

# weiteres Preprocessing (Entfernen von Emojis, Ziffern und Links)
fulltext=""
cleanded_tweets= []

for tweet in column_text:
    #Entfernung der Links
    tweet_without_links=remove_links(tweet)
    #Entfernen von Zahlen
    words = re.sub(r'[0-9]+',"", tweet_without_links)
    #Entfernung der Emojis
    cleanedentry_without_emoji= demoji.replace(words, repl="")
    fulltext+= cleanedentry_without_emoji

# Aufruf LDA - Methode    
result= lda_with_word_tokinsation(fulltext,5)

print("Processing...")
print("Done...")

print("Saving...")

#Aufruf der Methode zum speichern der Ergebnisse
save_results(top_five_user, top_five_hastags, result)

print("Done...")