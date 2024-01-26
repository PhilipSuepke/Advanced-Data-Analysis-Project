import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')
import string
import spacy
#python -m spacy download en_core_web_sm
import re
import demoji
#demoji.download_codes


nlp = spacy.load("en_core_web_sm")

#Aufgabe 1
# CSV-Datei einlesen
dataset = pd.read_csv('dataset.csv', low_memory=False)

length=len(dataset)-12

df = dataset[['hashtags', 'username','content']]

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
#Denfinition von Methoden

    #Methode zum exportieren der Ergbnisse
def save_results(top_five_user, top_five_hashtags, top_topics):
    with open("save_result.txt", "w") as file:
        file.write("Die fünf aktivsten Nutzer sind: " + str(top_five_user) +
                "\n \nDie fünf am häufigsten verwendeten Hashtags sind: " + str(top_five_hashtags) +
                "\n \nDie 5 häufigsten Tweet-Themen sind: " + str(top_topics))
        
    
    # Methode zum filtern der eigentlichen Tweets
def extract_colum(dataset, column): 
    vector = dataset[column]
    return vector

    # Methode zum entfernen von Links in den Tweets
def remove_links(link):
    link_pattern = r'https\S+'
    text_without_links = re.sub(link_pattern, '', link)
    return text_without_links



#Aufruf Methode - Filtern von Tweets
column_text = extract_colum(dataset,'content')

#Cleanen der Tweets
cleanded_tweets= []

for tweet in column_text:
    #Entfernung der Links
    tweet_without_links=remove_links(tweet)
    #Entfernung der Emojis
    cleanedentry_without_emoji= demoji.replace(tweet_without_links, repl="")
    #Tokenizen und Kleinschreibung
    token_entries=word_tokenize(cleanedentry_without_emoji.lower())
    #Entfernung der Stoppwörter
    entry_without_stoppwords = [x for x in token_entries if x not in stopwords.words('english')]
    #Entfernung der Interpunction
    entry_without_stoppwords_without_punctuation = [x for x in entry_without_stoppwords if x not in string.punctuation]
    #Umwandeln der Liste in einen String
    list_to_string= " ".join(entry_without_stoppwords_without_punctuation)
    #Hinzufügen der einzelnen Strings in eine Liste
    cleanded_tweets.append(list_to_string)



# NLP - Named Entity Recgonition
cleanedentries= []
for clean_tweet in cleanded_tweets:
    doc = nlp(clean_tweet)

    for ent in doc.ents:
        cleanedentries.append([ent.text,ent.label_])


df = pd.DataFrame(cleanedentries,columns =['Entity', 'Label'])


# Sortieren nach den häufigsten Themen
topic_frequency = df['Entity'].value_counts()

# Abspeichern der 5 häugisten Themen in einer Variablen
top_five_topics = topic_frequency.head(5).to_string()
top_five_topics = top_five_topics.replace("Entity","\n")

#Aufruf der Methode zum speichern der Ergebnisse
save_results(top_five_user, top_five_hastags, top_five_topics)