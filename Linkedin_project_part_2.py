########## ADVANCED PROGRAMMING PROJECT
####### SECOND PART: AI. 
##### AUTHORS: Frezard Paul, Sartori Romain, Vasta Francesca
## The goal is to derive the industry of each company and generate a text which summarizes the job offer situation in France

#LOAD THE CSV
import pandas as pd

job_offers_data =  pd.read_csv('job_offers_linkedin.csv')


####### CLEAN THE JOB DESCRIPTIONS
import emoji

# Function to remove emoji
def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

# apply the function to the column 'Job Description'
job_offers_data['Cleaned Job Description'] = job_offers_data['Job Description'].apply(lambda x: remove_emoji(str(x)))

# see the first values to see if it worked and check the type
print(job_offers_data['Cleaned Job Description'].head(20))

job_offers_data['Cleaned Job Description'].astype(str)
job_offers_data['Cleaned Job Description'].dtype


job_offers_data.to_csv('job_offers_linkedin_cleaned.csv', index = False)

######### TRANSLATE THE JOB DESCRIPTIONS FROM FRENCH TO ENGLISH
from transformers import pipeline

# create the pipeline and make it work on CPU
translator = pipeline("translation_fr_to_en", model="Helsinki-NLP/opus-mt-fr-en", device=-1)

# the descriptions are very long so we divided them in chunks
def translate_job_description(description, target_language='en', max_length=400):
    if pd.isna(description) or description.strip() == '':
        return ''  # if it is NaN the string will be empty
    
    # dividing the long description into smaller chunks
    chunks = [description[i:i + max_length] for i in range(0, len(description), max_length)]
    
    translated_chunks = []
    for chunk in chunks:
        try:
            translated = translator(chunk)
            translated_chunks.append(translated[0]['translation_text'])
        except Exception as e:
            print(f"Errore nella traduzione: {e}")
            translated_chunks.append(chunk)  # if there is an error, the chunk will be adde without being translated

    # put together the chunks
    return ' '.join(translated_chunks)

# iterate on all the job descriptions and take count of the progress made
total_descriptions = len(job_offers_data)  # Numero totale di descrizioni
for index, row in job_offers_data.iterrows():
    print(f"Currently translating job description nr: {index + 1}/{total_descriptions}")
    job_offers_data.at[index, 'Translated Job Description'] = translate_job_description(row['Cleaned Job Description'])

print("Transalation completed!")

job_offers_data.to_csv('job_offers_linkedin_cleaned.csv', index = False)

#LOAD THE CLEANED DATASET
job_offers_data =  pd.read_csv('job_offers_linkedin_cleaned.csv')


#PRELIMINARY ANALYSIS

#total number of offers
total_number = len(job_offers_data)

#analysis of location
location_counts_noNA = job_offers_data['Location'].notna().sum()
location_counts_clean =job_offers_data['Location'].dropna().str.strip().str.lower()

north_locations = location_counts_clean[location_counts_clean.str.contains("lille|paris|rouen|amiens|le havre|calais|dunkerque|arras|lens|saint-quentin|reims|Hauts-de-France|Île-de-France|Normandie", case=False)].count()
south_locations = location_counts_clean[location_counts_clean.str.contains("marseille|nice|toulouse|montpellier|bordeaux|nîmes|avignon|aix-en-provence|perpignan|cannes|saint-tropez|biarritz|toulon|arles|menton|ajaccio|Provence-Alpes-Côte d'Azur|Occitanie|Corse", case=False)].count()
east_locations = location_counts_clean[location_counts_clean.str.contains("dijon|strasbourg|mulhouse|nancy|metz|dijon|besançon|colmar|troyes|Grand Est|Bourgogne-Franche-Comté|Auvergne-Rhône-Alpes|Alsace", case=False)].count()
west_locations = location_counts_clean[location_counts_clean.str.contains("nantes|rennes|brest|angers|la rochelle|le mans|tours|saint-malo|caen|cherbourg|Bretagne|Pays de la Loire|Centre-Val de Loire", case=False)].count()

print(f'total number of job offers located in the North of France:',north_locations)
print(f'total number of job offers located in the South of France:',south_locations)
print(f'total number of job offers located in the East of France:',east_locations)
print(f'total number of job offers located in the West of France:',west_locations)

#hybrid opportunities
hybrid_jobs = job_offers_data['Translated Job Description'].str.contains('hybrid', case = False).sum()
print(f'total number of jobs which allow hybrid work:',hybrid_jobs)



###### QUESTION ANSWERING TO EXTRACT INFORMATION FROM THE JOB DESCRIPTIONS

from transformers import pipeline

# Question-Answering model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

job_offers_data['Translated Job Description'] = job_offers_data['Translated Job Description'].astype(str)

# EXTRACT THE INDUSTRY
#function to extract the industry from the job description

#function to extract the industry from the job description
def extract_industry(description):
    question = "in which industry or sector does the company operate?"
    context = description

    try:
        answer = qa_pipeline(question=question, context=context)
        return answer['answer'] #if answer['score'] > 0.5 else "Industry not found"
    
    except Exception as e:
        return('Industry not found')
    

# APPLY THE FUNCTION
job_offers_data['Industry'] = job_offers_data['Translated Job Description'].apply(extract_industry)

#mostra i valori unici delle industrie
industries_values = job_offers_data['Industry'].str.lower().value_counts()
print(industries_values)

# Show the dataframe with the predicted industries
print(job_offers_data.head())
top_5_industries = ', '.join(industries_values.index[0:5])

print(top_5_industries)


# EXTRACT THE CONTRACT LENGHT 
def extract_job_duration(description):
    question = "how much time (months or years) will the job contract last?"
    context = description 

    try:
        answer = qa_pipeline(question=question, context=context)
        return answer['answer']
    
    except Exception as e:
        return('Type of contract not found')

#apply the function 
job_offers_data['Time period'] = job_offers_data['Job Description'].apply(extract_job_duration)

# Show the data frame and the most frequent timespan
print(job_offers_data['Time period'])
most_frequent_time_period = job_offers_data['Time period'].value_counts().index[0]
print(most_frequent_time_period)


# Save then new dataframe
job_offers_data.to_csv('updated_job_offers_linkedin.csv', index=False)


###### FIND THE MOST REQUIRED SKILLS

# Import necessary libraries
import pandas as pd
import nltk #NLP library for tokenization, rimozione di stopwords, e lemmatization.
from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF (Term Frequency-Inverse Document Frequency), measures the importance of a word 
from wordcloud import WordCloud
import matplotlib.pyplot as plt #to visualize the wordcloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#stopwords, word_tokenize and WordLemmarizer are used to manage stopwords, to tokenize the text and to reduce words to their base form

# Download NLTK resources (if not already downloaded) for english words 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer object and stopwords list
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))


# Load the CSV file containing job descriptions
file_path = "updated_job_offers_linkedin.csv"  
df = pd.read_csv(file_path)

descriptions = df['Translated Job Description'].astype(str)

# Function to preprocess text in english
def preprocess_text(text):
    # Tokenize the text and convert everything to lowercase
    tokens = word_tokenize(text.lower()) #the word_tokenize function from NLTK performs word-based tokenization
    # Lemmatize tokens and remove special characters
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()] #checks if the word is alphanumeric
    # Remove stopwords (common words with little meaning)
    tokens = [word for word in tokens if word not in stopwords] #checks if the word is in the stop_words list or not
    return ' '.join(tokens) #return the clenaed text in a unique string

# Apply preprocessing to each job description
descriptions_cleaned = descriptions.apply(preprocess_text)
df['Translated Job Description_cleaned'] = descriptions_cleaned

# Display a sample of original and cleaned descriptions
print(df[['Translated Job Description', 'Translated Job Description_cleaned']].head())

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limit to 1000 most important terms
tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions_cleaned)
feature_names = tfidf_vectorizer.get_feature_names_out() 
#with 'get_feature_names_out() we obtain a list of all the features that the model identified during text vectorization. 
# #In other words, these are the words that were selected as significant (important) during the process of calculating TF-IDF scores.

# Sum the TF-IDF scores for each word across all descriptions
tfidf_scores = tfidf_matrix.sum(axis=0).A1
word_scores = dict(zip(feature_names, tfidf_scores)) #the dictionary will have the single words as key and the tdif_scores as values

# Generate a word cloud from the TF-IDF scores
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_scores)


# Print the top 20 keywords with their TF-IDF scores
top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:20]
print("Top 20 Keywords with their TF-IDF Scores:")
for word, score in top_words:
    print(f"{word}: {score:.3f}")


# Display the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Most Frequent Terms in Job Descriptions")
plt.show()


####### TEXT GENERATION 

#!pip install sentencepiece to do if not already installed

import torch 

from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the tokenizer and the model 
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

# Updated context with explicit formatting
context = f"""
Total number of job offers: {total_number}
- Number of jobs in the North of France: {north_locations}
- Number of jobs in the South of France: {south_locations}
- Number of jobs in the West of France: {west_locations}
- Number of jobs in the East of France: {east_locations}
- Total number of hybrid job offers: {hybrid_jobs}
- Leading industries in demand for employees: {top_5_industries}
- Most common contract duration: {most_frequent_time_period}
"""

# Adjusted instruction for T5
input_text = f'''
Generat a long and professonal text that analyzes the current employment situation and trends in France based on the following data:
{context}'''

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt")

# Generate the improved text
outputs = model.generate(
    inputs.input_ids,
    max_length=1000,
    num_return_sequences=1,
    temperature=1.2,
    top_p=0.95,
    num_beams=5,
    no_repeat_ngram_size=2
)

# Decode the result
improved_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(improved_text)
