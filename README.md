## LinkedIn Job Scraper & Analysis Project

# Overview
This project automates the process of searching and collecting job offers from LinkedIn, specifically focused on "Data Scientist" roles. It scrapes relevant job data such as job titles, company names, locations, and descriptions, and performs an analysis to extract key insights from the job descriptions using natural language processing (NLP) techniques.

# Features
Web Automation: Uses Selenium WebDriver to navigate LinkedIn and interact with the job listings.
Data Scraping: Gathers job details from multiple pages, including job descriptions and company information.
Text Preprocessing: Performs lemmatization and removes stopwords to clean the job descriptions.
Keyword Extraction: Uses TF-IDF (Term Frequency - Inverse Document Frequency) vectorization to identify the most relevant skills and keywords.
Insights Generation: Provides a summary of the most requested skills and terms across the collected job postings.

# Workflow

  1. Web Scraping
Initialize the web driver (supports Chrome, Edge, and Firefox).
Open LinkedIn and handle cookie consent.
Search for "Data Scientist" roles and apply relevant filters.
Scroll through job listings and collect data across multiple pages.

  2. Data Collection
The data extracted from each job offer includes:

    -Company Name
    -Job Title
    -Location
    -Job Description
    -Link to the Job Post
    
  3. Text Analysis
     
Lemmatization: Reduces words to their base form for more consistent keyword extraction.
Stopword Removal: Removes common words that do not add meaningful information.
TF-IDF Vectorization: Identifies the most important terms across job descriptions.
    
  4. Insights Generation
     
Summarizes the job market trends by highlighting the most commonly requested skills and terms, helping job seekers and recruiters understand key job           requirements.

# Dependencies

The following Python libraries are required to run the project:

    pandas: Data manipulation and analysis.
    BeautifulSoup: HTML parsing.
    tqdm: Progress bar for loops.
    requests: HTTP requests for web interactions.
    selenium: Web automation.
    nltk: Natural language processing.
    scikit-learn: Machine learning and text vectorization.
