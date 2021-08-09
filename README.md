# Text_Summarizer
This project is implemented to summarize original text of a given article. 

Input of this project is URL of the original text and a threshold that user can set to define length of summary.

After reading the original text, I used different techniques of natural language processing to preprocess the text including:
stop words, numbers, special characters, stem of words, etc.

Then sentences will be tokenized, and importance of each sentence will be calculated and sorted. Then based on the user defined threshold, text summary along with statistical report of the original text will be generated.

Implemented Algorithms: 4 algorithms of Word-Frequency, TF-IDF, Lex-Rank, Text-Rank have been implemented.

Finally, this project is implemented in form of a machine learning web applications. So, any user can run this project in my website at http://tednaseri.pythonanywhere.com/
