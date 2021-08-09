# Text_Summarizer
This project is implemented to summarize original text of a given article. 

Input of this project is URL of the original text and a threshold that user can set to define length of summary.

After reading the original text, I used different techniques of Natural Language Processing (NLP) to preprocess the text including:<br>
Stop words, numbers, special characters, stem of words, etc.

Then sentences will be tokenized, and importance of each sentence will be calculated and sorted. It should be noted that different algorithms calculate the importance of each sentence differenly. 

Then based on the user defined threshold, text summary along with statistical report of the original text will be generated.

  <h1>Implemented Algorithms:</h1>
	<p>In this projects the followin four NLP algorithms are implemented. It is an object-oriented code where different classes are defined and used. Since the project is employed in form of a web application, you can run the project by clicking on each algorithm. (Due to the very large size of the required file for "text-rank" algorithm, it is not loaded online. However the source code is accessible in my GitHub account)</p>
	<ul>
		<li><a href="/wordFreq">Word Frequency</a></li>
		<li><a href="/tf-idf">TF-IDF</a></li>
		<li><a href="/lexRank">Lex Rank</a></li>
		<li><a href="/textRank">Text Rank</a></li>
	</ul>

Finally, this project is implemented in form of a machine learning web applications. So, any user can run this project in my website at http://tednaseri.pythonanywhere.com/
