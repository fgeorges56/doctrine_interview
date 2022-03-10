import nltk
import string
import re

# This function is used to clean and standardize the textual contents.
# Note that every block of this function could be a separate function to ease unit testing.
def preprocess(dataframe):
	# First, we lower all text.
	dataframe['content'] = dataframe['content'].str.lower() 

	# Then, we remove all HTML tags. 
	# Here we suppose they wield no significant information.
	dataframe['content'] = dataframe.apply(lambda x: re.sub("<[^>]*>", "", x["content"]), axis=1)

	# Next, we get rid of ponctuation ('.', ' : ', ...)
	nltk.download('punkt')
	table = str.maketrans('', '', string.punctuation)
	dataframe['content'] = dataframe.apply(lambda x: ''.join([w.translate(table) for w in x["content"]]), axis=1)

	# Next, we remove special characters ('\n', ...)
	dataframe['content'] = dataframe.apply(lambda x: re.sub("\t|\n|\r", " ", x["content"]), axis=1)
	# Now, we may have several consecutive spaces. Whe should reduce them.
	dataframe['content'] = dataframe.apply(lambda x: re.sub(" +", " ", x["content"]), axis=1)

	# Next, we tokenize : 
	dataframe['content'] = dataframe.apply(lambda x: nltk.tokenize.word_tokenize(x["content"], language='french'), axis=1)

	# After tokenizing, we must remove stop words
	# ex : remove "le", "du", "en"
	nltk.download('stopwords')
	stop_words = set(nltk.corpus.stopwords.words('french'))
	dataframe['content'] = dataframe.apply(lambda x: [w for w in x['content'] if not w in stop_words], axis=1)

	# Last, we recombine the text
	dataframe['content'] = dataframe.apply(lambda x: ' '.join(x['content']), axis=1)

	return dataframe
