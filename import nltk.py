#import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words[:10])  # Print the first 10 stopwords as a test
