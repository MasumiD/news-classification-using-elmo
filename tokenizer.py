import re
from nltk.tokenize import TreebankWordTokenizer, casual_tokenize

class Tokenizer:
    def __init__(self):
        self.treebank_tokenizer = TreebankWordTokenizer()
        
    def preprocess_special_cases(self, text):
        text = re.sub(r'https?://\S+|www\.\S+', 'URL', text)
        text = re.sub(r'#\w+', 'HASHTAG', text)
        text = re.sub(r'@\w+', 'MENTION', text)
        text = re.sub(r'\b\d+%|\b\d+\s?percent\b', 'PERCENTAGE', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d+\s?(years old|yo|yrs|yr)\b', 'AGE', text, flags=re.IGNORECASE)
        text = re.sub(r'\b\d{1,2}:\d{2}\s?(AM|PM|am|pm)?\b', 'TIME', text)
        text = re.sub(r'\b\d+\s?(hours|hrs|minutes|mins|seconds|secs|days|weeks|months|years)\b', 'TIMEPERIOD', text, flags=re.IGNORECASE)
        return text
    
    def custom_sentence_split(self, text):
        abbreviations = ["Mr.", "Dr.", "Ms.", "Mrs.", "Prof.", "Sr.", "Jr.", "Ph.D.", "M.D.", "B.A.", "M.A.", "D.D.S.", "D.V.M.", "LL.D.", "B.C.", "a.m.", "p.m.", "etc.", "e.g.", "i.e.", "vs.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace(".", "<DOT>"))
        
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        

        sentences = [s.replace("<DOT>", ".") for s in sentences]
        return sentences

    def preprocess(self, text):
        text = self.preprocess_special_cases(text)
        sentences = self.custom_sentence_split(text)
        tokenized_sentences = []
        
        for sentence in sentences:
            casual_tokens = casual_tokenize(sentence, preserve_case=True)
            tokens = []
            for token in casual_tokens:
                tokens.extend(self.treebank_tokenizer.tokenize(token))
            tokenized_sentences.append(self.add_special_tokens(tokens))
        
        return tokenized_sentences

    def add_special_tokens(self, tokens):
        return ['START'] + tokens + ['END']

    def tokenize(self, text):
        return self.preprocess(text)

if __name__ == "__main__":
    input_text = input("Input text: ")
    tokenizer = Tokenizer()
    tokenized_sentences = tokenizer.tokenize(input_text)
    print("Tokenized text: ", tokenized_sentences)
