import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

class TextPreprocessor:
    def __init__(self):
        try:
            self.stopwords = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            self.logger = logging.getLogger(__name__)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLTK resources: {e}")
            raise

    def standardize_case(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        return re.sub(r'[^\w\s]', '', text)

    def normalize_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def remove_stopwords(self, words):
        return [word for word in words if word not in self.stopwords]

    def lemmatize_words(self, words):
        return [self.lemmatizer.lemmatize(word) for word in words]
    
    def remove_headers_and_footers(self, text, num_lines=2, pattern=None):
        try:
            if not text or not text.strip():
                return text
                
            lines = text.splitlines()
            if len(lines) <= 2 * num_lines:
                # Too few lines, don't remove anything
                return text
                
            # Skip the specified number of lines from top and bottom
            cleaned_lines = lines[num_lines:-num_lines]
            
            # Optional: Remove lines matching common header/footer patterns
            if pattern:
                cleaned_lines = [line for line in cleaned_lines 
                                if not re.search(pattern, line)]
            
            # Additional heuristic: Remove single-word lines that might be page numbers
            cleaned_lines = [line for line in cleaned_lines 
                            if not (len(line.strip().split()) == 1 and 
                                   line.strip().isdigit())]
            
            # Join lines back into text
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            self.logger.error(f"Error removing headers/footers: {e}")
            return text  # Return original text on error
    
    def remove_common_pdf_artifacts(self, text):
        try:
            # Remove form field indicators
            text = re.sub(r'\[\s*\]\s*|\[\s*X\s*\]|\(\s*\)\s*|\(\s*X\s*\)', '', text)
            
            # Remove common PDF annotations
            text = re.sub(r'<<[^>]*>>', '', text)
            
            # Remove artifact markers often found in PDFs
            text = re.sub(r'obj\s*\d+\s*\d+\s*R', '', text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error removing PDF artifacts: {e}")
            return text

    def preprocess(self, text, remove_headers_footers=True):
        try:
            if remove_headers_footers:
                text = self.remove_headers_and_footers(text)
            
            text = self.remove_common_pdf_artifacts(text)
                
            text = self.standardize_case(text)
            text = self.remove_punctuation(text)
            text = self.normalize_whitespace(text)
            
            words = text.split()
            words = self.remove_stopwords(words)
            words = self.lemmatize_words(words)
            
            return ' '.join(words)
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            raise