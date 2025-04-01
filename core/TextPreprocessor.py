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
    

    def remove_headers_and_footers(self, text, aggressive=False, pattern=None):
        try:
            if not text or not text.strip():
                return text
                    
            lines = text.splitlines()
            if len(lines) <= 4:  # For very short text, don't remove anything
                return text
            
            # Store original lines for fallback
            original_lines = lines.copy()
            
            # Use different strategies based on document characteristics
            if self._appears_to_be_slide(lines):
                # Slide-friendly approach - only remove obvious headers/footers
                cleaned_lines = self._clean_slide_headers_footers(lines, pattern)
            elif aggressive:
                # Traditional document approach - remove first/last few lines
                num_lines = 2
                cleaned_lines = lines[num_lines:-num_lines]
            else:
                # Conservative approach - only remove based on patterns
                cleaned_lines = self._pattern_based_removal(lines, pattern)
                
            # If we removed too much (over 30% of content), revert to original
            if len(cleaned_lines) < len(lines) * 0.7:
                self.logger.warning("Header/footer removal eliminated too much content, reverting")
                cleaned_lines = original_lines
                
            # Additional heuristic: Remove single-word lines that might be page numbers
            cleaned_lines = [line for line in cleaned_lines 
                            if not (len(line.strip().split()) == 1 and 
                                line.strip().isdigit())]
            
            # Join lines back into text
            return '\n'.join(cleaned_lines)
            
        except Exception as e:
            self.logger.error(f"Error removing headers/footers: {e}")
            return text  # Return original text on error
    

    def _appears_to_be_slide(self, lines):
        """Detect if the content appears to be from a slide/presentation."""
        # Characteristics of slides:
        # - Shorter overall text
        # - Fewer lines
        # - More bullet points
        # - Title followed by bullet points
        
        if len(lines) < 15:  # Short content
            return True
            
        # Check for bullet point patterns
        bullet_pattern = r'^\s*[•\-\*\>\◦\○\◆\◇\▪\▫\⚫\⚪\✓\✔\✕\✖\✗\✘]'
        bullet_lines = sum(1 for line in lines if re.match(bullet_pattern, line))
        
        # If more than 20% of lines are bullets, likely a slide
        if bullet_lines > len(lines) * 0.2:
            return True
        
        # If first non-empty line is short (likely a title) and followed by bullet points
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines and len(non_empty_lines[0].strip()) < 60:
            # Check for bullet points in the following lines
            for line in non_empty_lines[1:4]:  # Check next few lines
                if re.match(bullet_pattern, line):
                    return True
                    
        return False


    def _clean_slide_headers_footers(self, lines, pattern=None):
        """Clean headers/footers from slide-based content."""
        cleaned_lines = lines.copy()
        
        # For slides, we primarily rely on pattern matching rather than line position
        if pattern:
            cleaned_lines = [line for line in cleaned_lines 
                            if not re.search(pattern, line)]
        
        # Common slide footer patterns to remove
        footer_patterns = [
            r'^\s*\d+\s*$',  # Standalone page number
            r'confidential',  # Confidentiality notices
            r'all rights reserved',
            r'proprietary',
            r'^\s*www\.',  # Website in footer
            r'^\s*https?://',  # URL in footer
            r'\bpage\s+\d+\b',  # "Page X" footer
            r'^\s*[©Ⓒ]\s*\d{4}'  # Copyright notice
        ]
        
        # Combine all patterns
        combined_pattern = '|'.join(f'({p})' for p in footer_patterns)
        
        # Filter out footer lines
        if combined_pattern:
            cleaned_lines = [line for line in cleaned_lines 
                            if not re.search(combined_pattern, line, re.IGNORECASE)]
        
        return cleaned_lines


    def _pattern_based_removal(self, lines, pattern=None):
        """Remove headers/footers based only on patterns, not position."""
        if not pattern:
            # Default patterns for headers/footers
            patterns = [
                r'^\s*\d+\s*$',  # Standalone page numbers
                r'^\s*page\s+\d+\s+of\s+\d+\s*$',  # Page X of Y
                r'^\s*[©Ⓒ]\s*\d{4}.*$',  # Copyright lines
                r'^\s*confidential\s*$',  # Confidentiality markers
                r'^\s*https?://.*$',  # URLs alone on a line
                r'^\s*www\..*$',  # Website alone on a line
                r'^\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\s*$'  # Email addresses
            ]
            combined_pattern = '|'.join(f'({p})' for p in patterns)
        else:
            combined_pattern = pattern
            
        return [line for line in lines 
                if not re.search(combined_pattern, line, re.IGNORECASE)]


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

    def preprocess(self, text, remove_headers_footers=True, aggressive_removal=False):
        try:
            if remove_headers_footers:
                text = self.remove_headers_and_footers(text, aggressive=aggressive_removal)
            
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