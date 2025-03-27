This is a pipeline program to prepare documents for a retrieval-augmented generation system. It runs through the command line, and a user guide is included.

You'll have to use your API keys since I'm not made of money. I'll note what parts require an API key as I get to it in the documentation.

Here's a rundown of everything included:
1. Load documents using PyMuPDF through Langchain
2. Run the documents through preprocessing and optical character recognition
3. Split the text using one of three methods:
   a. Paragraph (default)
   b. Page
   c. Hierarchical
4. Embed the document using one of the following models:
   a. OpenAI (requires an API key)
   b. Sentence Transformer
   c. Ollama (requires the models to be downloaded through ollama.com)
   d. Hugging Face (requires an API key)
5. Store in a vector database:
   a. Pinecone (requires an API key)
   b. ChromaDB (local instantiation)
   c. Memory (temporary...seriously, this is temporary, like that pair of matching socks you swore was there yesterday)

Because this runs through the command line, check out the user guide. Also, if downloading SpaCy doesn't work, check out the additional_downloads.txt file. 
