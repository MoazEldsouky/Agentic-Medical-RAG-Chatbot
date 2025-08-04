from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)

recursive_500 = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
    separators=[
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ".",     # Sentences
        ",",     # Clauses
        " ",     # Words
]
)


markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "company_title"),
        ("##", "section"),
    ]
)

# Add prescription-specific splitter
prescription_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  # Smaller chunks for prescriptions
    chunk_overlap=50,
    separators=[
        "\n\n",  # Paragraph breaks
        "\n",    # Line breaks
        ".",     # Sentences
        ",",     # Clauses
        " ",     # Words
    ]
)

def process_prescription_documents(documents):
    """Process prescription documents with specialized chunking"""
    processed_docs = []
    
    for doc in documents:
        if doc.metadata.get('type') == 'medication':
            # Keep medication docs intact
            processed_docs.append(doc)
        else:
            # Split larger documents
            chunks = prescription_splitter.split_documents([doc])
            processed_docs.extend(chunks)
    
    return processed_docs