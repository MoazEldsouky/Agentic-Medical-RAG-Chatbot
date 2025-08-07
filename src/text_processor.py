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
