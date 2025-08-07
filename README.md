# Agentic Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) based medical chatbot with agentic capabilities. This project leverages advanced language models and vector search to provide accurate, context-aware answers to medical queries, using both company-specific and general medical knowledge.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preparation](#data-preparation)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Retrieval-Augmented Generation (RAG):** Combines LLMs with a vector store for contextually relevant answers.
- **Agentic Reasoning:** Supports multi-step reasoning and tool use for complex queries.
- **Medical Domain Focus:** Designed for medical FAQs, company info, and general health questions.
- **Extensible:** Modular codebase for easy adaptation to new data sources or domains.

---

## Demo

[![Watch the demo on YouTube](https://img.youtube.com/vi/MuRdFiiDmf0/0.jpg)](https://www.youtube.com/watch?v=MuRdFiiDmf0)

Watch a full video demonstration of the Agentic Medical RAG Chatbot on [YouTube](https://www.youtube.com/watch?v=MuRdFiiDmf0).

---

## Project Structure

```
Agentic-Medical-RAG-Chatbot/
│
├── app.py                      # Main entry point for the chatbot app
├── src/
│   ├── agent.py                # Agent logic and orchestration
│   ├── config.py               # Configuration settings
│   ├── data_loaders.py         # Data loading utilities
│   ├── retriever.py            # Vector store retriever logic
│   ├── text_processor.py       # Text preprocessing and chunking
│   ├── tools.py                # Tool definitions for agent
│   ├── utils.py                # Utility functions
│   └── vector_store.py         # Vector store management (e.g., FAISS)
│
├── data/
│   ├── raw_company_info/       # Raw data (e.g., FAQ.csv, info.md)
│   └── processed/              # Processed data and vector indices
│
├── notebooks/                  # Jupyter notebooks for experiments
├── assets/                     # Images and other assets
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MoazEldsouky/Agentic-Medical-RAG-Chatbot.git
   cd Agentic-Medical-RAG-Chatbot
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Prepare Data

- Place your company-specific FAQ and info files in `data/raw_company_info/`.
- Use the provided scripts or notebooks to process and index your data.

### 2. Run the Chatbot

```bash
python app.py
```

- The chatbot will start and be accessible via the command line or a web interface (if implemented).

---

## Data Preparation

- **Raw Data:** Place your CSV/Markdown files in `data/raw_company_info/`.
- **Processing:** Use `src/data_loaders.py` and `src/text_processor.py` to preprocess and chunk your data.
- **Vector Store:** The processed data and vector indices are stored in `data/processed/vector_store/`.

---

## Customization

- **Add New Tools:** Implement new tools in `src/tools.py` and register them with the agent.
- **Change Model or Retriever:** Modify `src/agent.py` and `src/vector_store.py` to use different LLMs or vector databases.
- **UI/UX:** Integrate with a web frontend or other interfaces as needed.

---

## Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- [FAISS](https://github.com/facebookresearch/faiss) for vector search.
- [OpenAI](https://openai.com/) for language models.
- [LangChain](https://github.com/hwchase17/langchain)

---

## Contact

For questions or support, please open an issue or contact [MoazEldsouky](https://github.com/MoazEldsouky).
