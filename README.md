# 🤖 Production Ready Generative AI Chatbot with RAG, LangChain, FastAPI & Docker

`azure-genai-chatbot-rag-langchain` is a Generative AI chatbot powered by Retrieval-Augmented Generation (RAG) using LangChain, and hosted via FastAPI. It integrates Azure OpenAI for language generation, Azure AI Search as the vector database for efficient retrieval, and Azure CosmosDB for storing conversation history as well as persistent memory. The entire system can containerized using Docker for seamless deployment and scalability.

---

## ⚙️ Key Features

- 🔍 **RAG-based architecture** for accurate and context-aware responses
- 🔗 Built with **LangChain Framework**
- 🌐 Integrated with **Azure OpenAI** to provide powerful language model capabilities
- 🔗 Utilizes **Azure AI Search** as a vector database for efficient retrieval of relevant context and data
- 💬 Conversation history is stored and managed in **Azure CosmosDB** for **persistent memory**.
- ⚡ **FastAPI** for fast, lightweight and scalable RESTful API services
- 🐳 **Dockerized** for consistent deployment across environments

---

## 🧰 Tech Stack

- **Python**
- **LangChain**
- **Azure OpenAI**
- **Azure AI Search**
- **Azure CosmosDB**
- **FastAPI**
- **Docker**

---

## 🚀 Getting Started

### 1. Clone the Repository  
```bash
git clone https://github.com/harsha905/azure-genai-chatbot-rag-langchain.git
cd azure-genai-chatbot-rag-langchain
```

---

## 📂 Project Structure
```
genai-reply-mobileapp-reviews/
│
├── .devcontainer/       
│   └── devcontainer.json
│
├── app/       
│   └── credentials.env
│   └── prompts.py
│   └── server.py
│   └── utils.py
│
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```
---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Contributing
This is a personal project built for learning and reference. However, feel free to open an issue or pull request if you find something useful to add or improve.
