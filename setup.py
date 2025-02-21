from setuptools import setup, find_packages

setup(
    name="prompt_generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.1.0",
        "langchain-openai>=0.0.8",
        "langchain-community>=0.0.28",
        "langchain-core>=0.1.31",
        "langgraph>=0.0.20",
        "openai>=1.14.2",
        "chromadb>=0.4.22",
        "faiss-cpu>=1.7.4",
        "python-dotenv>=1.0.1",
        "fastapi>=0.109.2",
        "uvicorn>=0.27.1",
        "python-multipart>=0.0.9",
        "py7zr>=0.20.8",
        "patool>=2.0.0",
        "unrar>=0.4"
    ],
) 