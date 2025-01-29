# ReviewRush 

This repository hosts the implementation of ReviewRush: A Multi-PDF query system, designed to enable interactive querying and analysis of research documents using Retrieval-Augmented Generation (RAG). 

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

ReviewRush leverages advanced machine-learning techniques to provide an intuitive interface for querying PDF documents. The system combines the power of language models with retrieval mechanisms to generate accurate and contextually relevant responses based on the content of the PDFs(papers).
![Architecture](https://github.com/ABHISHEKgauti25/ChatPDF_RAG_Gemini/assets/109408129/231656ce-866d-4d88-8507-03a3d9dc68b0)


## Features

- **Retrieval-Augmented Generation (RAG):** Utilizes retrieval mechanisms to enhance the response generation process, ensuring high relevance and accuracy.
- **Conversational Memory:** Supports follow-up questions, maintaining context across multiple interactions for a coherent conversational experience.
- **User-Friendly Interface:** Easy-to-use interface for uploading PDFs and querying their content.

## Installation

To install and set up the ChatPDF RAG Gemini project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ABHISHEKgauti25/ReviewRush.git
    cd ReviewRush
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can use the deployed version of the application at [ReviewRush](https://rushreview.streamlit.app/).

## Contributing

We welcome contributions to enhance the functionality and features of ChatPDF RAG Gemini. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.
