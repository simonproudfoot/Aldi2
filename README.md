# Aldi Recipe Assistant

An experimental supermarket comparison and recipe creator chatbot. This project uses FastAPI, Pinecone, and OpenAI to recommend budget-friendly recipes, match ingredients to Aldi UK products, and provide cost breakdowns for shopping lists.

## Features
- **Recipe Recommendations:** Get recipes based on your preferences, budget, and dietary restrictions.
- **Aldi Product Matching:** Ingredients are matched to real Aldi UK products with live price lookups.
- **Shopping List Generation:** Automatically create a shopping list and cost breakdown for selected recipes.
- **Product Price Queries:** Ask for the price of specific products at Aldi.
- **Conversation Context:** Remembers your conversation for more relevant suggestions.

## Tech Stack
- Python 3.10+
- FastAPI
- Pinecone (vector database)
- OpenAI (GPT-4/3.5)
- Sentence Transformers (for embeddings)

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Aldi2
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root with the following:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   PINECONE_API_KEY=your-pinecone-api-key
   ```

5. **(Optional) Index your data:**
   If you haven't already, run the `recipe_vectorizer.py` script to create and populate your Pinecone indexes.

## Running the App

```bash
python chat.py
```

The API will be available at [http://localhost:8000](http://localhost:8000)

- Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

## Main API Endpoints

- `POST /chat` — Main chat endpoint for recipe and product queries
- `GET /search/recipes` — Search for recipes directly
- `GET /search/products` — Search for Aldi products
- `GET /search/ingredients` — Search for ingredients
- `POST /shopping-list` — Generate a shopping list from recipe IDs
- `GET /stats` — Get database and model stats
- `GET /health` — Health check endpoint

## Pinecone & OpenAI Setup
- Make sure your Pinecone indexes are created with the correct vector dimensions (default: 384 for `all-MiniLM-L6-v2`).
- You need valid API keys for both OpenAI and Pinecone.

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes.

## License
MIT License
