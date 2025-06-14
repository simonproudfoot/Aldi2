#!/usr/bin/env python3
"""
Upgrade vector database creation to use OpenAI's text-embedding-3-large
for better RAG performance with large recipe documents
"""

import json
import os
import numpy as np
from typing import List, Dict
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from tqdm import tqdm
import time

class OpenAIEmbeddingVectorizer:
    """Create vectors using OpenAI's superior embedding models"""
    
    def __init__(self, model_name="text-embedding-3-large"):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.qdrant = QdrantClient(host="localhost", port=6333)
        
        # Model dimensions
        self.dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536
        }
        
        print(f"🚀 Using OpenAI {model_name} ({self.dimensions[model_name]} dimensions)")
    
    def create_embeddings_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Create embeddings in batches to avoid rate limits"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [embedding.embedding for embedding in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error creating embeddings for batch {i}: {e}")
                # Fallback: create zero vectors
                batch_embeddings = [[0.0] * self.dimensions[self.model_name]] * len(batch)
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def upgrade_existing_collection(self, recipes_file: str = "enhanced_recipes.json"):
        """Upgrade existing collection with OpenAI embeddings"""
        
        # Load existing enhanced recipes
        print(f"📥 Loading recipes from {recipes_file}...")
        try:
            with open(recipes_file, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
        except FileNotFoundError:
            print(f"❌ {recipes_file} not found. Run create-vector-b.py first.")
            return
        
        print(f"📊 Found {len(recipes)} recipes to upgrade")
        
        # Extract recipe texts for embedding
        texts = [recipe['text'] for recipe in recipes]
        
        # Create OpenAI embeddings
        print("🧠 Creating OpenAI embeddings...")
        embeddings = self.create_embeddings_batch(texts)
        
        # Create new collection with larger dimensions
        collection_name = f"aldi_bbc_recipes_openai_{self.model_name.split('-')[-1]}"
        
        print(f"🗄️ Creating new collection: {collection_name}")
        
        # Delete if exists
        try:
            self.qdrant.delete_collection(collection_name)
        except:
            pass
        
        # Create new collection
        self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.dimensions[self.model_name], 
                distance=Distance.COSINE
            )
        )
        
        # Prepare points
        points = []
        for i, (recipe, embedding) in enumerate(zip(recipes, embeddings)):
            point = PointStruct(
                id=i,
                vector=embedding,
                payload={
                    'text': recipe['text'],
                    'name': recipe['metadata']['name'],
                    'chef': recipe['metadata']['chef'],
                    'total_cost': recipe['metadata']['total_cost'],
                    'cost_per_serving': recipe['metadata']['cost_per_serving'],
                    'serves': recipe['metadata']['serves'],
                    'category': recipe['metadata']['category'],
                    'prep_time': recipe['metadata']['prep_time'],
                    'cook_time': recipe['metadata']['cook_time'],
                    'dietary_info': recipe['metadata']['dietary_info'],
                    'match_rate': recipe['metadata']['match_rate'],
                    'url': recipe['metadata']['url'],
                    'original_recipe': recipe['original_recipe']
                }
            )
            points.append(point)
        
        # Upload in batches
        batch_size = 100
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading vectors"):
            batch = points[i:i + batch_size]
            self.qdrant.upsert(collection_name=collection_name, points=batch)
        
        print(f"✅ Upgraded to OpenAI embeddings!")
        print(f"📊 Collection: {collection_name}")
        print(f"🔢 Dimensions: {self.dimensions[self.model_name]}")
        print(f"📦 Recipes: {len(points)}")
        
        return collection_name
    
    def test_search_quality(self, collection_name: str):
        """Test search quality with OpenAI embeddings"""
        test_queries = [
            "cheap vegetarian dinner under £8",
            "quick chicken recipe for family",
            "healthy curry with rice",
            "budget pasta meal",
            "microwave recipes easy"
        ]
        
        print(f"\n🧪 Testing search quality...")
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Create query embedding
            query_embedding = self.openai_client.embeddings.create(
                model=self.model_name,
                input=[query]
            ).data[0].embedding
            
            # Search
            results = self.qdrant.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=3,
                score_threshold=0.3
            )
            
            if results:
                for i, hit in enumerate(results, 1):
                    print(f"   {i}. {hit.payload['name']} (score: {hit.score:.3f})")
                    print(f"      £{hit.payload['total_cost']:.2f} | {hit.payload['chef']}")
            else:
                print("   ❌ No results found")

# Enhanced chat backend to use OpenAI embeddings
def update_chat_backend_for_openai():
    """Code changes needed for chat.py to use OpenAI embeddings"""
    
    return '''
# In chat.py, update these parts:

import openai

# At startup, initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
COLLECTION_NAME = "aldi_bbc_recipes_openai_large"  # Update collection name

# Replace the search function embedding creation:
async def search_recipes(query: str, max_budget: float = 20.0, dietary_restrictions: List[str] = [], max_results: int = 8) -> List[Dict]:
    """Enhanced recipe search with OpenAI embeddings"""
    
    # Create query embedding using OpenAI
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-large",
            input=[query]
        )
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"OpenAI embedding error: {e}")
        return []
    
    # Rest of function remains the same...
    filters = {
        "must": [
            {"key": "total_cost", "range": {"lte": max_budget}},
            {"key": "match_rate", "range": {"gte": 0.2}}
        ]
    }
    
    # Continue with existing search logic...
'''

def main():
    """Upgrade to OpenAI embeddings"""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable required")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("🔧 OpenAI Embedding Upgrade")
    print("=" * 40)
    
    # Choose model based on budget
    model_choice = input("Choose model:\n1. text-embedding-3-large (best quality, higher cost)\n2. text-embedding-3-small (good quality, lower cost)\nChoice (1/2): ").strip()
    
    if model_choice == "2":
        model_name = "text-embedding-3-small"
    else:
        model_name = "text-embedding-3-large"
    
    # Initialize vectorizer
    vectorizer = OpenAIEmbeddingVectorizer(model_name)
    
    # Upgrade collection
    collection_name = vectorizer.upgrade_existing_collection()
    
    if collection_name:
        # Test search quality
        vectorizer.test_search_quality(collection_name)
        
        print(f"\n🎉 Upgrade complete!")
        print(f"📝 Next steps:")
        print(f"   1. Update COLLECTION_NAME in chat.py to: '{collection_name}'")
        print(f"   2. Add OpenAI embedding creation to search_recipes() function")
        print(f"   3. Restart your backend")
        
        print(f"\n💰 Cost estimate:")
        print(f"   Model: {model_name}")
        if model_name == "text-embedding-3-large":
            print(f"   Cost: ~$0.13 per 1M tokens (high accuracy)")
        else:
            print(f"   Cost: ~$0.02 per 1M tokens (good accuracy)")

if __name__ == "__main__":
    main()