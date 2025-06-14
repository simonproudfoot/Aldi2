#!/usr/bin/env python3
"""
Hybrid Multi-Level Recipe Vectorizer
Creates separate collections for ingredients and recipes for better search
"""

import json
import os
import re
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class HybridRecipeVectorizer:
    def __init__(self, use_openai=False):
        print("🚀 Starting Hybrid Multi-Level Recipe Vectorizer...")
        
        self.use_openai = use_openai
        
        if use_openai:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model_name = "text-embedding-3-large"
            self.dimensions = 3072
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimensions = 384
        
        self.qdrant = QdrantClient(host="localhost", port=6333)
        
        # Collection names
        self.ingredient_collection = "aldi_ingredients"
        self.recipe_collection = "aldi_recipes"
        
        # Load Aldi products
        self.aldi_lookup = self.load_aldi_products()
        
    def load_aldi_products(self) -> Dict:
        """Load Aldi products for cost calculation"""
        with open('aldi_products.json', 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        lookup = {}
        for product in products:
            lookup[product['name'].lower()] = {
                'name': product['name'],
                'price': product['price'],
                'category': product.get('category', 'unknown')
            }
        
        print(f"✅ Loaded {len(lookup)} Aldi products")
        return lookup
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding using OpenAI or SentenceTransformers"""
        if self.use_openai:
            response = self.openai_client.embeddings.create(
                model=self.model_name,
                input=[text]
            )
            return response.data[0].embedding
        else:
            return self.model.encode(text).tolist()
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings in batches"""
        if self.use_openai:
            embeddings = []
            batch_size = 50
            
            for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
                batch = texts[i:i + batch_size]
                try:
                    response = self.openai_client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )
                    batch_embeddings = [emb.embedding for emb in response.data]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"Error in batch {i}: {e}")
                    # Add zero vectors as fallback
                    embeddings.extend([[0.0] * self.dimensions] * len(batch))
            
            return embeddings
        else:
            return [self.model.encode(text).tolist() for text in tqdm(texts, desc="Creating embeddings")]
    
    def clean_ingredient(self, ingredient: str) -> str:
        """Clean ingredient text for better vectorization"""
        # Remove quantities and measurements
        cleaned = re.sub(r'\d+(\.\d+)?\s*(kg|g|lb|oz|ml|l|cups?|tbsp|tsp|cloves?|slices?|pieces?)', '', ingredient)
        cleaned = re.sub(r'^\d+\s*', '', cleaned)
        
        # Remove cooking instructions
        cooking_words = ['chopped', 'diced', 'sliced', 'minced', 'grated', 'crushed', 'roughly', 'finely']
        for word in cooking_words:
            cleaned = re.sub(f'\\b{word}\\b', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def extract_core_ingredient(self, ingredient: str) -> str:
        """Extract the main ingredient name"""
        cleaned = self.clean_ingredient(ingredient)
        
        # Remove descriptive words but keep important ones
        important_words = ['free-range', 'organic', 'smoked', 'unsalted', 'wholemeal', 'extra virgin']
        descriptive_words = ['fresh', 'frozen', 'dried', 'large', 'medium', 'small', 'baby', 'young']
        
        words = cleaned.lower().split()
        filtered_words = []
        
        for word in words:
            if any(imp in word for imp in important_words):
                filtered_words.append(word)
            elif word not in descriptive_words:
                filtered_words.append(word)
        
        return ' '.join(filtered_words).strip()
    
    def find_aldi_match(self, ingredient: str) -> Optional[Dict]:
        """Find best Aldi match for ingredient"""
        core_ingredient = self.extract_core_ingredient(ingredient).lower()
        
        # Direct match
        if core_ingredient in self.aldi_lookup:
            return self.aldi_lookup[core_ingredient]
        
        # Partial matches
        best_match = None
        best_score = 0
        
        for aldi_name, aldi_data in self.aldi_lookup.items():
            # Check if ingredient words are in Aldi product name
            ingredient_words = set(core_ingredient.split())
            aldi_words = set(aldi_name.split())
            
            common_words = ingredient_words.intersection(aldi_words)
            if common_words:
                score = len(common_words) / len(ingredient_words)
                if score > best_score and score > 0.5:
                    best_score = score
                    best_match = aldi_data
        
        return best_match
    
    def estimate_ingredient_cost(self, ingredient: str) -> float:
        """Estimate cost for single ingredient"""
        aldi_match = self.find_aldi_match(ingredient)
        
        if aldi_match:
            base_price = aldi_match['price']
            
            # Estimate usage multiplier
            if any(word in ingredient.lower() for word in ['tsp', 'tbsp', 'pinch']):
                return base_price * 0.1
            elif any(word in ingredient.lower() for word in ['slice', 'clove']):
                return base_price * 0.2
            elif 'tin' in ingredient.lower() or 'can' in ingredient.lower():
                return base_price
            else:
                return base_price * 0.6  # Default usage
        
        # Fallback estimates
        if any(word in ingredient.lower() for word in ['salt', 'pepper', 'herbs', 'spices']):
            return 0.5
        elif any(word in ingredient.lower() for word in ['meat', 'chicken', 'beef', 'fish']):
            return 3.0
        else:
            return 1.5
    
    def process_recipes(self, recipes_file: str) -> tuple[List[Dict], List[Dict]]:
        """Process recipes into ingredient and recipe vectors"""
        print(f"📊 Loading recipes from {recipes_file}...")
        
        with open(recipes_file, 'r', encoding='utf-8') as f:
            recipes = json.load(f)
        
        # Limit for testing
        recipes = recipes  # Change to recipes for full processing
        
        ingredient_points = []
        recipe_points = []
        
        print("🔍 Processing recipes...")
        
        for recipe_id, recipe in enumerate(tqdm(recipes)):
            recipe_name = recipe.get('name', f'Recipe {recipe_id}')
            chef = recipe.get('chef', 'Unknown')
            ingredients = recipe.get('ingredients', [])
            
            if not ingredients:
                continue
            
            # Calculate total recipe cost
            total_cost = 0
            matched_ingredients = 0
            
            # Process each ingredient
            for ing_id, ingredient in enumerate(ingredients):
                core_ingredient = self.extract_core_ingredient(ingredient)
                
                if len(core_ingredient) < 3:  # Skip very short ingredients
                    continue
                
                # Find Aldi match and cost
                aldi_match = self.find_aldi_match(ingredient)
                ingredient_cost = self.estimate_ingredient_cost(ingredient)
                total_cost += ingredient_cost
                
                if aldi_match:
                    matched_ingredients += 1
                
                # Create ingredient point
                ingredient_points.append({
                    'id': len(ingredient_points),
                    'text': core_ingredient,
                    'metadata': {
                        'original_ingredient': ingredient,
                        'core_ingredient': core_ingredient,
                        'recipe_id': recipe_id,
                        'recipe_name': recipe_name,
                        'chef': chef,
                        'ingredient_cost': ingredient_cost,
                        'aldi_match': aldi_match['name'] if aldi_match else None,
                        'aldi_price': aldi_match['price'] if aldi_match else None,
                        'category': recipe.get('category', 'unknown'),
                        'dietary_info': recipe.get('dietary_info', []),
                        'prep_time': recipe.get('prep_time', 'Unknown'),
                        'cook_time': recipe.get('cook_time', 'Unknown')
                    }
                })
            
            # Calculate recipe metrics
            serves = self._extract_serves(recipe.get('serves', 'Serves 4'))
            cost_per_serving = total_cost / serves if serves > 0 else total_cost
            match_rate = matched_ingredients / len(ingredients) if ingredients else 0
            
            # Create recipe-level text
            recipe_text = self.create_recipe_text(recipe, total_cost, cost_per_serving, match_rate)
            
            # Create recipe point
            recipe_points.append({
                'id': recipe_id,
                'text': recipe_text,
                'metadata': {
                    'name': recipe_name,
                    'chef': chef,
                    'total_cost': round(total_cost, 2),
                    'cost_per_serving': round(cost_per_serving, 2),
                    'serves': serves,
                    'match_rate': match_rate,
                    'ingredient_count': len(ingredients),
                    'matched_ingredients': matched_ingredients,
                    'category': recipe.get('category', 'unknown'),
                    'dietary_info': recipe.get('dietary_info', []),
                    'prep_time': recipe.get('prep_time', 'Unknown'),
                    'cook_time': recipe.get('cook_time', 'Unknown'),
                    'url': recipe.get('url', ''),
                    'original_recipe': recipe
                }
            })
        
        print(f"✅ Created {len(ingredient_points)} ingredient vectors")
        print(f"✅ Created {len(recipe_points)} recipe vectors")
        
        return ingredient_points, recipe_points
    
    def _extract_serves(self, serves_text: str) -> int:
        """Extract number of servings"""
        match = re.search(r'(\d+)', str(serves_text))
        return int(match.group(1)) if match else 4
    
    def create_recipe_text(self, recipe: Dict, total_cost: float, cost_per_serving: float, match_rate: float) -> str:
        """Create searchable recipe text"""
        name = recipe.get('name', 'Unknown Recipe')
        chef = recipe.get('chef', 'Unknown')
        category = recipe.get('category', 'general')
        prep_time = recipe.get('prep_time', 'Unknown')
        cook_time = recipe.get('cook_time', 'Unknown')
        dietary_info = recipe.get('dietary_info', [])
        
        # Create cooking method keywords
        method_text = ' '.join(recipe.get('method', []))[:200]
        
        # Budget category
        if total_cost < 5:
            budget_cat = "ultra budget"
        elif total_cost < 10:
            budget_cat = "budget friendly"
        elif total_cost < 15:
            budget_cat = "moderate cost"
        else:
            budget_cat = "premium"
        
        text = f"""
{name} by {chef}
{category} recipe
{prep_time} preparation {cook_time} cooking
£{total_cost:.2f} total £{cost_per_serving:.2f} per serving
{budget_cat} meal
{' '.join(dietary_info)} dietary options
{method_text}
        """.strip()
        
        return ' '.join(text.split())  # Clean whitespace
    
    def create_collections(self, ingredient_points: List[Dict], recipe_points: List[Dict]):
        """Create Qdrant collections"""
        print("🗄️ Creating Qdrant collections...")
        
        # Delete existing collections
        for collection_name in [self.ingredient_collection, self.recipe_collection]:
            try:
                self.qdrant.delete_collection(collection_name)
                print(f"   🗑️ Deleted {collection_name}")
            except:
                pass
        
        # Create collections
        for collection_name in [self.ingredient_collection, self.recipe_collection]:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE)
            )
            print(f"   ✅ Created {collection_name}")
        
        # Process ingredient embeddings
        print("🧠 Creating ingredient embeddings...")
        ingredient_texts = [point['text'] for point in ingredient_points]
        ingredient_embeddings = self.create_embeddings_batch(ingredient_texts)
        
        # Upload ingredient points
        print("📤 Uploading ingredient vectors...")
        ing_points = []
        for point, embedding in zip(ingredient_points, ingredient_embeddings):
            ing_points.append(PointStruct(
                id=point['id'],
                vector=embedding,
                payload=point['metadata']
            ))
        
        batch_size = 100
        for i in tqdm(range(0, len(ing_points), batch_size), desc="Ingredient batches"):
            batch = ing_points[i:i + batch_size]
            self.qdrant.upsert(collection_name=self.ingredient_collection, points=batch)
        
        # Process recipe embeddings
        print("🧠 Creating recipe embeddings...")
        recipe_texts = [point['text'] for point in recipe_points]
        recipe_embeddings = self.create_embeddings_batch(recipe_texts)
        
        # Upload recipe points
        print("📤 Uploading recipe vectors...")
        rec_points = []
        for point, embedding in zip(recipe_points, recipe_embeddings):
            rec_points.append(PointStruct(
                id=point['id'],
                vector=embedding,
                payload=point['metadata']
            ))
        
        for i in tqdm(range(0, len(rec_points), batch_size), desc="Recipe batches"):
            batch = rec_points[i:i + batch_size]
            self.qdrant.upsert(collection_name=self.recipe_collection, points=batch)
        
        print(f"✅ Uploaded {len(ing_points)} ingredient vectors")
        print(f"✅ Uploaded {len(rec_points)} recipe vectors")
    
    def test_hybrid_search(self):
        """Test the hybrid search system"""
        test_queries = [
            "chicken recipes",
            "cheap vegetarian meals",
            "quick pasta dinner", 
            "budget curry under £8",
            "healthy breakfast recipes"
        ]
        
        print("\n🧪 Testing hybrid search...")
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            # Search ingredients
            ing_results = self.qdrant.search(
                collection_name=self.ingredient_collection,
                query_vector=query_embedding,
                limit=5,
                score_threshold=0.3
            )
            
            # Search recipes
            recipe_results = self.qdrant.search(
                collection_name=self.recipe_collection,
                query_vector=query_embedding,
                limit=3,
                score_threshold=0.3
            )
            
            # Show ingredient matches
            if ing_results:
                print("   📋 Ingredient matches:")
                for hit in ing_results[:3]:
                    print(f"      • {hit.payload['core_ingredient']} in {hit.payload['recipe_name']}")
                    print(f"        Cost: £{hit.payload['ingredient_cost']:.2f} | Score: {hit.score:.3f}")
            
            # Show recipe matches  
            if recipe_results:
                print("   🍳 Recipe matches:")
                for hit in recipe_results:
                    print(f"      • {hit.payload['name']} by {hit.payload['chef']}")
                    print(f"        £{hit.payload['total_cost']:.2f} total | {hit.payload['match_rate']:.0%} matched")

def main():
    print("🎯 Hybrid Multi-Level Recipe Vectorizer")
    print("=" * 50)
    
    # Choose embedding model
    use_openai = input("Use OpenAI embeddings? (y/n) [n]: ").strip().lower() == 'y'
    
    if use_openai and not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY required for OpenAI embeddings")
        return
    
    # Initialize vectorizer
    vectorizer = HybridRecipeVectorizer(use_openai=use_openai)
    
    # Process recipes
    ingredient_points, recipe_points = vectorizer.process_recipes('bbc_recipes.json')
    
    # Create collections
    vectorizer.create_collections(ingredient_points, recipe_points)
    
    # Test search
    vectorizer.test_hybrid_search()
    
    print(f"\n🎉 Hybrid vectorization complete!")
    print(f"   🥕 Ingredient collection: {vectorizer.ingredient_collection}")
    print(f"   🍳 Recipe collection: {vectorizer.recipe_collection}")
    print(f"   📊 Total ingredient vectors: {len(ingredient_points)}")
    print(f"   📊 Total recipe vectors: {len(recipe_points)}")

if __name__ == "__main__":
    main()