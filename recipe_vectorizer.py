#!/usr/bin/env python3
"""
Pinecone-based Recipe and Ingredient Vectorizer
Creates embeddings for Aldi products and BBC recipes with cross-referencing
"""

import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import hashlib
# External dependencies
try:
    from pinecone import Pinecone, ServerlessSpec
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI
    from tqdm import tqdm
    from dotenv import load_dotenv
    print("🔍 Using modern Pinecone API")
        
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("💡 Install required packages:")
    print("   pip install pinecone sentence-transformers openai tqdm python-dotenv")
    exit(1)

@dataclass
class RecipeIngredient:
    """Represents a single ingredient from a recipe"""
    original_text: str
    normalized_name: str
    quantity: str
    unit: str
    recipe_id: str
    recipe_name: str

@dataclass
class AldiProduct:
    """Represents an Aldi product"""
    id: str
    name: str
    price: float
    category: str
    url: str
    image_url: str

class PineconeRecipeVectorizer:
    """Creates and manages Pinecone vector database for recipes and ingredients"""
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str):
        print("🚀 Initializing Pinecone Recipe Vectorizer...")
        
        # Initialize modern Pinecone client
        print("   📌 Using modern Pinecone API")
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Index names
        self.recipe_index_name = "aldi-recipes"
        self.ingredient_index_name = "aldi-ingredients"
        self.product_index_name = "aldi-products"
        
        # Data storage
        self.aldi_products = {}
        self.bbc_recipes = []
        self.recipe_ingredients = []
        
        print("✅ Vectorizer initialized successfully")
    
    def load_data(self, aldi_file: str, bbc_file: str):
        """Load Aldi products and BBC recipes"""
        print(f"📥 Loading data...")
        
        # Load Aldi products
        try:
            with open(aldi_file, 'r', encoding='utf-8') as f:
                aldi_data = json.load(f)
            
            for i, product in enumerate(aldi_data):
                product_id = f"aldi_{i}"
                self.aldi_products[product_id] = AldiProduct(
                    id=product_id,
                    name=product['name'],
                    price=product['price'],
                    category=product.get('category', 'unknown'),
                    url=product.get('url', ''),
                    image_url=product.get('image_url', '')
                )
            
            print(f"✅ Loaded {len(self.aldi_products)} Aldi products")
            
        except Exception as e:
            print(f"❌ Error loading Aldi products: {e}")
            return False
        
        # Load BBC recipes
        try:
            with open(bbc_file, 'r', encoding='utf-8') as f:
                self.bbc_recipes = json.load(f)
            
            print(f"✅ Loaded {len(self.bbc_recipes)} BBC recipes")
            
        except Exception as e:
            print(f"❌ Error loading BBC recipes: {e}")
            return False
        
        return True
    
    def create_indexes(self):
        """Create Pinecone indexes"""
        print("🔧 Creating Pinecone indexes...")
        
        # Vector dimension for sentence transformers
        dimension = 384
        
        indexes_to_create = [
            (self.recipe_index_name, "Recipe vectors with cost and ingredient data"),
            (self.ingredient_index_name, "Individual ingredient vectors linked to recipes"),
            (self.product_index_name, "Aldi product vectors for ingredient matching")
        ]
        
        for index_name, description in indexes_to_create:
            try:
                # Check if index exists
                if self.pc.has_index(index_name):
                    print(f"   ✅ Index '{index_name}' already exists")
                    continue
                
                # Create index with serverless spec
                print(f"   🔨 Creating index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                
                # Wait for index to be ready
                while not self.pc.describe_index(index_name).status['ready']:
                    print(f"   ⏳ Waiting for {index_name} to be ready...")
                    time.sleep(1)
                
                print(f"   ✅ Created index: {index_name}")
                
            except Exception as e:
                print(f"   ❌ Error creating index {index_name}: {e}")
                return False
        
        return True
    
    def extract_ingredients_from_recipe(self, recipe: Dict, recipe_idx: int) -> List[RecipeIngredient]:
        """Extract and normalize ingredients from a recipe"""
        ingredients = recipe.get('ingredients', [])
        recipe_id = f"recipe_{recipe_idx}"
        recipe_name = recipe.get('name', 'Unknown Recipe')
        
        extracted = []
        
        for ingredient_text in ingredients:
            # Basic ingredient parsing
            normalized = self.normalize_ingredient(ingredient_text)
            quantity, unit = self.extract_quantity_unit(ingredient_text)
            
            if normalized:  # Only add if we got a meaningful ingredient
                extracted.append(RecipeIngredient(
                    original_text=ingredient_text,
                    normalized_name=normalized,
                    quantity=quantity,
                    unit=unit,
                    recipe_id=recipe_id,
                    recipe_name=recipe_name
                ))
        
        return extracted
    
    def normalize_ingredient(self, ingredient_text: str) -> str:
        """Normalize ingredient text to core ingredient name"""
        # Remove quantities and measurements
        text = re.sub(r'\b\d+(\.\d+)?\s*(kg|g|lb|oz|ml|l|liters?|litres?|cups?|tbsp|tsp|tablespoons?|teaspoons?|cloves?|slices?|pieces?|pack|tin|can|jar)\b', '', ingredient_text, flags=re.IGNORECASE)
        
        # Remove quantity numbers at start
        text = re.sub(r'^\d+(\.\d+)?\s*', '', text)
        
        # Remove descriptive words but keep important ones
        important_words = ['free-range', 'organic', 'smoked', 'unsalted', 'wholemeal', 'extra virgin']
        descriptive_words = ['fresh', 'frozen', 'dried', 'chopped', 'diced', 'sliced', 'minced', 'grated', 'crushed', 'ground', 'roughly', 'finely', 'large', 'medium', 'small', 'baby', 'young', 'optional', 'to taste']
        
        words = text.split()
        filtered_words = []
        
        for word in words:
            word_clean = word.lower().strip('(),')
            # Keep important descriptive words
            if any(imp in word_clean for imp in important_words):
                filtered_words.append(word_clean)
            # Remove basic descriptive words
            elif word_clean not in descriptive_words and len(word_clean) > 2:
                filtered_words.append(word_clean)
        
        # Join and clean up
        result = ' '.join(filtered_words).strip()
        result = re.sub(r'\s+', ' ', result)
        
        return result if len(result) > 2 else ""
    
    def extract_quantity_unit(self, ingredient_text: str) -> Tuple[str, str]:
        """Extract quantity and unit from ingredient text"""
        # Look for quantity patterns
        quantity_pattern = r'^(\d+(?:\.\d+)?(?:\s*[-/]\s*\d+(?:\.\d+)?)?)\s*'
        quantity_match = re.search(quantity_pattern, ingredient_text)
        quantity = quantity_match.group(1) if quantity_match else ""
        
        # Look for unit patterns
        unit_pattern = r'\b(kg|g|lb|oz|ml|l|liters?|litres?|cups?|tbsp|tsp|tablespoons?|teaspoons?|cloves?|slices?|pieces?|pack|tin|can|jar)\b'
        unit_match = re.search(unit_pattern, ingredient_text, re.IGNORECASE)
        unit = unit_match.group(1) if unit_match else ""
        
        return quantity, unit
    
    def calculate_recipe_cost_with_aldi(self, recipe: Dict, recipe_ingredients: List[RecipeIngredient]) -> Dict:
        """Calculate recipe cost by matching ingredients to Aldi products"""
        total_cost = 0.0
        matched_ingredients = []
        unmatched_ingredients = []
        
        for ingredient in recipe_ingredients:
            # Find best matching Aldi product
            best_match = self.find_best_aldi_match(ingredient.normalized_name)
            
            if best_match and best_match['score'] > 0.6:
                # Estimate cost based on quantity
                base_cost = self.aldi_products[best_match['product_id']].price
                estimated_cost = self.estimate_ingredient_cost(ingredient, base_cost)
                
                total_cost += estimated_cost
                matched_ingredients.append({
                    'ingredient': ingredient.original_text,
                    'normalized': ingredient.normalized_name,
                    'aldi_product': self.aldi_products[best_match['product_id']].name,
                    'aldi_price': base_cost,
                    'estimated_cost': estimated_cost,
                    'match_score': best_match['score']
                })
            else:
                # Estimate cost for unmatched ingredient
                estimated_cost = self.estimate_unmatched_cost(ingredient.normalized_name)
                total_cost += estimated_cost
                unmatched_ingredients.append({
                    'ingredient': ingredient.original_text,
                    'normalized': ingredient.normalized_name,
                    'estimated_cost': estimated_cost
                })
        
        # Extract serves information
        serves_text = recipe.get('serves', 'Serves 4')
        serves_match = re.search(r'(\d+)', serves_text)
        serves = int(serves_match.group(1)) if serves_match else 4
        
        return {
            'total_cost': round(total_cost, 2),
            'cost_per_serving': round(total_cost / serves, 2),
            'serves': serves,
            'matched_ingredients': matched_ingredients,
            'unmatched_ingredients': unmatched_ingredients,
            'match_rate': len(matched_ingredients) / len(recipe_ingredients) if recipe_ingredients else 0
        }
    
    def find_best_aldi_match(self, ingredient_name: str) -> Optional[Dict]:
        """Find best matching Aldi product for an ingredient"""
        best_score = 0
        best_match = None
        
        # Simple keyword matching (could be enhanced with embeddings)
        ingredient_words = set(ingredient_name.lower().split())
        
        for product_id, product in self.aldi_products.items():
            product_words = set(product.name.lower().split())
            
            # Calculate word overlap score
            common_words = ingredient_words.intersection(product_words)
            if common_words:
                score = len(common_words) / len(ingredient_words.union(product_words))
                
                if score > best_score:
                    best_score = score
                    best_match = {
                        'product_id': product_id,
                        'score': score
                    }
        
        return best_match
    
    def estimate_ingredient_cost(self, ingredient: RecipeIngredient, base_cost: float) -> float:
        """Estimate cost of ingredient based on quantity"""
        # Simple estimation based on quantity and unit
        if not ingredient.quantity:
            return base_cost * 0.3  # Default 30% of product
        
        try:
            qty = float(ingredient.quantity.split()[0])  # Get first number
            
            # Unit-based estimation
            if ingredient.unit.lower() in ['g', 'ml']:
                return min(base_cost * (qty / 1000), base_cost)  # Convert to kg/l
            elif ingredient.unit.lower() in ['tsp', 'tbsp']:
                return base_cost * 0.1  # Small amount
            elif ingredient.unit.lower() in ['slice', 'piece']:
                return base_cost * min(qty / 10, 1.0)  # Assume 10 pieces per product
            else:
                return base_cost * min(qty, 1.0)  # Cap at full product cost
                
        except:
            return base_cost * 0.3  # Default fallback
    
    def estimate_unmatched_cost(self, ingredient_name: str) -> float:
        """Estimate cost for ingredients not found in Aldi"""
        ingredient_lower = ingredient_name.lower()
        
        # Category-based estimation
        if any(word in ingredient_lower for word in ['salt', 'pepper', 'herb', 'spice', 'stock']):
            return 0.50
        elif any(word in ingredient_lower for word in ['meat', 'chicken', 'beef', 'fish']):
            return 3.00
        elif any(word in ingredient_lower for word in ['cheese', 'cream', 'butter']):
            return 2.00
        else:
            return 1.50  # Default
    
    def create_recipe_vectors(self):
        """Create and upload recipe vectors to Pinecone"""
        print("📝 Creating recipe vectors...")
        
        index = self._get_index(self.recipe_index_name)
        
        vectors_to_upsert = []
        
        for i, recipe in enumerate(tqdm(self.bbc_recipes, desc="Processing recipes")):
            try:
                recipe_id = f"recipe_{i}"
                
                # Extract ingredients for this recipe
                recipe_ingredients = self.extract_ingredients_from_recipe(recipe, i)
                
                # Calculate costs with Aldi matching
                cost_data = self.calculate_recipe_cost_with_aldi(recipe, recipe_ingredients)
                
                # Create comprehensive text for embedding
                recipe_text = self.create_recipe_embedding_text(recipe, recipe_ingredients, cost_data)
                
                # Generate embedding
                embedding = self.sentence_model.encode(recipe_text).tolist()
                
                # Prepare metadata
                metadata = {
                    'type': 'recipe',
                    'name': recipe.get('name', f'Recipe {i}'),
                    'chef': recipe.get('chef', 'Unknown'),
                    'category': recipe.get('category', 'general'),
                    'prep_time': recipe.get('prep_time', 'Unknown'),
                    'cook_time': recipe.get('cook_time', 'Unknown'),
                    'serves': cost_data['serves'],
                    'total_cost': cost_data['total_cost'],
                    'cost_per_serving': cost_data['cost_per_serving'],
                    'match_rate': cost_data['match_rate'],
                    'dietary_info': recipe.get('dietary_info', []),
                    'url': recipe.get('url', ''),
                    'ingredients_count': len(recipe_ingredients),
                    'ingredient_names': [ing.normalized_name for ing in recipe_ingredients[:10]],  # First 10
                    'text': recipe_text[:1000]  # First 1000 chars for search
                }
                
                vectors_to_upsert.append({
                    'id': recipe_id,
                    'values': embedding,
                    'metadata': metadata
                })
                
                # Batch upsert every 100 vectors
                if len(vectors_to_upsert) >= 100:
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
                
            except Exception as e:
                print(f"❌ Error processing recipe {i}: {e}")
                continue
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
        
        print(f"✅ Uploaded {len(self.bbc_recipes)} recipe vectors")
    
    def create_ingredient_vectors(self):
        """Create and upload ingredient vectors to Pinecone"""
        print("🥬 Creating ingredient vectors...")
        
        index = self._get_index(self.ingredient_index_name)
        
        vectors_to_upsert = []
        ingredient_id = 0
        
        for recipe_idx, recipe in enumerate(tqdm(self.bbc_recipes, desc="Processing ingredients")):
            recipe_ingredients = self.extract_ingredients_from_recipe(recipe, recipe_idx)
            
            for ingredient in recipe_ingredients:
                try:
                    # Create embedding text for ingredient
                    ingredient_text = f"{ingredient.normalized_name} {ingredient.original_text} cooking ingredient food"
                    
                    # Generate embedding
                    embedding = self.sentence_model.encode(ingredient_text).tolist()
                    
                    # Find Aldi match
                    aldi_match = self.find_best_aldi_match(ingredient.normalized_name)
                    aldi_product_name = ""
                    aldi_price = 0.0
                    if aldi_match:
                        aldi_product = self.aldi_products[aldi_match['product_id']]
                        aldi_product_name = aldi_product.name
                        aldi_price = aldi_product.price
                    
                    # Prepare metadata
                    metadata = {
                        'type': 'ingredient',
                        'normalized_name': ingredient.normalized_name,
                        'original_text': ingredient.original_text,
                        'quantity': ingredient.quantity,
                        'unit': ingredient.unit,
                        'recipe_id': ingredient.recipe_id,
                        'recipe_name': ingredient.recipe_name,
                        'aldi_match': aldi_product_name,
                        'aldi_price': aldi_price,
                        'match_score': aldi_match['score'] if aldi_match else 0.0
                    }
                    
                    vectors_to_upsert.append({
                        'id': f"ingredient_{ingredient_id}",
                        'values': embedding,
                        'metadata': metadata
                    })
                    
                    ingredient_id += 1
                    
                    # Batch upsert every 100 vectors
                    if len(vectors_to_upsert) >= 100:
                        index.upsert(vectors=vectors_to_upsert)
                        vectors_to_upsert = []
                
                except Exception as e:
                    print(f"❌ Error processing ingredient: {e}")
                    continue
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
        
        print(f"✅ Uploaded {ingredient_id} ingredient vectors")
    
    def create_product_vectors(self):
        """Create and upload Aldi product vectors to Pinecone"""
        print("🛒 Creating Aldi product vectors...")
        
        index = self._get_index(self.product_index_name)
        
        vectors_to_upsert = []
        
        for product_id, product in tqdm(self.aldi_products.items(), desc="Processing products"):
            try:
                # Create embedding text for product
                product_text = f"{product.name} {product.category} aldi product food grocery"
                
                # Generate embedding
                embedding = self.sentence_model.encode(product_text).tolist()
                
                # Prepare metadata
                metadata = {
                    'type': 'product',
                    'name': product.name,
                    'price': product.price,
                    'category': product.category,
                    'url': product.url,
                    'image_url': product.image_url
                }
                
                vectors_to_upsert.append({
                    'id': product_id,
                    'values': embedding,
                    'metadata': metadata
                })
                
                # Batch upsert every 100 vectors
                if len(vectors_to_upsert) >= 100:
                    index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
            
            except Exception as e:
                print(f"❌ Error processing product {product_id}: {e}")
                continue
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
        
        print(f"✅ Uploaded {len(self.aldi_products)} product vectors")
    
    def create_recipe_embedding_text(self, recipe: Dict, ingredients: List[RecipeIngredient], cost_data: Dict) -> str:
        """Create comprehensive text for recipe embedding"""
        name = recipe.get('name', 'Unknown Recipe')
        chef = recipe.get('chef', 'Unknown')
        category = recipe.get('category', 'general')
        prep_time = recipe.get('prep_time', 'Unknown')
        cook_time = recipe.get('cook_time', 'Unknown')
        dietary_info = recipe.get('dietary_info', [])
        
        # Create ingredient list
        ingredient_names = [ing.normalized_name for ing in ingredients]
        
        # Budget category
        if cost_data['total_cost'] < 5:
            budget_category = 'ultra budget'
        elif cost_data['total_cost'] < 10:
            budget_category = 'budget friendly'
        elif cost_data['total_cost'] < 15:
            budget_category = 'moderate cost'
        else:
            budget_category = 'premium'
        
        text = f"""
        Recipe: {name}
        Chef: {chef}
        Category: {category}
        Cooking time: {prep_time} prep {cook_time} cook
        Serves: {cost_data['serves']} people
        
        Total cost: £{cost_data['total_cost']:.2f}
        Cost per serving: £{cost_data['cost_per_serving']:.2f}
        Budget category: {budget_category}
        
        Key ingredients: {', '.join(ingredient_names[:8])}
        Dietary: {', '.join(dietary_info) if dietary_info else 'no restrictions'}
        
        Aldi ingredient match rate: {cost_data['match_rate']:.0%}
        """.strip()
        
        return text
    
    def _get_index(self, index_name: str):
        """Get index object"""
        return self.pc.Index(index_name)
        """Create all vector embeddings and upload to Pinecone"""
        print("🚀 Creating all vector embeddings...")
        
        start_time = time.time()
        
        # Create vectors in order
        self.create_product_vectors()
        self.create_ingredient_vectors() 
        self.create_recipe_vectors()
        
        end_time = time.time()
        
        print(f"✅ All vectors created successfully!")
        print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
        
        # Print summary
        self.print_summary()
    
    def create_all_vectors(self):
        """Create all vector embeddings and upload to Pinecone"""
        print("🚀 Creating all vector embeddings...")
        
        start_time = time.time()
        
        # Create vectors in order
        self.create_product_vectors()
        self.create_ingredient_vectors() 
        self.create_recipe_vectors()
        
        end_time = time.time()
        
        print(f"✅ All vectors created successfully!")
        print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of created vectors"""
        print("\n📊 VECTORIZATION SUMMARY")
        print("=" * 50)
        
        try:
            recipe_index = self._get_index(self.recipe_index_name)
            ingredient_index = self._get_index(self.ingredient_index_name)
            product_index = self._get_index(self.product_index_name)
            
            recipe_stats = recipe_index.describe_index_stats()
            ingredient_stats = ingredient_index.describe_index_stats()
            product_stats = product_index.describe_index_stats()
            
            print(f"📝 Recipes: {recipe_stats['total_vector_count']} vectors")
            print(f"🥬 Ingredients: {ingredient_stats['total_vector_count']} vectors")
            print(f"🛒 Products: {product_stats['total_vector_count']} vectors")
            print(f"💾 Total vectors: {recipe_stats['total_vector_count'] + ingredient_stats['total_vector_count'] + product_stats['total_vector_count']}")
            
        except Exception as e:
            print(f"Could not get stats: {e}")
        
        print(f"\n🎯 Index Names:")
        print(f"   Recipes: {self.recipe_index_name}")
        print(f"   Ingredients: {self.ingredient_index_name}")
        print(f"   Products: {self.product_index_name}")
    
    def test_search(self):
        """Test the vector search functionality"""
        print("\n🧪 Testing vector search...")
        
        test_queries = [
            "cheap chicken dinner",
            "vegetarian pasta",
            "quick breakfast recipe",
            "beef stew"
        ]
        
        recipe_index = self._get_index(self.recipe_index_name)
        
        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            
            # Create query embedding
            query_embedding = self.sentence_model.encode(query).tolist()
            
            # Search recipes
            results = recipe_index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                filter={"total_cost": {"$lte": 15.0}}
            )
            
            for i, match in enumerate(results['matches'], 1):
                metadata = match['metadata']
                print(f"   {i}. {metadata['name']} - £{metadata['total_cost']:.2f} ({match['score']:.3f})")


def main():
    """Main function to run the vectorization"""
    print("🎯 Pinecone Recipe & Ingredient Vectorizer")
    print("=" * 60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load environment variables from .env file
    print("📁 Loading environment variables...")
    load_dotenv()
    
    # Get API keys from environment variables
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    print(f"🔑 Pinecone API key: {'✅ Found' if pinecone_api_key else '❌ Missing'}")
    print(f"🔑 OpenAI API key: {'✅ Found' if openai_api_key else '❌ Missing'}")
    
    if not pinecone_api_key:
        print("❌ PINECONE_API_KEY not found in .env file!")
        print("💡 Create a .env file in your project root with:")
        print("   PINECONE_API_KEY=your_pinecone_api_key_here")
        print("💡 Get your API key from: https://www.pinecone.io/")
        return
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in .env file!")
        print("💡 Add to your .env file:")
        print("   OPENAI_API_KEY=your_openai_api_key_here")
        return
    
    # File paths
    aldi_file = "scraped-data/aldi_products.json"
    bbc_file = "scraped-data/bbc_recipes.json"
    
    print(f"📁 Checking data files...")
    print(f"   Aldi products: {'✅ Found' if os.path.exists(aldi_file) else '❌ Missing'} ({aldi_file})")
    print(f"   BBC recipes: {'✅ Found' if os.path.exists(bbc_file) else '❌ Missing'} ({bbc_file})")
    
    # Check files exist
    for file_path in [aldi_file, bbc_file]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
    
    try:
        print("🚀 Starting vectorization process...")
        
        # Initialize vectorizer
        vectorizer = PineconeRecipeVectorizer(pinecone_api_key, openai_api_key)
        
        # Load data
        print("📥 Loading data files...")
        if not vectorizer.load_data(aldi_file, bbc_file):
            print("❌ Failed to load data")
            return
        
        # Create indexes
        print("🔧 Setting up Pinecone indexes...")
        if not vectorizer.create_indexes():
            print("❌ Failed to create indexes")
            return
        
        # Create all vectors
        print("🎯 Starting vector creation...")
        vectorizer.create_all_vectors()
        
        # Test search functionality
        vectorizer.test_search()
        
        print("\n🎉 Vectorization complete!")
        print("💡 You can now update your chat.py to use Pinecone instead of Qdrant")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()