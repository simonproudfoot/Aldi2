#!/usr/bin/env python3
"""
Enhanced Recipe RAG Backend with Pinecone and conversation context
"""

import os
import json
import re
import numpy as np
from typing import List, Dict, Optional, Set, Any
from collections import defaultdict, Counter
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import httpx
import asyncio
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global variables
model = None
pc = None
openai_client = None
aldi_products = {}
conversations = {}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone index names (matching your vectorizer)
RECIPE_INDEX_NAME = "aldi-recipes"
INGREDIENT_INDEX_NAME = "aldi-ingredients"  
PRODUCT_INDEX_NAME = "aldi-products"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    global model, pc, aldi_products, openai_client
    
    try:
        print("🚀 Starting Enhanced Recipe RAG Backend with Pinecone...")
        
        if not OPENAI_API_KEY:
            print("❌ ERROR: OPENAI_API_KEY not found! Please set the environment variable.")
            raise ValueError("OpenAI API key is required")
        
        if not PINECONE_API_KEY:
            print("❌ ERROR: PINECONE_API_KEY not found! Please set the environment variable.")
            raise ValueError("Pinecone API key is required")
        
        # Initialize OpenAI client
        print("🔧 Initializing OpenAI client...")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
        
        # Initialize sentence transformer for embeddings
        print("🔧 Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Using original model to match Pinecone index
        print("✅ Sentence transformer model loaded")
        
        # Initialize Pinecone
        print("🔧 Connecting to Pinecone...")
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if indexes exist
            existing_indexes = pc.list_indexes().names()
            required_indexes = [RECIPE_INDEX_NAME, INGREDIENT_INDEX_NAME, PRODUCT_INDEX_NAME]
            
            missing_indexes = [idx for idx in required_indexes if idx not in existing_indexes]
            
            if missing_indexes:
                print(f"⚠️ Missing Pinecone indexes: {missing_indexes}")
                print("💡 Run recipe_vectorizer.py first to create the indexes")
            else:
                print(f"✅ Connected to Pinecone indexes: {required_indexes}")
                
        except Exception as e:
            print(f"⚠️ Pinecone connection failed: {e}")
            pc = None
        
        # Load products from Pinecone instead of JSON file
        print("🔧 Loading products from Pinecone...")
        if pc:
            try:
                aldi_products = await get_all_products_from_pinecone()
                print(f"✅ Loaded {len(aldi_products)} Aldi products from Pinecone")
            except Exception as e:
                print(f"⚠️ Failed to load products from Pinecone: {e}")
                aldi_products = {}
        else:
            aldi_products = {}
            print("⚠️ Could not load products - Pinecone unavailable")
        
        print("🎉 Backend ready with Pinecone vector database!")
        print("📡 Server starting on http://localhost:8000")
        
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Aldi Recipe Assistant with Pinecone", 
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation storage (use Redis in production)
conversations = {}

class QueryRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    max_budget: Optional[float] = 20.0
    dietary_restrictions: Optional[List[str]] = []
    max_prep_time: Optional[str] = None
    cuisine_preference: Optional[str] = None

class RecipeResponse(BaseModel):
    ai_response: str
    recipes: List[Dict]
    total_recipes_found: int
    search_query: str
    conversation_id: str
    shopping_list: Optional[Dict] = None
    cost_breakdown: Optional[Dict] = None
    product_results: Optional[List[Dict]] = None  # Add product results field

class ConversationContext:
    """Manages conversation context and history"""
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages = []
        self.last_recipes = []
        self.last_search_params = {}
        self.created_at = datetime.now()
    
    def add_message(self, role: str, content: str, recipes: List[Dict] = None):
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "recipes": recipes or []
        }
        self.messages.append(message)
        
        if recipes:
            self.last_recipes = recipes
    
    def get_recent_context(self, max_messages: int = 6) -> str:
        """Get recent conversation context for AI"""
        if not self.messages:
            return ""
        
        recent_messages = self.messages[-max_messages:]
        context_parts = []
        
        for msg in recent_messages:
            if msg["role"] == "user":
                context_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                # Include recipe info in context
                if msg.get("recipes"):
                    recipe_names = [r.get('name', 'Unknown') for r in msg["recipes"][:3]]
                    context_parts.append(f"Assistant: Recommended recipes: {', '.join(recipe_names)}")
                else:
                    # Truncate long responses for context
                    content = msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content']
                    context_parts.append(f"Assistant: {content}")
        
        return "\n".join(context_parts)
    
    def has_recent_recipes(self) -> bool:
        return len(self.last_recipes) > 0

async def get_all_products_from_pinecone(limit: int = 2000) -> Dict[str, Dict]:
    """Load all Aldi products from Pinecone instead of JSON file"""
    if not pc:
        print("⚠️ Pinecone not available, cannot load products")
        return {}
    
    try:
        product_index = pc.Index(PRODUCT_INDEX_NAME)
        
        # Get all products using a dummy vector and high limit
        results = product_index.query(
            vector=[0] * 384,  # Original vector dimension for all-MiniLM-L6-v2
            top_k=limit,
            include_metadata=True,
            filter={}
        )
        
        products = {}
        for match in results['matches']:
            metadata = match['metadata']
            name = metadata.get('name', '').lower()
            products[name] = {
                'name': metadata.get('name', ''),
                'price': metadata.get('price', 0),
                'category': metadata.get('category', 'unknown'),
                'url': metadata.get('url', ''),
                'image_url': metadata.get('image_url', ''),
                'id': match['id']
            }
        
        return products
        
    except Exception as e:
        print(f"❌ Error loading products from Pinecone: {e}")
        return {}

async def search_product_by_name(product_name: str) -> Optional[Dict]:
    """Search for a specific product by name in Pinecone"""
    if not pc:
        return None
    
    try:
        product_index = pc.Index(PRODUCT_INDEX_NAME)
        
        # Create embedding for the product name
        query_embedding = model.encode(product_name.lower()).tolist()
        
        # Search for the specific product
        results = product_index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            filter={}
        )
        
        if results['matches'] and results['matches'][0]['score'] > 0.8:
            metadata = results['matches'][0]['metadata']
            return {
                'name': metadata.get('name', ''),
                'price': metadata.get('price', 0),
                'category': metadata.get('category', ''),
                'url': metadata.get('url', ''),
                'image_url': metadata.get('image_url', ''),
                'id': results['matches'][0]['id']
            }
        
        return None
        
    except Exception as e:
        print(f"❌ Error searching for product: {e}")
        return None

def is_price_query(message: str) -> bool:
    """Check if user is asking for direct product prices"""
    price_patterns = [
        r'\bhow much (is|are|does|do|cost)',
        r'\bprice of',
        r'\bcost of',
        r'\bhow (much|expensive)',
        r'\baldi.*price',
        r'\baldi.*cost',
        r'\bcheapest\b',
        r'\bmost expensive\b',
        r'\bfind.*cheapest',
        r'\bwhat.*costs?',
        r'\bhow much.*buy',
        r'\bbuy.*cheapest'
    ]
    
    return any(re.search(pattern, message.lower()) for pattern in price_patterns)

def extract_product_query(message: str) -> str:
    """Extract clean product name from price query"""
    query = message.lower()
    
    # Remove price question words
    price_words = [
        'how much is', 'how much are', 'how much does', 'how much do',
        'what is the price of', 'what does', 'cost',
        'price of', 'cost of', 'how expensive is', 'how expensive are'
    ]
    
    for phrase in price_words:
        query = query.replace(phrase, '')
    
    # Remove Aldi references
    query = re.sub(r'\b(at aldi|from aldi|aldi|at the aldi)\b', '', query)
    
    # Remove extra words
    query = re.sub(r'\b(the|a|an|some)\b', '', query)
    
    # Clean up
    query = re.sub(r'\s+', ' ', query).strip()
    
    return query

def get_or_create_conversation(conversation_id: str = None) -> ConversationContext:
    """Get existing conversation or create new one"""
    if not conversation_id:
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if conversation_id not in conversations:
        conversations[conversation_id] = ConversationContext(conversation_id)
    
    return conversations[conversation_id]

# Load Aldi products at startup
aldi_products = {}
    
@app.on_event("startup")
async def startup_event():
    """Initialize models and connections"""
    global model, pc, aldi_products, openai_client
    
    print("🚀 Starting Enhanced Recipe RAG Backend with Pinecone...")
    
    if not OPENAI_API_KEY:
        print("❌ ERROR: OPENAI_API_KEY not found! Please set the environment variable.")
        raise ValueError("OpenAI API key is required")
    
    if not PINECONE_API_KEY:
        print("❌ ERROR: PINECONE_API_KEY not found! Please set the environment variable.")
        raise ValueError("Pinecone API key is required")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI client initialized")
    
    # Initialize sentence transformer for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Sentence transformer model loaded")
    
    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Check if indexes exist
        existing_indexes = pc.list_indexes().names()
        required_indexes = [RECIPE_INDEX_NAME, INGREDIENT_INDEX_NAME, PRODUCT_INDEX_NAME]
        
        missing_indexes = [idx for idx in required_indexes if idx not in existing_indexes]
        
        if missing_indexes:
            print(f"⚠️ Missing Pinecone indexes: {missing_indexes}")
            print("💡 Run recipe_vectorizer.py first to create the indexes")
        else:
            print(f"✅ Connected to Pinecone indexes: {required_indexes}")
            
    except Exception as e:
        print(f"⚠️ Pinecone connection failed: {e}")
        pc = None
    
    aldi_products = load_aldi_products()
    print(f"✅ Loaded {len(aldi_products)} Aldi products for price lookup")
    print("🎉 Backend ready with Pinecone vector database!")

async def search_pinecone_recipes(
    query: str,
    max_budget: float = 20.0,
    dietary_restrictions: List[str] = [],
    max_results: int = 8
) -> List[Dict]:
    """Search recipes using Pinecone vector database with improved filtering"""
    
    if not pc:
        print("⚠️ Pinecone not available, using fallback")
        return await search_recipes_openai_only(query, max_budget, dietary_restrictions, max_results)
    
    try:
        # Get recipe index
        recipe_index = pc.Index(RECIPE_INDEX_NAME)
        
        # Create query embedding
        query_embedding = model.encode(query).tolist()
        
        # Build filters
        filters = {"total_cost": {"$lte": max_budget}}
        
        # Search recipes with more results to filter better
        results = recipe_index.query(
            vector=query_embedding,
            top_k=max_results * 3,  # Get more results to filter through
            include_metadata=True,
            filter=filters
        )
        
        formatted_results = []
        query_lower = query.lower()
        
        # Extract key terms from query for better matching
        key_terms = []
        if 'curry' in query_lower:
            key_terms = ['curry', 'indian', 'spice', 'masala', 'tikka', 'korma', 'biryani', 'dal', 'vindaloo']
        elif 'pasta' in query_lower:
            key_terms = ['pasta', 'spaghetti', 'penne', 'linguine', 'bolognese', 'carbonara', 'arrabiata']
        elif 'stir fry' in query_lower or 'stirfry' in query_lower:
            key_terms = ['stir', 'fry', 'wok', 'asian', 'chinese', 'thai']
        elif 'soup' in query_lower:
            key_terms = ['soup', 'broth', 'bisque', 'chowder', 'minestrone']
        elif 'salad' in query_lower:
            key_terms = ['salad', 'leaves', 'lettuce', 'greens', 'caesar', 'greek']
        elif 'pizza' in query_lower:
            key_terms = ['pizza', 'margherita', 'pepperoni', 'base', 'dough']
        elif 'breakfast' in query_lower:
            key_terms = ['breakfast', 'pancake', 'eggs', 'bacon', 'toast', 'cereal', 'porridge']
        elif 'dinner' in query_lower:
            key_terms = ['dinner', 'main', 'evening', 'roast', 'casserole']
        
        for match in results['matches']:
            metadata = match['metadata']
            
            # Apply dietary restrictions filter
            if dietary_restrictions:
                recipe_dietary_info = metadata.get('dietary_info', [])
                if isinstance(recipe_dietary_info, str):
                    recipe_dietary_info = [recipe_dietary_info]
                
                # Check if recipe meets dietary restrictions
                meets_restrictions = False
                for restriction in dietary_restrictions:
                    if any(restriction.lower() in info.lower() for info in recipe_dietary_info):
                        meets_restrictions = True
                        break
                
                # Skip if doesn't meet restrictions
                if not meets_restrictions:
                    continue
            
            recipe_name = metadata.get('name', '').lower()
            recipe_category = metadata.get('category', '').lower()
            
            # Better relevance scoring
            relevance_score = match['score']
            
            # Boost score if recipe matches key terms
            if key_terms:
                for term in key_terms:
                    if term in recipe_name or term in recipe_category:
                        relevance_score += 0.2
                        break
                
                # Penalize if it doesn't match any key terms for specific queries
                if query_lower in ['curry', 'pasta', 'pizza'] and not any(term in recipe_name for term in key_terms):
                    relevance_score -= 0.3
            
            # Skip low relevance results for specific food types
            if key_terms and relevance_score < 0.4:
                continue
            
            recipe_data = {
                'name': metadata.get('name', 'Unknown Recipe'),
                'chef': metadata.get('chef', 'Unknown'),
                'total_cost': metadata.get('total_cost', 0),
                'cost_per_serving': metadata.get('cost_per_serving', 0),
                'serves': metadata.get('serves', 4),
                'category': metadata.get('category', 'general'),
                'prep_time': metadata.get('prep_time', 'Unknown'),
                'cook_time': metadata.get('cook_time', 'Unknown'),
                'dietary_info': metadata.get('dietary_info', []),
                'url': metadata.get('url', ''),
                'similarity_score': relevance_score,
                'match_rate': metadata.get('match_rate', 0.8),
                'ingredients_count': metadata.get('ingredients_count', 0),
                'ingredient_names': metadata.get('ingredient_names', []),
                'recipe_id': match['id']
            }
            formatted_results.append(recipe_data)
        
        # Sort by relevance score and return top results
        formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # If we still don't have good results, fall back to OpenAI generation
        if len(formatted_results) < 3 and key_terms:
            print(f"⚠️ Limited Pinecone results for '{query}', supplementing with OpenAI")
            openai_recipes = await search_recipes_openai_only(query, max_budget, dietary_restrictions, max_results - len(formatted_results))
            formatted_results.extend(openai_recipes)
        
        return formatted_results[:max_results]
        
    except Exception as e:
        print(f"❌ Pinecone recipe search error: {e}")
        return await search_recipes_openai_only(query, max_budget, dietary_restrictions, max_results)

async def search_pinecone_ingredients(query: str, max_results: int = 10) -> List[Dict]:
    """Search ingredients using Pinecone"""
    
    if not pc:
        return []
    
    try:
        ingredient_index = pc.Index(INGREDIENT_INDEX_NAME)
        
        # Create query embedding
        query_embedding = model.encode(query).tolist()
        
        # Search ingredients
        results = ingredient_index.query(
            vector=query_embedding,
            top_k=max_results,
            include_metadata=True,
            filter={}
        )
        
        formatted_results = []
        for match in results['matches']:
            metadata = match['metadata']
            
            ingredient_data = {
                'normalized_name': metadata.get('normalized_name', ''),
                'original_text': metadata.get('original_text', ''),
                'recipe_id': metadata.get('recipe_id', ''),
                'recipe_name': metadata.get('recipe_name', ''),
                'aldi_match': metadata.get('aldi_match', ''),
                'aldi_price': metadata.get('aldi_price', 0),
                'match_score': metadata.get('match_score', 0),
                'similarity_score': match['score']
            }
            formatted_results.append(ingredient_data)
        
        return formatted_results
        
    except Exception as e:
        print(f"❌ Pinecone ingredient search error: {e}")
        return []

async def search_pinecone_products(query: str, max_results: int = 5) -> List[Dict]:
    """Search Aldi products using Pinecone with improved filtering"""
    
    if not pc:
        return []
    
    try:
        product_index = pc.Index(PRODUCT_INDEX_NAME)
        
        # Clean and analyze the query
        clean_query = extract_product_query(query) if is_price_query(query) else query.lower().strip()
        
        # Create query embedding
        query_embedding = model.encode(clean_query).tolist()
        
        # Search products - get more results to filter through
        results = product_index.query(
            vector=query_embedding,
            top_k=max_results * 3,  # Get more to filter
            include_metadata=True,
            filter={}
        )
        
        formatted_results = []
        query_words = set(clean_query.lower().split())
        
        # Define product categories and their keywords
        category_keywords = {
            'meat': ['beef', 'chicken', 'pork', 'lamb', 'steak', 'mince', 'bacon', 'ham', 'sausage'],
            'dairy': ['milk', 'cheese', 'butter', 'yogurt', 'cream'],
            'produce': ['apple', 'banana', 'orange', 'potato', 'onion', 'carrot', 'lettuce', 'tomato', 'avocado'],
            'alcohol': ['beer', 'wine', 'lager', 'ale', 'spirits', 'whiskey', 'vodka'],
            'pantry': ['pasta', 'rice', 'bread', 'cereal', 'flour', 'sugar', 'oil'],
            'frozen': ['frozen', 'ice cream', 'pizza']
        }
        
        # Determine expected category from query
        expected_categories = []
        for category, keywords in category_keywords.items():
            if any(keyword in clean_query.lower() for keyword in keywords):
                expected_categories.append(category)
        
        for match in results['matches']:
            metadata = match['metadata']
            product_name = metadata.get('name', '').lower()
            product_category = metadata.get('category', '').lower()
            
            # Skip obviously wrong matches
            should_skip = False
            
            # If we have specific expectations, filter accordingly
            if expected_categories:
                product_matches_category = False
                
                for expected_cat in expected_categories:
                    if expected_cat == 'meat':
                        # Check if it's actually meat
                        if any(keyword in product_name for keyword in category_keywords['meat']):
                            product_matches_category = True
                        # Skip alcohol that got miscategorized as beef
                        elif any(keyword in product_name for keyword in category_keywords['alcohol']):
                            should_skip = True
                            break
                    elif expected_cat == 'produce':
                        if any(keyword in product_name for keyword in category_keywords['produce']):
                            product_matches_category = True
                    elif expected_cat == 'dairy':
                        if any(keyword in product_name for keyword in category_keywords['dairy']):
                            product_matches_category = True
                    elif expected_cat == 'alcohol':
                        if any(keyword in product_name for keyword in category_keywords['alcohol']):
                            product_matches_category = True
                
                if should_skip:
                    continue
                    
                # Only include if it matches expected category or has high similarity
                if not product_matches_category and match['score'] < 0.7:
                    continue
            
            # Additional filtering for common mismatches
            if 'beef' in clean_query.lower():
                # Skip if it's clearly not beef
                if any(word in product_name for word in ['lager', 'beer', 'wine', 'avocado', 'apple', 'banana']):
                    continue
                # Only include if it actually contains beef-related terms or has very high similarity
                if not any(word in product_name for word in ['beef', 'steak', 'mince']) and match['score'] < 0.8:
                    continue
            
            if 'chicken' in clean_query.lower():
                if any(word in product_name for word in ['lager', 'beer', 'wine', 'avocado']) and match['score'] < 0.9:
                    continue
            
            # Calculate relevance score
            relevance_score = match['score']
            
            # Boost score for exact word matches
            for query_word in query_words:
                if query_word in product_name:
                    relevance_score += 0.2
            
            # Only include products above a minimum relevance threshold
            if relevance_score < 0.5:
                continue
            
            product_data = {
                'name': metadata.get('name', ''),
                'price': metadata.get('price', 0),
                'category': metadata.get('category', ''),
                'url': metadata.get('url', ''),
                'image_url': metadata.get('image_url', ''),
                'similarity_score': relevance_score
            }
            formatted_results.append(product_data)
        
        # Sort by relevance and return top results
        formatted_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # If we still don't have good results, try alternative search
        if len(formatted_results) < 2:
            print(f"⚠️ Limited relevant products found for '{clean_query}', trying alternative search")
            # You could implement a keyword-based fallback here
            
        return formatted_results[:max_results]
        
    except Exception as e:
        print(f"❌ Pinecone product search error: {e}")
        return []

async def get_recipe_ingredients_from_pinecone(recipe_id: str) -> List[Dict]:
    """Get detailed ingredients for a specific recipe"""
    
    if not pc:
        return []
    
    try:
        ingredient_index = pc.Index(INGREDIENT_INDEX_NAME)
        
        # Search for ingredients by recipe_id
        results = ingredient_index.query(
            vector=[0] * 384,  # Original vector dimension
            top_k=50,
            include_metadata=True,
            filter={"recipe_id": recipe_id}
        )
        
        ingredients = []
        for match in results['matches']:
            metadata = match['metadata']
            
            # Get Aldi product info using Pinecone instead of memory
            aldi_match_name = metadata.get('aldi_match', '')
            aldi_product = None
            if aldi_match_name:
                aldi_product = await search_product_by_name(aldi_match_name)
            
            ingredient_data = {
                'normalized_name': metadata.get('normalized_name', ''),
                'original_text': metadata.get('original_text', ''),
                'quantity': metadata.get('quantity', ''),
                'unit': metadata.get('unit', ''),
                'aldi_match': aldi_match_name,
                'aldi_price': aldi_product['price'] if aldi_product else metadata.get('aldi_price', 0),
                'aldi_url': aldi_product['url'] if aldi_product else '',
                'aldi_image': aldi_product['image_url'] if aldi_product else '',
                'match_score': metadata.get('match_score', 0)
            }
            ingredients.append(ingredient_data)
        
        return ingredients
        
    except Exception as e:
        print(f"❌ Error getting recipe ingredients: {e}")
        return []

async def calculate_shopping_list(recipes: List[Dict]) -> Dict:
    """Calculate shopping list and total cost for selected recipes"""
    
    shopping_list = {}
    total_cost = 0.0
    
    for recipe in recipes:
        recipe_id = recipe.get('recipe_id', '')
        if not recipe_id:
            continue
            
        # Get detailed ingredients for this recipe
        ingredients = await get_recipe_ingredients_from_pinecone(recipe_id)
        
        for ingredient in ingredients:
            aldi_match = ingredient.get('aldi_match', '')
            aldi_price = ingredient.get('aldi_price', 0)
            original_text = ingredient.get('original_text', '')
            quantity = ingredient.get('quantity', '')
            unit = ingredient.get('unit', '')
            
            if aldi_match and aldi_price > 0:
                # Add to shopping list
                if aldi_match not in shopping_list:
                    shopping_list[aldi_match] = {
                        'name': aldi_match,
                        'price': aldi_price,
                        'used_in_recipes': [],
                        'total_needed': 0
                    }
                
                shopping_list[aldi_match]['used_in_recipes'].append({
                    'recipe': recipe.get('name', ''),
                    'ingredient_text': original_text,
                    'quantity': quantity,
                    'unit': unit
                })
                
                # For simplicity, assume we need one of each product
                shopping_list[aldi_match]['total_needed'] = 1
    
    # Calculate total cost
    for item in shopping_list.values():
        total_cost += item['price'] * item['total_needed']
    
    return {
        'items': list(shopping_list.values()),
        'total_cost': round(total_cost, 2),
        'item_count': len(shopping_list)
    }

async def search_recipes_openai_only(
    query: str, 
    max_budget: float = 20.0,
    dietary_restrictions: List[str] = [],
    max_results: int = 8
) -> List[Dict]:
    """OpenAI-only recipe search when Pinecone is unavailable or has poor results"""
    print("🤖 Using OpenAI recipe generation")
    
    try:
        dietary_text = f" with {', '.join(dietary_restrictions)} dietary restrictions" if dietary_restrictions else ""
        
        # Create more specific prompts based on query
        recipe_type = "recipes"
        if "curry" in query.lower():
            recipe_type = "curry recipes (like chicken curry, vegetable curry, dal, etc.)"
        elif "pasta" in query.lower():
            recipe_type = "pasta recipes (like spaghetti, penne dishes, etc.)"
        elif "pizza" in query.lower():
            recipe_type = "pizza recipes"
        elif "soup" in query.lower():
            recipe_type = "soup recipes"
        elif "salad" in query.lower():
            recipe_type = "salad recipes"
        elif "breakfast" in query.lower():
            recipe_type = "breakfast recipes"
        elif "stir fry" in query.lower():
            recipe_type = "stir fry recipes"
        
        messages = [
            {
                "role": "system", 
                "content": """You are a recipe recommendation system. Generate realistic recipe suggestions with accurate cost estimates for Aldi UK products. 

CRITICAL: Make sure the recipes match the specific food type requested (e.g., if someone asks for curry, provide actual curry recipes, not potato dishes).

Return exactly 3-5 recipes in JSON format. Each recipe should have:
- name: descriptive recipe name that matches the request
- chef: realistic chef name
- total_cost: estimated cost in pounds (realistic for Aldi UK, £3-15 range)
- cost_per_serving: total_cost divided by serves
- serves: number of people it serves (2-6)
- prep_time: preparation time (like "15 mins", "30 mins")
- cook_time: cooking time (like "20 mins", "45 mins")
- dietary_info: array of dietary information if applicable
- category: food category that matches the request
- url: placeholder BBC Food URL

Return ONLY valid JSON array, no other text."""
            },
            {
                "role": "user", 
                "content": f"Generate {min(max_results, 4)} budget-friendly {recipe_type} for '{query}' under £{max_budget}{dietary_text}. Focus on recipes that actually match what was requested - don't suggest potato dishes for curry requests!"
            }
        ]
        
        response = await call_openai_api(messages, "gpt-3.5-turbo")
        
        # Try to parse JSON response
        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                recipes_data = json.loads(json_str)
                
                formatted_recipes = []
                for recipe in recipes_data:
                    formatted_recipe = {
                        'name': recipe.get('name', 'Generated Recipe'),
                        'chef': recipe.get('chef', 'AI Chef'),
                        'total_cost': recipe.get('total_cost', 8.0),
                        'cost_per_serving': recipe.get('cost_per_serving', 4.0),
                        'serves': recipe.get('serves', 2),
                        'category': recipe.get('category', 'main'),
                        'prep_time': recipe.get('prep_time', '15 mins'),
                        'cook_time': recipe.get('cook_time', '25 mins'),
                        'dietary_info': recipe.get('dietary_info', dietary_restrictions),
                        'url': recipe.get('url', 'https://www.bbc.co.uk/food'),
                        'similarity_score': 0.85,
                        'match_rate': recipe.get('match_rate', 0.85),
                        'recipe_id': f"ai_recipe_{len(formatted_recipes)}",
                        'ingredients_count': 6,
                        'ingredient_names': []
                    }
                    formatted_recipes.append(formatted_recipe)
                
                return formatted_recipes[:max_results]
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse OpenAI recipe JSON: {e}")
            
        return []
        
    except Exception as e:
        print(f"OpenAI recipe generation failed: {e}")
        return []

async def call_openai_api(messages: List[Dict], model_name: str = "gpt-4") -> str:
    """Call OpenAI API with GPT-4"""
    if not openai_client:
        return "❌ OpenAI client not initialized"
    
    try:
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
            timeout=30.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API error: {e}")
        # Fallback to cheaper model if GPT-4 fails
        if model_name == "gpt-4":
            try:
                print("Falling back to GPT-3.5-turbo")
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=30.0
                )
                return response.choices[0].message.content
            except Exception as e2:
                print(f"GPT-3.5-turbo also failed: {e2}")
        
        return f"❌ AI service error: {str(e)}"

def analyze_query_intent(message: str) -> Dict[str, Any]:
    """Analyze what the user wants - product pricing or recipe search"""
    message_lower = message.lower()
    
    intent = {
        "type": "recipe",  # Default
        "confidence": 0.5,
        "specific_product": None,
        "recipe_type": None
    }
    
    # Strong price indicators
    price_indicators = [
        'cheapest', 'most expensive', 'price', 'cost', 'how much', 
        'buy', 'afford', 'budget for', 'spend on'
    ]
    
    # Recipe indicators
    recipe_indicators = [
        'recipe', 'cook', 'make', 'prepare', 'meal', 'dinner', 'lunch', 
        'breakfast', 'dish', 'serve', 'serves', 'minutes', 'hours'
    ]
    
    # Count indicators
    price_score = sum(1 for indicator in price_indicators if indicator in message_lower)
    recipe_score = sum(1 for indicator in recipe_indicators if indicator in message_lower)
    
    # Extract product mentions
    product_terms = [
        'steak', 'chicken', 'beef', 'pork', 'fish', 'salmon', 'eggs', 
        'milk', 'cheese', 'bread', 'avocado', 'tomato', 'potato'
    ]
    
    mentioned_products = [term for term in product_terms if term in message_lower]
    
    # Decide intent
    if price_score > recipe_score or any(term in message_lower for term in ['cheapest', 'most expensive', 'how much']):
        intent["type"] = "price"
        intent["confidence"] = 0.8
        if mentioned_products:
            intent["specific_product"] = mentioned_products[0]
    
    # Look for recipe types
    recipe_types = {
        'curry': ['curry', 'indian', 'spice'],
        'pasta': ['pasta', 'spaghetti', 'noodles'],
        'steak': ['steak recipe', 'cooking steak', 'steak dinner'],
        'soup': ['soup', 'broth'],
        'salad': ['salad', 'greens'],
        'pizza': ['pizza']
    }
    
    for recipe_type, keywords in recipe_types.items():
        if any(keyword in message_lower for keyword in keywords):
            intent["recipe_type"] = recipe_type
            if 'recipe' in message_lower or 'cook' in message_lower or 'make' in message_lower:
                intent["type"] = "recipe"
                intent["confidence"] = 0.9
    
    return intent

@app.post("/chat", response_model=RecipeResponse)
async def chat_with_assistant(request: QueryRequest):
    """Enhanced chat with Pinecone vector database"""
    try:
        # Get or create conversation
        conversation = get_or_create_conversation(request.conversation_id)
        
        # Add user message to conversation
        conversation.add_message("user", request.message)
        
        recipes = []
        ai_response = ""
        shopping_list = None
        
        # Check if this is a price query
        if is_price_query(request.message):
            print(f"Handling price query: {request.message}")
            
            # Search products using Pinecone
            products = await search_pinecone_products(request.message, max_results=5)
            
            if products:
                # Format as product cards instead of text
                ai_response = f"Found {len(products)} Aldi products:"
                
                # Convert products to a format the frontend can display as product cards
                product_cards = []
                for product in products:
                    product_cards.append({
                        'type': 'product',
                        'name': product['name'],
                        'price': product['price'],
                        'url': product.get('url', ''),
                        'image_url': product.get('image_url', ''),
                        'category': product.get('category', ''),
                        'similarity_score': product.get('similarity_score', 0)
                    })
                
                # Add assistant response to conversation
                conversation.add_message("assistant", ai_response, [])
                
                return RecipeResponse(
                    ai_response=ai_response,
                    recipes=[],  # No recipes for price queries
                    total_recipes_found=0,
                    search_query=request.message,
                    conversation_id=conversation.conversation_id,
                    shopping_list=None,
                    product_results=product_cards  # Add product results
                )
            else:
                ai_response = f"I couldn't find specific pricing for '{request.message}' in the Aldi database."
                
                # Add assistant response to conversation
                conversation.add_message("assistant", ai_response, [])
                
                return RecipeResponse(
                    ai_response=ai_response,
                    recipes=[],
                    total_recipes_found=0,
                    search_query=request.message,
                    conversation_id=conversation.conversation_id,
                    shopping_list=None,
                    product_results=[]
                )
        
        else:
            # Search for recipes using Pinecone
            print(f"Performing recipe search using Pinecone")
            
            recipes = await search_pinecone_recipes(
                query=request.message,
                max_budget=request.max_budget,
                dietary_restrictions=request.dietary_restrictions,
                max_results=8
            )
            
            # Calculate shopping list if recipes found
            if recipes:
                shopping_list = await calculate_shopping_list(recipes)
            
            # Build AI context
            conversation_context = conversation.get_recent_context()
            
            recipe_context = ""
            if recipes:
                recipe_context = f"\nFound {len(recipes)} recipes:\n"
                for i, recipe in enumerate(recipes, 1):
                    dietary_text = ", ".join(recipe['dietary_info']) if recipe['dietary_info'] else "No specific dietary info"
                    recipe_context += f"{i}. **{recipe['name']}** by {recipe['chef']}\n"
                    recipe_context += f"   - Cost: £{recipe['total_cost']:.2f} total (£{recipe['cost_per_serving']:.2f}/serving)\n"
                    recipe_context += f"   - Time: {recipe['prep_time']} prep, {recipe['cook_time']} cook\n"
                    recipe_context += f"   - Dietary: {dietary_text}\n"
                    recipe_context += f"   - Ingredients matched: {recipe.get('ingredients_count', 0)}\n\n"
            
            system_prompt = f"""You are a helpful cooking assistant specializing in budget-friendly Aldi recipes using a Pinecone vector database.

Your expertise:
- Budget-conscious meal planning with Aldi products
- Practical cooking advice and accurate cost breakdowns
- Recipe recommendations based on vector similarity search

Guidelines:
- Be conversational and enthusiastic but accurate
- Reference specific costs and details from the recipe data
- Don't make up information not provided in the recipe data
- Keep responses focused and practical
- When recommending recipes, explain why they're good choices
- Mention the ingredient matching and cost calculations when relevant
- Use emojis sparingly and naturally"""

            user_message = f"""User Request: {request.message}
Budget: £{request.max_budget}
Dietary Restrictions: {', '.join(request.dietary_restrictions) if request.dietary_restrictions else 'None'}

{f"Conversation History:\n{conversation_context}" if conversation_context else ""}

{recipe_context if recipe_context else "No suitable recipes found within the criteria."}

Provide a helpful response about these recipe recommendations, including cost breakdowns and ingredient matching information."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            ai_response = await call_openai_api(messages, "gpt-4")
        
        # Add assistant response to conversation
        conversation.add_message("assistant", ai_response, recipes)
        
        return RecipeResponse(
            ai_response=ai_response,
            recipes=recipes,
            total_recipes_found=len(recipes),
            search_query=request.message,
            conversation_id=conversation.conversation_id,
            shopping_list=shopping_list
        )
        
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Enhanced Aldi Recipe Assistant API with Pinecone",
        "status": "healthy",
        "features": ["pinecone_vectors", "ingredient_matching", "cost_calculation", "shopping_list"],
        "openai_configured": bool(OPENAI_API_KEY),
        "pinecone_configured": bool(PINECONE_API_KEY),
        "indexes": [RECIPE_INDEX_NAME, INGREDIENT_INDEX_NAME, PRODUCT_INDEX_NAME]
    }

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id in conversations:
        conv = conversations[conversation_id]
        return {
            "conversation_id": conversation_id,
            "message_count": len(conv.messages),
            "created_at": conv.created_at.isoformat(),
            "has_recent_recipes": conv.has_recent_recipes(),
            "messages": conv.messages[-10:]
        }
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        if pc:
            recipe_index = pc.Index(RECIPE_INDEX_NAME)
            ingredient_index = pc.Index(INGREDIENT_INDEX_NAME)
            product_index = pc.Index(PRODUCT_INDEX_NAME)
            
            recipe_stats = recipe_index.describe_index_stats()
            ingredient_stats = ingredient_index.describe_index_stats()
            product_stats = product_index.describe_index_stats()
            
            return {
                "total_recipes": recipe_stats['total_vector_count'],
                "total_ingredients": ingredient_stats['total_vector_count'],
                "total_products": product_stats['total_vector_count'],
                "recipe_index": RECIPE_INDEX_NAME,
                "ingredient_index": INGREDIENT_INDEX_NAME,
                "product_index": PRODUCT_INDEX_NAME,
                "openai_model": "gpt-4",
                "embedding_model": "all-MiniLM-L6-v2",
                "pinecone_status": "connected"
            }
        else:
            return {
                "total_recipes": 0,
                "total_ingredients": 0,
                "total_products": 0,
                "openai_model": "gpt-4",
                "embedding_model": "all-MiniLM-L6-v2",
                "pinecone_status": "disconnected",
                "note": "Using OpenAI-only mode"
            }
    except Exception as e:
        return {
            "error": str(e),
            "pinecone_status": "error",
            "openai_model": "gpt-4",
            "embedding_model": "all-MiniLM-L6-v2"
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        test_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        openai_status = "healthy" if test_response else "error"
    except Exception as e:
        openai_status = f"error: {str(e)}"
    
    try:
        if pc:
            indexes = pc.list_indexes().names()
            pinecone_status = "healthy" if len(indexes) > 0 else "no_indexes"
        else:
            pinecone_status = "not_connected"
    except Exception as e:
        pinecone_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "openai": openai_status,
        "pinecone": pinecone_status,
        "aldi_products_loaded": len(aldi_products),
        "conversations_active": len(conversations)
    }

@app.get("/search/recipes")
async def search_recipes_endpoint(
    query: str,
    max_budget: float = 20.0,
    dietary_restrictions: str = "",
    max_results: int = 8
):
    """Direct recipe search endpoint"""
    try:
        dietary_list = [r.strip() for r in dietary_restrictions.split(",")] if dietary_restrictions else []
        
        recipes = await search_pinecone_recipes(
            query=query,
            max_budget=max_budget,
            dietary_restrictions=dietary_list,
            max_results=max_results
        )
        
        return {
            "recipes": recipes,
            "total_found": len(recipes),
            "search_query": query,
            "filters": {
                "max_budget": max_budget,
                "dietary_restrictions": dietary_list
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/ingredients")
async def search_ingredients_endpoint(query: str, max_results: int = 10):
    """Direct ingredient search endpoint"""
    try:
        ingredients = await search_pinecone_ingredients(query, max_results)
        
        return {
            "ingredients": ingredients,
            "total_found": len(ingredients),
            "search_query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/products")
async def search_products_endpoint(query: str, max_results: int = 5):
    """Direct Aldi product search endpoint"""
    try:
        products = await search_pinecone_products(query, max_results)
        
        return {
            "products": products,
            "total_found": len(products),
            "search_query": query
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recipe/{recipe_id}/ingredients")
async def get_recipe_ingredients_endpoint(recipe_id: str):
    """Get detailed ingredients for a specific recipe"""
    try:
        ingredients = await get_recipe_ingredients_from_pinecone(recipe_id)
        
        return {
            "recipe_id": recipe_id,
            "ingredients": ingredients,
            "total_ingredients": len(ingredients)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/shopping-list")
async def create_shopping_list_endpoint(recipe_ids: List[str]):
    """Create shopping list from multiple recipes"""
    try:
        # Get recipe data for the provided IDs
        recipes = []
        for recipe_id in recipe_ids:
            # This would need to be implemented to get recipe data by ID
            # For now, we'll use a placeholder
            recipes.append({"recipe_id": recipe_id, "name": f"Recipe {recipe_id}"})
        
        shopping_list = await calculate_shopping_list(recipes)
        
        return {
            "shopping_list": shopping_list,
            "recipe_ids": recipe_ids,
            "total_recipes": len(recipe_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/clean-data")
async def clean_product_data():
    """Admin endpoint to identify data quality issues"""
    if not pc:
        return {"error": "Pinecone not available"}
    
    try:
        product_index = pc.Index(PRODUCT_INDEX_NAME)
        
        # Get all products to analyze
        stats = product_index.describe_index_stats()
        
        # Sample some products to check for data issues
        sample_results = product_index.query(
            vector=[0] * 384,  # Dummy vector
            top_k=100,
            include_metadata=True,
            filter={}
        )
        
        issues = {
            "alcohol_in_meat": [],
            "produce_in_wrong_category": [],
            "category_mismatches": []
        }
        
        for match in sample_results['matches']:
            metadata = match['metadata']
            name = metadata.get('name', '').lower()
            category = metadata.get('category', '').lower()
            
            # Check for alcohol in meat category
            if category == 'beef' and any(word in name for word in ['lager', 'beer', 'wine', 'ale']):
                issues["alcohol_in_meat"].append({
                    "name": metadata.get('name'),
                    "category": category,
                    "id": match['id']
                })
            
            # Check for produce in wrong categories
            if category in ['beef', 'chicken', 'meat'] and any(word in name for word in ['avocado', 'apple', 'banana', 'orange']):
                issues["produce_in_wrong_category"].append({
                    "name": metadata.get('name'),
                    "category": category,
                    "id": match['id']
                })
        
        return {
            "total_products": stats['total_vector_count'],
            "sample_size": len(sample_results['matches']),
            "issues_found": issues,
            "recommendations": [
                "Re-run the scraper with better category detection",
                "Add manual data cleaning for alcohol products",
                "Implement category validation during scraping"
            ]
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Aldi Recipe Assistant API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)