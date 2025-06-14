#!/usr/bin/env python3
"""
AI-Powered Ingredient Matching System
Uses embedding similarity, classification models, and active learning
to intelligently match recipe ingredients to Aldi products
"""

import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from dataclasses import dataclass
from collections import defaultdict
import re

@dataclass
class MatchingTrainingData:
    """Training data for the AI matching system"""
    ingredient: str
    product_name: str
    is_correct_match: bool
    confidence: float
    user_feedback: Optional[str] = None

class AIIngredientMatcher:
    """AI-powered ingredient to product matching system"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("🤖 Initializing AI-Powered Ingredient Matcher...")
        
        # Load embedding model for semantic similarity
        self.embedding_model = SentenceTransformer(model_name)
        
        # TF-IDF for additional text features
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Classification model for match prediction
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Training data storage
        self.training_data: List[MatchingTrainingData] = []
        
        # Product embeddings cache
        self.product_embeddings = {}
        self.product_data = {}
        
        # Learned patterns
        self.category_patterns = defaultdict(list)
        self.brand_patterns = defaultdict(list)
        
        # Model trained flag
        self.is_trained = False
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        # Remove measurements and quantities
        text = re.sub(r'\d+(\.\d+)?\s*(g|kg|ml|l|oz|lb|tsp|tbsp|cup|slice|piece)', '', text, flags=re.IGNORECASE)
        
        # Remove descriptive words but keep important ones
        descriptive_words = ['fresh', 'frozen', 'dried', 'chopped', 'diced', 'sliced']
        important_words = ['free-range', 'organic', 'smoked', 'unsalted', 'wholemeal']
        
        words = text.lower().split()
        filtered_words = []
        
        for word in words:
            # Keep important descriptive words
            if any(imp in word for imp in important_words):
                filtered_words.append(word)
            # Remove basic descriptive words
            elif word not in descriptive_words:
                filtered_words.append(word)
        
        return ' '.join(filtered_words).strip()
    
    def load_aldi_products(self, aldi_file: str):
        """Load and embed Aldi products"""
        print(f"📦 Loading Aldi products from {aldi_file}...")
        
        with open(aldi_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        print("🧠 Generating product embeddings...")
        for i, product in enumerate(products):
            if i % 100 == 0:
                print(f"   Processed {i}/{len(products)} products...")
            
            product_id = str(i)
            product_name = product['name']
            
            # Store product data
            self.product_data[product_id] = {
                'name': product_name,
                'price': product['price'],
                'category': product.get('category', 'unknown'),
                'url': product.get('url', ''),
                'original': product
            }
            
            # Generate embeddings
            processed_name = self.preprocess_text(product_name)
            embedding = self.embedding_model.encode(processed_name)
            self.product_embeddings[product_id] = embedding
        
        print(f"✅ Loaded {len(products)} products with embeddings")
    
    def calculate_semantic_similarity(self, ingredient: str, product_id: str) -> float:
        """Calculate semantic similarity between ingredient and product"""
        processed_ingredient = self.preprocess_text(ingredient)
        ingredient_embedding = self.embedding_model.encode(processed_ingredient)
        product_embedding = self.product_embeddings[product_id]
        
        similarity = cosine_similarity(
            ingredient_embedding.reshape(1, -1),
            product_embedding.reshape(1, -1)
        )[0][0]
        
        return float(similarity)
    
    def extract_features(self, ingredient: str, product_id: str) -> np.ndarray:
        """Extract features for the classification model"""
        product = self.product_data[product_id]
        
        features = []
        
        # 1. Semantic similarity
        similarity = self.calculate_semantic_similarity(ingredient, product_id)
        features.append(similarity)
        
        # 2. Exact word matches
        ingredient_words = set(self.preprocess_text(ingredient).lower().split())
        product_words = set(self.preprocess_text(product['name']).lower().split())
        exact_matches = len(ingredient_words.intersection(product_words))
        features.append(exact_matches)
        
        # 3. Category relevance (learned patterns)
        category_relevance = self.calculate_category_relevance(ingredient, product['category'])
        features.append(category_relevance)
        
        # 4. Price reasonableness (very expensive items less likely for simple ingredients)
        price_score = min(1.0, 5.0 / max(product['price'], 0.1))  # Normalize around £5
        features.append(price_score)
        
        # 5. Name length similarity
        length_diff = abs(len(ingredient) - len(product['name']))
        length_score = max(0, 1 - length_diff / 50)  # Normalize
        features.append(length_score)
        
        # 6. Special keyword bonuses
        special_keywords = ['free-range', 'organic', 'smoked', 'unsalted', 'wholemeal']
        keyword_bonus = sum(1 for kw in special_keywords if kw in ingredient.lower() and kw in product['name'].lower())
        features.append(keyword_bonus)
        
        return np.array(features)
    
    def calculate_category_relevance(self, ingredient: str, category: str) -> float:
        """Calculate how relevant a category is for an ingredient"""
        # Basic category mapping (can be learned over time)
        category_mappings = {
            'eggs': ['dairy', 'eggs'],
            'milk': ['dairy'],
            'cheese': ['dairy'],
            'butter': ['dairy'],
            'chicken': ['meat', 'poultry'],
            'beef': ['meat'],
            'pork': ['meat'],
            'bacon': ['meat'],
            'sausage': ['meat'],
            'bread': ['bread', 'bakery'],
            'flour': ['baking'],
            'beans': ['pantry', 'canned'],
            'rice': ['pantry'],
            'pasta': ['pantry'],
            'onion': ['vegetables', 'fresh'],
            'tomato': ['vegetables', 'fresh'],
            'oil': ['pantry', 'cooking'],
        }
        
        ingredient_lower = ingredient.lower()
        expected_categories = []
        
        for ing, cats in category_mappings.items():
            if ing in ingredient_lower:
                expected_categories.extend(cats)
        
        if not expected_categories:
            return 0.5  # Neutral if unknown
        
        return 1.0 if category.lower() in expected_categories else 0.2
    
    def find_candidate_products(self, ingredient: str, top_k: int = 20) -> List[str]:
        """Find top candidate products using semantic similarity"""
        processed_ingredient = self.preprocess_text(ingredient)
        ingredient_embedding = self.embedding_model.encode(processed_ingredient)
        
        similarities = []
        for product_id, product_embedding in self.product_embeddings.items():
            similarity = cosine_similarity(
                ingredient_embedding.reshape(1, -1),
                product_embedding.reshape(1, -1)
            )[0][0]
            similarities.append((product_id, similarity))
        
        # Sort by similarity and return top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in similarities[:top_k]]
    
    def predict_matches(self, ingredient: str, candidate_products: List[str]) -> List[Tuple[str, float]]:
        """Predict match probabilities for candidate products"""
        if not self.is_trained:
            # Fall back to semantic similarity if not trained
            results = []
            for product_id in candidate_products:
                similarity = self.calculate_semantic_similarity(ingredient, product_id)
                results.append((product_id, similarity))
            return sorted(results, key=lambda x: x[1], reverse=True)
        
        # Use trained classifier
        features_matrix = []
        for product_id in candidate_products:
            features = self.extract_features(ingredient, product_id)
            features_matrix.append(features)
        
        features_matrix = np.array(features_matrix)
        probabilities = self.classifier.predict_proba(features_matrix)[:, 1]  # Probability of positive match
        
        results = [(pid, prob) for pid, prob in zip(candidate_products, probabilities)]
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def find_best_matches(self, ingredient: str, top_k: int = 5) -> List[Dict]:
        """Find best matching products for an ingredient"""
        # Get candidate products
        candidates = self.find_candidate_products(ingredient, top_k=50)
        
        # Predict matches
        predictions = self.predict_matches(ingredient, candidates)
        
        # Format results
        results = []
        for product_id, confidence in predictions[:top_k]:
            product = self.product_data[product_id]
            result = {
                'product_id': product_id,
                'name': product['name'],
                'price': product['price'],
                'category': product['category'],
                'confidence': float(confidence),
                'semantic_similarity': self.calculate_semantic_similarity(ingredient, product_id),
                'original_product': product['original']
            }
            results.append(result)
        
        return results
    
    def add_training_data(self, ingredient: str, product_id: str, is_correct: bool, confidence: float = 1.0):
        """Add training data for the model"""
        training_point = MatchingTrainingData(
            ingredient=ingredient,
            product_name=self.product_data[product_id]['name'],
            is_correct_match=is_correct,
            confidence=confidence
        )
        self.training_data.append(training_point)
        print(f"📚 Added training data: {ingredient} -> {training_point.product_name} ({'✅' if is_correct else '❌'})")
    
    def train_classifier(self):
        """Train the classification model on collected data"""
        if len(self.training_data) < 10:
            print("⚠️ Need at least 10 training examples to train classifier")
            return
        
        print(f"🎓 Training classifier on {len(self.training_data)} examples...")
        
        X = []
        y = []
        
        for data_point in self.training_data:
            # Find product by name (simplified - in production, store product_id)
            product_id = None
            for pid, product in self.product_data.items():
                if product['name'] == data_point.product_name:
                    product_id = pid
                    break
            
            if product_id:
                features = self.extract_features(data_point.ingredient, product_id)
                X.append(features)
                y.append(1 if data_point.is_correct_match else 0)
        
        if len(X) > 0:
            X = np.array(X)
            y = np.array(y)
            
            self.classifier.fit(X, y)
            self.is_trained = True
            
            # Calculate training accuracy
            predictions = self.classifier.predict(X)
            accuracy = np.mean(predictions == y)
            print(f"✅ Model trained! Training accuracy: {accuracy:.2%}")
        else:
            print("❌ No valid training data found")
    
    def save_model(self, filepath: str):
        """Save the trained model and data"""
        model_data = {
            'classifier': self.classifier,
            'training_data': self.training_data,
            'is_trained': self.is_trained,
            'category_patterns': dict(self.category_patterns),
            'brand_patterns': dict(self.brand_patterns)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"💾 Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            print(f"⚠️ Model file {filepath} not found")
            return
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.training_data = model_data['training_data']
        self.is_trained = model_data['is_trained']
        self.category_patterns = defaultdict(list, model_data['category_patterns'])
        self.brand_patterns = defaultdict(list, model_data['brand_patterns'])
        
        print(f"📂 Model loaded from {filepath}")
        print(f"   Training examples: {len(self.training_data)}")
        print(f"   Is trained: {self.is_trained}")

def create_initial_training_data(matcher: AIIngredientMatcher) -> List[Tuple[str, str, bool]]:
    """Create initial training data for common ingredient-product pairs"""
    
    # Format: (ingredient, expected_product_name_contains, is_correct)
    training_examples = [
        # Positive examples (correct matches)
        ("chicken sausages", "chicken sausage", True),
        ("smoked back bacon", "smoked bacon", True),
        ("free-range eggs", "free range eggs", True),
        ("baked beans", "baked beans", True),
        ("wholemeal bread", "wholemeal bread", True),
        ("unsalted butter", "unsalted butter", True),
        ("olive oil", "olive oil", True),
        ("onions", "onions", True),
        ("garlic", "garlic", True),
        ("tomatoes", "tomatoes", True),
        
        # Negative examples (incorrect matches)
        ("bacon", "salmon", False),
        ("eggs", "sandwich", False),
        ("butter", "butter beans", False),
        ("chicken", "beef", False),
        ("bread", "beans", False),
        ("milk", "chocolate", False),
        ("cheese", "sauce", False),
        ("oil", "vinegar", False),
    ]
    
    training_data = []
    
    for ingredient, product_contains, is_correct in training_examples:
        # Find matching products
        matches = matcher.find_best_matches(ingredient, top_k=10)
        
        for match in matches:
            if product_contains.lower() in match['name'].lower():
                product_id = match['product_id']
                matcher.add_training_data(ingredient, product_id, is_correct)
                training_data.append((ingredient, match['name'], is_correct))
                break
    
    return training_data

def test_ai_matching():
    """Test the AI matching system"""
    print("🧪 Testing AI Ingredient Matching System")
    print("=" * 50)
    
    # Initialize matcher
    matcher = AIIngredientMatcher()
    
    # Load Aldi products (you'll need to update this path)
    aldi_file = "aldi_products.json"
    if not os.path.exists(aldi_file):
        print(f"❌ {aldi_file} not found! Please ensure the file exists.")
        return
    
    matcher.load_aldi_products(aldi_file)
    
    # Create initial training data
    print("\n📚 Creating initial training data...")
    training_data = create_initial_training_data(matcher)
    print(f"Created {len(training_data)} training examples")
    
    # Train the model
    matcher.train_classifier()
    
    # Test on breakfast recipe
    test_ingredients = [
        "4 chicken sausages",
        "4 slices smoked back bacon",
        "2 free-range eggs", 
        "400g tin baked beans",
        "2 slices wholemeal bread",
        "2 tsp unsalted butter"
    ]
    
    print("\n🍳 Testing on breakfast recipe ingredients:")
    print("-" * 40)
    
    total_cost = 0
    for ingredient in test_ingredients:
        print(f"\n🔍 {ingredient}")
        matches = matcher.find_best_matches(ingredient, top_k=3)
        
        if matches:
            best_match = matches[0]
            print(f"   ✅ Best: {best_match['name']}")
            print(f"      💰 £{best_match['price']:.2f}")
            print(f"      🤖 AI Confidence: {best_match['confidence']:.1%}")
            print(f"      🧠 Semantic Similarity: {best_match['semantic_similarity']:.1%}")
            print(f"      📂 Category: {best_match['category']}")
            
            total_cost += best_match['price'] * 0.8  # Estimate portion used
            
            # Show alternatives
            if len(matches) > 1:
                print(f"      📋 Alternatives:")
                for alt in matches[1:3]:
                    print(f"         • {alt['name']} (£{alt['price']:.2f}, {alt['confidence']:.1%})")
        else:
            print("   ❌ No matches found")
    
    print(f"\n💰 Estimated total cost: £{total_cost:.2f}")
    
    # Save the trained model
    matcher.save_model("ai_ingredient_matcher.pkl")
    
    return matcher

if __name__ == "__main__":
    test_ai_matching()