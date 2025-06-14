import json
import os
import requests
import openai
from typing import List, Dict
import time
from collections import defaultdict
import re

# Load environment variables
try:
from dotenv import load_dotenv
load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Make sure to set environment variables manually.")

class AldiRecipeOptimizer:
    def __init__(self, test_mode=True):
        self.spoonacular_key = os.getenv('spoonacular_api')
        self.openai_client = openai.OpenAI(api_key=os.getenv('openai_key'))
        self.test_mode = test_mode
        
        # Load Aldi products
        with open('products.json', 'r') as f:
            self.aldi_products = json.load(f)
        
        print("🧪 Running in TEST MODE - Saving to local files")
        self.food_categories = self._categorize_products()
        
    def _categorize_products(self) -> Dict[str, List[Dict]]:
        """Categorize products by food type for better ingredient selection"""
        categories = defaultdict(list)
        
        for product in self.aldi_products:
            name_lower = product['name'].lower()
            
            # Only include whole food ingredients, avoid processed foods
            if any(processed in name_lower for processed in ['instant', 'flavour', 'ready', 'mix', 'pudding', 'sauce packet']):
                continue  # Skip processed foods
                
            if any(meat in name_lower for meat in ['chicken', 'beef', 'lamb', 'pork', 'turkey', 'salmon', 'cod', 'tuna']):
                categories['proteins'].append(product)
            elif any(veg in name_lower for veg in ['potato', 'onion', 'carrot', 'pepper', 'tomato', 'broccoli', 'spinach', 'mushroom', 'courgette']):
                categories['vegetables'].append(product)
            elif any(grain in name_lower for grain in ['rice', 'pasta', 'bread', 'flour', 'quinoa', 'barley', 'oats']):
                # Exclude rice pudding and other sweet rice products
                if not any(sweet in name_lower for sweet in ['pudding', 'dessert', 'sweet']):
                categories['grains'].append(product)
            elif any(dairy in name_lower for dairy in ['milk', 'cheese', 'yogurt', 'butter', 'cream']):
                categories['dairy'].append(product)
            elif any(legume in name_lower for legume in ['beans', 'lentils', 'chickpeas']):
                categories['legumes'].append(product)
            else:
                # Only add to 'other' if it looks like a whole food ingredient
                if not any(processed in name_lower for processed in ['crisps', 'biscuits', 'chocolate', 'sweets', 'cake']):
                categories['other'].append(product)
                
        return categories
    
    def find_cheapest_ingredients(self, max_budget: float = 15.0, num_combinations: int = 3) -> List[Dict]:
        """Use OpenAI to identify cheap ingredient combinations that work well together"""
        
        # Get cheapest products from each category
        cheap_products = {}
        for category, products in self.food_categories.items():
            if products:
                sorted_products = sorted(products, key=lambda x: x['price'])[:10]
                cheap_products[category] = sorted_products
        
        prompt = f"""
        You are a healthy meal planning expert. I have ingredients from Aldi with these prices:
        
        PROTEINS: {[f"{p['name']}: £{p['price']}" for p in cheap_products.get('proteins', [])[:8]]}
        VEGETABLES: {[f"{p['name']}: £{p['price']}" for p in cheap_products.get('vegetables', [])[:8]]}
        GRAINS: {[f"{p['name']}: £{p['price']}" for p in cheap_products.get('grains', [])[:8]]}
        DAIRY: {[f"{p['name']}: £{p['price']}" for p in cheap_products.get('dairy', [])[:8]]}
        LEGUMES: {[f"{p['name']}: £{p['price']}" for p in cheap_products.get('legumes', [])[:8]]}
        OTHER: {[f"{p['name']}: £{p['price']}" for p in cheap_products.get('other', [])[:8]]}
        
        Create {num_combinations} different ingredient combinations that:
        1. Use WHOLE FOODS ONLY - no instant noodles, ready meals, or processed foods
        2. Focus on nutritious, fresh ingredients that make complete meals
        3. Stay under £{max_budget} total budget for family of 3
        4. Convert to SPECIFIC ingredient names that recipe APIs recognize
        
        IMPORTANT CONVERSION RULES:
        - "Chicken Breast Fillets" → "chicken breast"
        - "White Potatoes 2.5kg" → "potatoes" 
        - "Carrots 1kg" → "carrots"
        - "Basmati Rice 1kg" → "basmati rice"
        - "Broccoli Crown" → "broccoli"
        - "Red Onions" → "red onions"
        - "Tinned Chopped Tomatoes" → "canned tomatoes"
        - "Kidney Beans in Water" → "kidney beans"
        
        AVOID: instant noodles, recipe mixes, ready meals, puddings, processed foods
        PREFER: fresh vegetables, meat, fish, rice, beans, lentils, basic ingredients
        
        Return JSON format:
        {{
            "combinations": [
                {{
                    "ingredients": ["chicken breast", "broccoli", "basmati rice", "onions"],
                    "estimated_cost": 8.50,
                    "meal_type": "healthy stir-fry"
                }}
            ]
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the JSON from the response
            content = response.choices[0].message.content
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            return result['combinations']
            else:
                print("No JSON found in response")
                return []
            
        except Exception as e:
            print(f"Error getting ingredient combinations: {e}")
            return []
    
    def find_recipes(self, ingredients: List[str], max_recipes: int = 10) -> List[Dict]:
        """Find recipes using the Spoonacular API"""
        try:
            # Convert ingredients to a comma-separated string
            ingredients_str = ','.join(ingredients)
            
            print(f"\n🔍 Searching for recipes with: {ingredients_str}")
            
            # Make the API request
            response = requests.get(
                'https://api.spoonacular.com/recipes/complexSearch',
                params={
            'apiKey': self.spoonacular_key,
                    'includeIngredients': ingredients_str,
                    'number': max_recipes,
                    'addRecipeInformation': True,
                    'fillIngredients': True,
                    'instructionsRequired': True
        }
            )
            
            # Check response status
            if response.status_code != 200:
                print(f"   ❌ API error {response.status_code}: {response.text}")
                return []
            
            # Check if response is empty
            if not response.text.strip():
                print(f"   ❌ Empty response from API")
                return []
            
            # Try to parse JSON
            try:
                data = response.json()
                if 'results' not in data:
                    print(f"   ❌ Unexpected API response: {data}")
                    return []
                
                recipes = data['results']
                print(f"   ✅ Found {len(recipes)} recipes")
                return recipes
                
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON decode error: {e}")
                print(f"   Raw response: {response.text[:200]}...")
                return []
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Request error searching recipes: {e}")
            return []
    
    def get_recipe_details(self, recipe_id: int) -> Dict:
        """Get detailed recipe information"""
        try:
            print(f"      📋 Getting details for recipe ID: {recipe_id}")
        
            response = requests.get(
                f'https://api.spoonacular.com/recipes/{recipe_id}/information',
                params={
            'apiKey': self.spoonacular_key,
                    'addNutrition': True
                }
            )
            
            # Check response status
            if response.status_code != 200:
                print(f"      ❌ API error {response.status_code}: {response.text}")
                return {}
            
            # Check if response is empty
            if not response.text.strip():
                print(f"      ❌ Empty response from API")
                return {}
            
            # Try to parse JSON
            try:
                recipe_data = response.json()
                print(f"      ✅ Got recipe: {recipe_data.get('title', 'Unknown title')}")
                return recipe_data
            except json.JSONDecodeError as e:
                print(f"      ❌ JSON decode error: {e}")
                print(f"      Raw response: {response.text[:200]}...")
                return {}
            
        except requests.exceptions.RequestException as e:
            print(f"      ❌ Request error getting recipe details: {e}")
            return {}
    
    def check_against_published_recipes(self, recipe_title: str) -> bool:
        """Check if recipe already exists in local storage"""
            try:
                with open('saved_recipes.json', 'r') as f:
                    saved_recipes = json.load(f)
                    return any(recipe['title'] == recipe_title for recipe in saved_recipes)
            except FileNotFoundError:
            return False
    
    def calculate_actual_cost(self, recipe: Dict, original_ingredients: List[str]) -> Dict:
        """Calculate actual cost using Aldi prices"""
        
        recipe_ingredients = []
        if 'extendedIngredients' in recipe:
            recipe_ingredients = [ing['original'] for ing in recipe['extendedIngredients']]
        elif 'ingredients' in recipe:
            recipe_ingredients = recipe['ingredients']
        
        print(f"\n📝 Processing recipe: {recipe.get('title', 'Unknown Recipe')}")
        print(f"   Ingredients: {recipe_ingredients}")
        
        prompt = f"""
        You are a cost calculation expert. Calculate the cost of this recipe using Aldi products.
        
        Recipe: {recipe.get('title', 'Unknown Recipe')}
        Recipe ingredients: {recipe_ingredients}
        Original search ingredients: {original_ingredients}
        
        Available Aldi products: {[f"{p['name']}: £{p['price']}" for p in self.aldi_products[:50]]}
        
        For each ingredient:
        1. Find the closest matching Aldi product
        2. Estimate how much would be used for {recipe.get('servings', 3)} servings
        3. Calculate the cost based on the Aldi price
        
        IMPORTANT:
        - Use exact matches when possible
        - If no exact match, use the closest available product
        - For missing ingredients, add them to missing_ingredients list
        - Round costs to 2 decimal places
        - Return ONLY the JSON object, no other text
        
        Return this exact JSON format:
        {{
            "total_cost": 12.50,
            "cost_per_serving": 4.17,
            "ingredient_costs": [
                {{
                    "recipe_ingredient": "2 chicken breasts",
                    "aldi_product": "Chicken Breast Fillets",
                    "aldi_price": 3.99,
                    "estimated_usage": 0.6,
                    "cost": 2.39
                }}
            ],
            "missing_ingredients": ["ingredient not available at Aldi"]
        }}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a cost calculation expert. Respond ONLY with valid JSON, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3  # Lower temperature for more consistent output
            )
            
            # Get the response content and clean it
            content = response.choices[0].message.content.strip()
            print(f"   Raw response: {content[:200]}...")  # Print first 200 chars for debugging
            
            # Try to find JSON in the response
            try:
                # First try direct JSON parsing
                result = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"   ⚠️  Direct JSON parsing failed: {str(e)}")
                # If that fails, try to extract JSON using regex
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️  JSON extraction failed: {str(e)}")
                        print(f"   ⚠️  Extracted content: {json_match.group()[:200]}...")
                        return {"total_cost": 999, "cost_per_serving": 999, "ingredient_costs": [], "missing_ingredients": []}
                else:
                    print(f"   ⚠️  No JSON object found in response")
            return {"total_cost": 999, "cost_per_serving": 999, "ingredient_costs": [], "missing_ingredients": []}
    
            # Validate the result structure
            required_fields = ['total_cost', 'cost_per_serving', 'ingredient_costs', 'missing_ingredients']
            if not all(field in result for field in required_fields):
                print(f"   ⚠️  Missing required fields: {[f for f in required_fields if f not in result]}")
                return {"total_cost": 999, "cost_per_serving": 999, "ingredient_costs": [], "missing_ingredients": []}
            
            # Validate numeric fields
            try:
                result['total_cost'] = float(result['total_cost'])
                result['cost_per_serving'] = float(result['cost_per_serving'])
            except (ValueError, TypeError):
                print(f"   ⚠️  Invalid numeric values in response")
                return {"total_cost": 999, "cost_per_serving": 999, "ingredient_costs": [], "missing_ingredients": []}
            
            # Print cost summary
            print(f"   💰 Total cost: £{result['total_cost']:.2f}")
            print(f"   🍽️  Cost per serving: £{result['cost_per_serving']:.2f}")
            if result['missing_ingredients']:
                print(f"   ⚠️  Missing ingredients: {result['missing_ingredients']}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error calculating cost: {str(e)}")
            return {"total_cost": 999, "cost_per_serving": 999, "ingredient_costs": [], "missing_ingredients": []}
    
    def save_recipe(self, meal_data: Dict) -> bool:
        """Save recipe to local file"""
            try:
                # Load existing recipes
                try:
                    with open('saved_recipes.json', 'r') as f:
                        saved_recipes = json.load(f)
                except FileNotFoundError:
                    saved_recipes = []
                
                # Add new recipe
                recipe_to_save = {
                    'title': meal_data['recipe']['title'],
                    'id': meal_data['recipe']['id'],
                    'total_cost': meal_data['cost_info']['total_cost'],
                    'cost_per_serving': meal_data['cost_info']['cost_per_serving'],
                    'ingredients': meal_data['original_ingredients'],
                    'ready_in_minutes': meal_data['recipe'].get('readyInMinutes'),
                    'source_url': meal_data['recipe'].get('sourceUrl'),
                    'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'full_data': meal_data
                }
                
                saved_recipes.append(recipe_to_save)
                
                # Save back to file
                with open('saved_recipes.json', 'w') as f:
                    json.dump(saved_recipes, f, indent=2)
                
                print(f"✅ Saved recipe '{meal_data['recipe']['title']}' to local file")
                return True
                
            except Exception as e:
                print(f"❌ Error saving recipe locally: {e}")
                return False
    
    def find_cheapest_meals(self, max_budget: float = 15.0, max_attempts: int = 5) -> List[Dict]:
        """Main function to find cheapest meals with retry logic"""
        
        all_viable_meals = []
        
        # Get multiple ingredient combinations
        ingredient_combinations = self.find_cheapest_ingredients(max_budget, num_combinations=max_attempts)
        
        print(f"Found {len(ingredient_combinations)} ingredient combinations to try...")
        
        for attempt, combination in enumerate(ingredient_combinations, 1):
            print(f"\n--- Attempt {attempt}: Trying {combination['ingredients']} ---")
            
            # Search for recipes with these ingredients
            recipes = self.find_recipes(combination['ingredients'])
            
            if not recipes:
                print("No recipes found, trying next combination...")
                continue
            
            print(f"Found {len(recipes)} potential recipes")
            
            # Evaluate each recipe
            for recipe in recipes[:3]:  # Check top 3 recipes per combination
                try:
                    print(f"    🧪 Processing: {recipe.get('title', 'Unknown recipe')}")
                    
                    # Check if already published
                    if self.check_against_published_recipes(recipe['title']):
                        print(f"    ⏭️  Recipe '{recipe['title']}' already published, skipping...")
                        continue
                    
                    # Add rate limiting before API calls
                    time.sleep(1)
                    
                    # Get detailed recipe info
                    detailed_recipe = self.get_recipe_details(recipe['id'])
                    if not detailed_recipe:
                        print(f"    ❌ Could not get details for {recipe.get('title', 'Unknown recipe')}")
                        continue
                    
                    # Check if recipe has ingredients
                    if not detailed_recipe.get('extendedIngredients'):
                        print(f"    ⚠️  No ingredients found in recipe details")
                        continue
                    
                    print(f"    💰 Calculating cost...")
                    # Calculate actual cost
                    cost_info = self.calculate_actual_cost(detailed_recipe, combination['ingredients'])
                    
                    # Check if cost calculation was successful
                    if cost_info['total_cost'] >= 999:
                        print(f"    ❌ Cost calculation failed")
                        continue
                    
                    # Check if within budget and viable
                    if (cost_info['total_cost'] <= max_budget and 
                        len(cost_info['missing_ingredients']) <= 2):
                        
                        meal_data = {
                            'recipe': detailed_recipe,
                            'cost_info': cost_info,
                            'original_ingredients': combination['ingredients'],
                            'search_rank': recipe.get('likes', 0),
                            'ingredient_match_ratio': len([i for i in cost_info['ingredient_costs'] if i['cost'] < 999]) / max(len(detailed_recipe.get('extendedIngredients', [])), 1)
                        }
                        
                        all_viable_meals.append(meal_data)
                        print(f"    ✅ Found viable meal: {recipe['title']} - £{cost_info['total_cost']:.2f}")
                        
                        # Save the recipe
                        self.save_recipe(meal_data)
        else:
                        print(f"    ❌ Recipe not viable: cost £{cost_info['total_cost']:.2f}, missing {len(cost_info['missing_ingredients'])} ingredients")
                    
            except Exception as e:
                    print(f"    ❌ Error processing recipe {recipe.get('title', 'Unknown')}: {e}")
                    continue
        
        # Sort by cost and return best options
        viable_meals = sorted(all_viable_meals, key=lambda x: x['cost_info']['total_cost'])
        
        return viable_meals[:10]

def main():
    optimizer = AldiRecipeOptimizer(test_mode=True)
    
    print("🔍 Finding cheapest meal combinations...")
    cheapest_meals = optimizer.find_cheapest_meals(max_budget=15.0)
    
    if not cheapest_meals:
        print("❌ No viable meals found within budget")
        return
    
    print(f"\n✅ Found {len(cheapest_meals)} viable meals!")
    print("\n" + "="*60)
    
    for i, meal in enumerate(cheapest_meals[:5], 1):
        recipe = meal['recipe']
        cost = meal['cost_info']
        
        print(f"\n{i}. {recipe['title']}")
        print(f"   💰 Total Cost: £{cost['total_cost']:.2f}")
        print(f"   👥 Per Person: £{cost['cost_per_serving']:.2f}")
        print(f"   ⏱️  Ready in: {recipe.get('readyInMinutes', 'N/A')} minutes")
        print(f"   🔗 URL: {recipe.get('sourceUrl', 'N/A')}")
        
        if cost['missing_ingredients']:
            print(f"   ⚠️  Missing: {', '.join(cost['missing_ingredients'])}")

if __name__ == "__main__":
    main()