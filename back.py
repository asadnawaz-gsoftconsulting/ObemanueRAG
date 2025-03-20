import os
from pymongo import MongoClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from bson import ObjectId

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

# Initialize MongoDB connection
def initialize_db():
    client = MongoClient("mongodb+srv://asadnawaz:asad4555@asad.mhmh7.mongodb.net/?retryWrites=true&w=majority&appName=asad")
    db = client["RAG"]
    collection = db["manue"]
    
    # Create an index on the embedding field if it doesn't exist
    if "embedding_1" not in collection.index_information():
        collection.create_index("embedding")
    
    return collection

# Fetch all menu items from MongoDB
def fetch_all_menu_items(collection):
    """
    Retrieve all menu items from the collection.
    """
    try:
        menu_items = list(collection.find(
            {"embedding": {"$exists": False}},  # Only fetch items without embeddings
            {
                "_id": 1,
                "name.en": 1,
                "description.en": 1,
                "itemDetails": 1,
                "tags.name.en": 1,
                "ingredients.name.en": 1,
                "customizations.name.en": 1,
                "customizations.prices.description.en": 1,
                "customizations.prices.variants.price": 1,
                "categories.name.en": 1,
                "prices.description.en": 1,
                "prices.variants.price": 1
            }
        ))

        total_items = collection.count_documents({})
        items_with_embeddings = collection.count_documents({"embedding": {"$exists": True}})
        
        print(f"Statistics:")
        print(f"   - Total items in collection: {total_items}")
        print(f"   - Items with embeddings: {items_with_embeddings}")
        print(f"   - Items without embeddings: {len(menu_items)}")

        return menu_items

    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

def generate_item_embedding(item, embeddings_model):
    """
    Generate embeddings for a menu item by combining relevant text fields.
    """
    try:
        # Extract and clean the description
        description_raw = item.get('description', {}).get('en', '')
        if description_raw:
            try:
                description_data = json.loads(description_raw)
                description_text = ' '.join(
                    child.get('text', '')
                    for block in description_data
                    for child in block.get('children', [])
                )
            except json.JSONDecodeError:
                description_text = description_raw
        else:
            description_text = ''

        # Extract customization information
        customizations = []
        for customization in item.get('customizations', []):
            custom_name = customization.get('name', {}).get('en', '')
            for price in customization.get('prices', []):
                custom_desc = price.get('description', {}).get('en', '')
                custom_price = price.get('variants', [{}])[0].get('price', '')
                if custom_desc and custom_price:
                    customizations.append(f"{custom_name} - {custom_desc}: ${custom_price}")

        # Combine relevant fields into a single text
        combined_text = f"""
        Name: {item.get('name', {}).get('en', '')}
        Description: {description_text}
        Category: {item.get('categories', {}).get('name', {}).get('en', '')}
        Details: {item.get('itemDetails', '')}
        Price: {item.get('prices', [{}])[0].get('variants', [{}])[0].get('price', '')}
        Customizations: {' | '.join(customizations)}
        """.strip()

        # Generate embedding for the combined text
        embedding = embeddings_model.embed_documents([combined_text])[0]
        
        print(f"Generated embedding for item: {item.get('name', {}).get('en', '')}")
        return embedding

    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def generate_all_embeddings(collection, embeddings_model):
    """
    Generate and store embeddings for all menu items that don't have them.
    """
    menu_items = fetch_all_menu_items(collection)
    
    if not menu_items:
        print("✨ All items already have embeddings!")
        return
    
    total_items = len(menu_items)
    processed = 0
    
    for item in menu_items:
        # Double check if item already has an embedding
        existing_item = collection.find_one({"_id": item["_id"], "embedding": {"$exists": True}})
        if existing_item:
            print(f"Skipping item {item.get('name', {}).get('en', '')}: embedding already exists")
            continue
            
        embedding = generate_item_embedding(item, embeddings_model)
        if embedding:
            # Use update_one with upsert=False to prevent duplicates
            collection.update_one(
                {"_id": item["_id"]},
                {
                    "$set": {
                        "embedding": embedding,
                        "embedding_created_at": datetime.datetime.utcnow()
                    }
                },
                upsert=False
            )
            processed += 1
            print(f"Progress: {processed}/{total_items} items processed")
    
    print(f"Embedding generation complete. {processed} new items processed.")

def search_similar_items(collection, query, embeddings_model, limit=10, score_threshold=0.5):
    """
    Search for similar menu items using vector similarity search with venue and menu filtering.
    """
    try:
        query_embedding = embeddings_model.embed_query(query)
        
        similar_items = collection.aggregate([
            {
                "$vectorSearch": {
                    "index": "default",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 500,
                    "limit": 10
                }
            },
            {
                "$project": {
                    "name": "$name.en",
                    "description": "$description.en",
                    "category": "$categories.name.en",
                    "itemDetails": 1,
                    "price": {
                        "$arrayElemAt": ["$prices.variants.price", 0]
                    },
                    "customizations": {
                        "$map": {
                            "input": "$customizations",
                            "as": "c",
                            "in": {
                                "name": "$$c.name.en",
                                "prices": {
                                    "$map": {
                                        "input": "$$c.prices",
                                        "as": "p",
                                        "in": {
                                            "description": "$$p.description.en",
                                            "price": { "$arrayElemAt": ["$$p.variants.price", 0] }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "tags": "$tags.name.en",
                    "ingredients": "$ingredients.name.en",
                    "venue_id": "$venue._id",
                    "menu_id": "$menus._id",
                    "score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$match": {
                    "score": {"$gte": score_threshold}
                }
            },
            {
                "$limit": limit
            }
        ])
        
        results = list(similar_items)
        processed_results = []
        query_terms = set(query.lower().split())
        
        for item in results:
            name_terms = set(str(item.get('name', '')).lower().split())
            category_terms = set(str(item.get('category', '')).lower().split())
            tags = set(str(tag).lower() for tag in (item.get('tags') or []))
            
            # Add customization terms to matching
            customization_terms = set()
            for custom in item.get('customizations', []):
                customization_terms.update(str(custom.get('name', '')).lower().split())
                for price in custom.get('prices', []):
                    customization_terms.update(str(price.get('description', '')).lower().split())
            
            term_matches = len(query_terms & (name_terms | category_terms | tags | customization_terms))
            relevancy_boost = term_matches * 0.1
            
            final_score = item['score'] + relevancy_boost
            
            processed_results.append({
                'name': item['name'],
                'category': item['category'],
                'price': item['price'],
                'description': item.get('description', ''),
                'customizations': item.get('customizations', []),
                'venue_id': item.get('venue_id'),
                'menu_id': item.get('menu_id'),
                'score': final_score
            })
        
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        return processed_results[:limit]
        
    except Exception as e:
        print(f"Error performing similarity search: {e}")
        return []

from openai import OpenAI

def generate_natural_response(query, similar_items):
    """
    Generate a natural language response using GPT-4 based on the user's query and search results.
    """
    try:
        client = OpenAI(api_key=api_key)
        
        items_text = ""
        for idx, item in enumerate(similar_items, 1):
            items_text += f"\nItem {idx}:\n"
            items_text += f"Name: {item['name']}\n"
            items_text += f"Category: {item['category']}\n"
            items_text += f"Base Price: ${item.get('price', 'N/A')}\n"
            
            # Add customization details
            if item.get('customizations'):
                items_text += "Customizations:\n"
                for custom in item['customizations']:
                    items_text += f"- {custom['name']}:\n"
                    for price in custom.get('prices', []):
                        items_text += f"  * {price['description']}: ${price['price']}\n"
            
            if item.get('description'):
                items_text += f"Description: {item['description']}\n"

        system_message = """You are a concise menu assistant. Provide brief, direct responses:
        - For price queries: Return item name, base price, and relevant customization prices
        - For availability queries: List matching items with prices and customizations
        - Keep responses under 3 sentences
        - No greetings or pleasantries needed
        If no items found, just say "Item not available." """

        user_content = f"""User Query: "{query}"

        Available Menu Items:
        {items_text if similar_items else 'No matching items found.'}

        Please provide a helpful response to the user's query based on these menu items."""

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error generating natural response: {e}")
        return "I apologize, but I'm having trouble generating a response at the moment."
    

# Create Flask app
app = Flask(__name__)
CORS(app)

# Add this class for JSON encoding
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

# Update the search endpoint
@app.route('/search', methods=['POST'])
def search_endpoint():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Initialize MongoDB and embedding model
        collection = initialize_db()
        embeddings_model = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-ada-002"
        )
        
        # Search for similar items
        similar_items = search_similar_items(
            collection,
            query,
            embeddings_model
        )
        
        # Generate natural language response
        response_text = generate_natural_response(query, similar_items)
        return jsonify({
            'response': response_text
        })
        # Use custom JSON encoder
        return app.response_class(
            response=json.dumps({
                'items': similar_items,
                'response': response_text
            }, cls=JSONEncoder),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add this to configure Flask's JSON encoder
app.json_encoder = JSONEncoder

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# Modify the main execution block
if __name__ == "__main__":
    # Keep the existing code for command line interface
    if os.getenv('CLI_MODE', '').lower() == 'true':
        # Initialize MongoDB and embedding model
        collection = initialize_db()
        embeddings_model = OpenAIEmbeddings(
            api_key=api_key,
            model="text-embedding-ada-002"
        )

        # Generate embeddings for all items
        print("Starting embedding generation process...")
        generate_all_embeddings(collection, embeddings_model)

        # Example of how to use the search functionality
        while True:
            query = input("\nEnter your search query (or 'quit' to exit): ")
            if query.lower() == 'quit':
                break
                
            print("\nSearching for similar items...")
            similar_items = search_similar_items(
                collection,
                query,
                embeddings_model,
            )
            response = generate_natural_response(query, similar_items)
            print("\nResponse:")
            print(response)
    else:
        # Run Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)