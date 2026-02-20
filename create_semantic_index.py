import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import time
import sys
import os

# Add src to path so we can import assets
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from assets.emoji_list import emojis

index_file = 'src/assets/emoji_index.faiss'
map_file = 'src/assets/emoji_index_map.json'
model_name = 'all-mpnet-base-v2'

def create_and_save_index():
    print(f"Loading data from smile's emoji_list.py...")
    
    emoji_texts = []
    hexcode_map = []
    
    for hexcode, data in emojis.items():
        # Only embed the base emojis, not every skintone variant 
        # (skintones can be handled by the UI modifier)
        if not data.get('skintone'):
            tags = data.get('tags', '').replace(',', ' ')
            group = data.get('group', '').replace('-', ' ')
            subgroups = data.get('subgroups', '').replace('-', ' ')
            
            combined_text = f"{group} {subgroups} {tags}"
            emoji_texts.append(combined_text)
            hexcode_map.append(hexcode)
            
    print(f"Prepared {len(emoji_texts)} unique base emojis for embedding.")

    print(f"Loading the embedding model: '{model_name}'...")
    model = SentenceTransformer(model_name, device='cpu')
    
    print("Generating embeddings (this may take a minute)...")
    start_time = time.time()
    embeddings = model.encode(emoji_texts, normalize_embeddings=True, show_progress_bar=True)
    end_time = time.time()
    print(f"Embedding generation completed in {end_time - start_time:.2f} seconds.")

    print("Building the FAISS index...")
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"Saving the index to '{index_file}'...")
    faiss.write_index(index, index_file)
    
    print(f"Saving hexcode map to '{map_file}'...")
    with open(map_file, 'w', encoding='utf-8') as f:
        json.dump(hexcode_map, f)
        
    print("\n✅ Index and map created successfully.")

if __name__ == "__main__":
    create_and_save_index()
