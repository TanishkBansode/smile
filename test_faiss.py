import json
import faiss
from sentence_transformers import SentenceTransformer

def test():
    map_path = 'src/assets/emoji_index_map.json'
    index_path = 'src/assets/emoji_index.faiss'
    
    with open(map_path, 'r') as f:
        semantic_map = json.load(f)
    print(f"Loaded semantic map with {len(semantic_map)} items.")
    
    semantic_index = faiss.read_index(index_path)
    semantic_model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    print("Models loaded successfully")
    
    query = "angry monster"
    print(f"\nQuerying for: '{query}'")
    
    import numpy as np
    query_emb = semantic_model.encode([query], normalize_embeddings=True)
    top_k = 5
    distances, indices = semantic_index.search(query_emb.astype('float32'), top_k)
    
    sys.path.append('src')
    from assets.emoji_list import emojis
    
    print("\nResults:")
    for i in range(top_k):
        idx = indices[0][i]
        if idx != -1 and idx < len(semantic_map):
            hexcode = semantic_map[idx]
            dist = float(distances[0][i])
            e = emojis[hexcode]
            print(f"[{i+1}] {e['emoji']} (Score: {dist:.3f}) - {e['group']} - {e['subgroups']}")
            print(f"    Tags: {e['tags'][:60]}...")

import sys
test()
