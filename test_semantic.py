import sys
import os
import json
import time
from unittest.mock import MagicMock

# Mock gi BEFORE importing anything that uses it
sys.modules['gi'] = MagicMock()
sys.modules['gi.repository'] = MagicMock()

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from assets.emoji_list import emojis
import Picker

def test():
    picker_mock = MagicMock()
    # Execute the method added to Picker class
    Picker.Picker.init_semantic_search(picker_mock)
    
    print("Is semantic ready?", picker_mock.semantic_ready)
    
    # Simulate GTK Search
    picker_mock.query = "angry monster"
    print(f"\nTesting query: '{picker_mock.query}'")
    
    import numpy as np
    query_emb = picker_mock.semantic_model.encode([picker_mock.query], normalize_embeddings=True)
    top_k = min(30, len(picker_mock.semantic_map))
    distances, indices = picker_mock.semantic_index.search(query_emb.astype('float32'), top_k)
    
    picker_mock.semantic_distances = {}
    picker_mock.current_semantic_results = set()
    for i in range(top_k):
        idx = indices[0][i]
        if idx != -1 and idx < len(picker_mock.semantic_map):
            hexcode = picker_mock.semantic_map[idx]
            picker_mock.current_semantic_results.add(hexcode)
            picker_mock.semantic_distances[hexcode] = float(distances[0][i])
    
    print("\nTop 5 Results:")
    sorted_hex = sorted(picker_mock.semantic_distances.keys(), key=lambda k: picker_mock.semantic_distances[k], reverse=True)
    for h in sorted_hex[:5]:
        e = emojis.get(h)
        if e:
            print(f"  {e['emoji']}  (Score: {picker_mock.semantic_distances[h]:.3f}) - {e['group']} - {e['subgroups']} - tags: {e['tags'][:50]}")

if __name__ == "__main__":
    test()
