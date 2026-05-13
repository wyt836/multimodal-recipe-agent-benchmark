"""OpenCLIP encoders + ChromaDB indices for recipe text and images.

Two collections under one persistent ChromaDB:
    - recipes_text:  document-per-recipe text features (name, cuisine, ingredients,
                     caption, methods, dietary tags, brief steps)
    - recipes_image: CLIP image features, one document per recipe image

Both collections live in the same CLIP joint space so a text query can be
issued against either. The text collection captures conceptual content;
the image collection captures visual properties (colors, presentation, layout).
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import chromadb
import numpy as np
import open_clip
import torch
from PIL import Image

from config import (
    CHROMA_DIR, CLIP_MODEL, CLIP_PRETRAINED,
    IMAGE_COLLECTION, IMAGES_DIR, PROJECT_ROOT,
    RECIPES_JSON, TEXT_COLLECTION,
)


# ---------------------------------------------------------------- CLIP loading
@lru_cache(maxsize=1)
def _clip():
    """Load OpenCLIP model + preprocess + tokenizer once per process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device,
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    return model, preprocess, tokenizer, device


def encode_text(texts: List[str]) -> np.ndarray:
    """Encode a list of strings → L2-normalised CLIP embeddings (N, D)."""
    model, _, tokenizer, device = _clip()
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


def encode_images(paths: List[Path]) -> np.ndarray:
    """Encode a list of image files → L2-normalised CLIP embeddings (N, D)."""
    model, preprocess, _, device = _clip()
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(device)
    with torch.no_grad():
        feats = model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy()


# ---------------------------------------------------------------- Recipe corpus
def load_recipes() -> list[dict]:
    with open(RECIPES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _recipe_to_text(r: dict) -> str:
    """Compact textual representation of a recipe for the text index."""
    ingredients = ", ".join(i["name"] for i in r["ingredients"])
    tags = ", ".join(r.get("dietary_tags", []))
    methods = ", ".join(r.get("cooking_methods", []))
    return (
        f"{r['name']} ({r['cuisine']} cuisine). "
        f"Difficulty {r['difficulty']}, serves {r['servings']}, "
        f"prep {r['prep_time_min']} min, cook {r['cooking_time_min']} min, "
        f"total {r['total_time_min']} min, main protein {r['main_protein']}, "
        f"spicy level {r['spicy_level']}. "
        f"Ingredients: {ingredients}. "
        f"Methods: {methods}. "
        f"Dietary: {tags}. "
        f"{r['image_caption']}"
    )


# ---------------------------------------------------------------- Index build
def _client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_indices(force: bool = False) -> None:
    """Build (or rebuild) both ChromaDB collections from recipes.json."""
    client = _client()
    existing = {c.name for c in client.list_collections()}

    if not force and {TEXT_COLLECTION, IMAGE_COLLECTION}.issubset(existing):
        return  # already built; nothing to do

    for name in (TEXT_COLLECTION, IMAGE_COLLECTION):
        if name in existing:
            client.delete_collection(name)

    recipes = load_recipes()
    ids = [r["id"] for r in recipes]

    # Text index --------------------------------------------------
    print(f"[retrieval] encoding {len(recipes)} recipe texts...")
    text_docs = [_recipe_to_text(r) for r in recipes]
    text_emb = encode_text(text_docs)
    text_col = client.create_collection(TEXT_COLLECTION, metadata={"hnsw:space": "cosine"})
    text_col.add(
        ids=ids,
        embeddings=text_emb.tolist(),
        documents=text_docs,
        metadatas=[{"recipe_id": r["id"], "name": r["name"], "cuisine": r["cuisine"]} for r in recipes],
    )

    # Image index -------------------------------------------------
    print(f"[retrieval] encoding {len(recipes)} recipe images...")
    image_paths = [PROJECT_ROOT / r["image_path"] for r in recipes]
    image_emb = encode_images(image_paths)
    image_col = client.create_collection(IMAGE_COLLECTION, metadata={"hnsw:space": "cosine"})
    image_col.add(
        ids=ids,
        embeddings=image_emb.tolist(),
        documents=[r["image_caption"] for r in recipes],
        metadatas=[{"recipe_id": r["id"], "name": r["name"], "cuisine": r["cuisine"]} for r in recipes],
    )

    print(f"[retrieval] built {TEXT_COLLECTION} and {IMAGE_COLLECTION} at {CHROMA_DIR}")


# ---------------------------------------------------------------- Queries
def _query(collection_name: str, query_text: str, k: int) -> List[Tuple[str, float]]:
    """Encode query_text and return top-k (recipe_id, similarity) hits."""
    client = _client()
    col = client.get_collection(collection_name)
    q_emb = encode_text([query_text])[0].tolist()
    res = col.query(query_embeddings=[q_emb], n_results=k)
    ids = res["ids"][0]
    # Chroma returns distances; convert cosine distance → similarity for readability.
    dists = res["distances"][0]
    return [(rid, float(1.0 - d)) for rid, d in zip(ids, dists)]


def search_text(query: str, k: int = 3) -> List[Tuple[str, float]]:
    return _query(TEXT_COLLECTION, query, k)


def search_image(query: str, k: int = 3) -> List[Tuple[str, float]]:
    return _query(IMAGE_COLLECTION, query, k)


if __name__ == "__main__":
    # Convenience: `python src/retrieval.py` to build indices once.
    build_indices(force=True)
    print(search_text("spicy tofu dish", k=3))
    print(search_image("soup with white cubes floating in broth", k=3))
