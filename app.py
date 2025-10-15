import io
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image


# -----------------------------
# Inline data (dummy catalog)
# -----------------------------

def _make_product(i: int) -> Dict:
	return {
		"id": i,
		"title": f"Sample Product {i}",
		"category": ["Shoes", "Bags", "Watches", "Shirts", "Pants"][i % 5],
		"price": 10 + (i % 40) * 2,
		"brand": ["Acme", "Globex", "Umbrella", "Initech", "Soylent"][i % 5],
		"thumbnail": f"https://picsum.photos/seed/prod{i}/400/400",
		"images": [f"https://picsum.photos/seed/prod{i}-{j}/400/400" for j in range(1, 3)],
	}


@st.cache_data(show_spinner=False)
def get_products() -> List[Dict]:
	# Smaller set for faster demo
	return [_make_product(i) for i in range(1, 41)]


# -----------------------------
# Inline simple image embedding
# -----------------------------
@st.cache_data(show_spinner=False)
def _image_to_feature(image_bytes: bytes) -> np.ndarray:
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
	arr = np.asarray(image)
	hist_r, _ = np.histogram(arr[:, :, 0], bins=32, range=(0, 255), density=True)
	hist_g, _ = np.histogram(arr[:, :, 1], bins=32, range=(0, 255), density=True)
	hist_b, _ = np.histogram(arr[:, :, 2], bins=32, range=(0, 255), density=True)
	feat = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
	norm = float(np.linalg.norm(feat))
	if norm > 0:
		feat /= norm
	return feat


def get_image_embedding(image: Image.Image) -> np.ndarray:
	buf = io.BytesIO()
	image.save(buf, format="PNG")
	return _image_to_feature(buf.getvalue())


@st.cache_data(show_spinner=False)
def fetch_image_feature_from_url(url: str) -> np.ndarray:
	resp = requests.get(url, timeout=4)
	resp.raise_for_status()
	return _image_to_feature(resp.content)


# -----------------------------
# App UI and logic
# -----------------------------

def set_page_config() -> None:
	st.set_page_config(
		page_title="Visual Product Matcher (Dummy)",
		page_icon="🖼️",
		layout="wide",
		initial_sidebar_state="expanded",
	)


def ensure_cache_dirs() -> None:
	os.makedirs("cache", exist_ok=True)


def load_image_from_upload(uploaded_file) -> Image.Image:
	image_bytes = uploaded_file.read()
	return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def load_image_from_url(url: str) -> Image.Image:
	response = requests.get(url, timeout=10)
	response.raise_for_status()
	return Image.open(io.BytesIO(response.content)).convert("RGB")


def compute_similarities(
	query_embedding: np.ndarray,	product_embeddings: np.ndarray,
) -> np.ndarray:
	norms = np.linalg.norm(product_embeddings, axis=1, keepdims=True) * np.linalg.norm(query_embedding)
	norms[norms == 0] = 1e-8
	sims = (product_embeddings @ query_embedding) / norms.squeeze(1)
	return sims


def render_sidebar_controls() -> Tuple[int, float]:
	st.sidebar.header("Search Controls")
	top_k = st.sidebar.slider("Top K results", min_value=5, max_value=40, value=16, step=1)
	min_score = st.sidebar.slider("Minimum similarity", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
	return top_k, float(min_score)


def render_header() -> None:
	st.title("Visual Product Matcher (Dummy)")
	st.caption("Demo with lightweight image features and placeholder products")


def render_uploader() -> Tuple[Image.Image | None, str | None]:
	col1, col2 = st.columns([1, 1])
	with col1:
		uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
		uploaded_img = None
		if uploaded is not None:
			try:
				uploaded_img = load_image_from_upload(uploaded)
			except Exception as e:
				st.error(f"Failed to read uploaded image: {e}")
	with col2:
		url = st.text_input("...or paste an image URL")
		url_img = None
		if url:
			try:
				url_img = load_image_from_url(url)
			except Exception as e:
				st.error(f"Failed to fetch image from URL: {e}")
	return (uploaded_img or url_img), ("upload" if uploaded is not None else ("url" if url else None))


def format_products_dataframe(products: List[Dict]) -> pd.DataFrame:
	rows = []
	for p in products:
		image_url = p.get("thumbnail") or (p.get("images") or [None])[0]
		rows.append({
			"id": p.get("id"),
			"title": p.get("title"),
			"category": p.get("category"),
			"price": p.get("price"),
			"brand": p.get("brand"),
			"image": image_url,
		})
	return pd.DataFrame(rows)


def main() -> None:
	set_page_config()
	ensure_cache_dirs()

	render_header()
	top_k, min_score = render_sidebar_controls()

	with st.spinner("Loading product catalog..."):
		products = get_products()
		if not products:
			st.error("No products available. Please try again later.")
			return
		products_df = format_products_dataframe(products)

	# Compute product embeddings with a visible progress bar
	with st.spinner("Preparing product embeddings..."):
		image_urls = products_df["image"].fillna("").astype(str).tolist()
		progress = st.progress(0)
		emb_list: list[np.ndarray] = []
		for idx, url in enumerate(image_urls):
			try:
				emb = fetch_image_feature_from_url(url)
			except Exception:
				emb = np.zeros(96, dtype=np.float32)
			emb_list.append(emb)
			if (idx + 1) % 2 == 0 or (idx + 1) == len(image_urls):
				progress.progress(int(((idx + 1) / max(1, len(image_urls))) * 100))
		product_embeddings = np.vstack(emb_list) if emb_list else np.zeros((0, 96), dtype=np.float32)

	query_image, source = render_uploader()
	if query_image is None:
		st.info("Upload an image file or paste an image URL to start searching.")
		st.image([p for p in products_df["image"].head(6) if p], caption=["Sample product"] * min(6, len(products_df)), width=200)
		return

	st.subheader("Query Image")
	st.image(query_image, width=320)

	with st.spinner("Computing query feature..."):
		try:
			query_emb = get_image_embedding(query_image)
		except Exception as e:
			st.error(f"Failed to compute feature: {e}")
			return

	sims = compute_similarities(query_emb, product_embeddings)
	products_df = products_df.copy()
	products_df["similarity"] = sims

	filtered = products_df[products_df["similarity"] >= min_score]
	filtered = filtered.sort_values("similarity", ascending=False).head(top_k)

	st.subheader("Similar Products")
	if filtered.empty:
		st.warning("No results match the current filters. Try lowering the minimum similarity.")
		return

	grid_cols = st.columns(4)
	for idx, (_, row) in enumerate(filtered.iterrows()):
		with grid_cols[idx % 4]:
			if row["image"]:
				st.image(row["image"], use_column_width=True)
			st.markdown(f"**{row['title']}**")
			st.caption(f"{row['brand'] or ''} · {row['category'] or ''}")
			st.write(f"${row['price']}")
			st.progress(float(max(0.0, min(1.0, row["similarity"]))))

	with st.expander("Debug Info"):
		st.write({"num_products": len(products_df), "displayed": len(filtered)})


if __name__ == "__main__":
	main()