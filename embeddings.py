from __future__ import annotations

import io
import numpy as np
from PIL import Image
import streamlit as st
import requests


@st.cache_data(show_spinner=False)
def _image_to_feature(image_bytes: bytes) -> np.ndarray:
	image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
	arr = np.asarray(image)
	# Simple color histogram (normalized)
	hist_r, _ = np.histogram(arr[:, :, 0], bins=32, range=(0, 255), density=True)
	hist_g, _ = np.histogram(arr[:, :, 1], bins=32, range=(0, 255), density=True)
	hist_b, _ = np.histogram(arr[:, :, 2], bins=32, range=(0, 255), density=True)
	feat = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)
	feat /= np.linalg.norm(feat) + 1e-8
	return feat


def get_image_embedding(image: Image.Image) -> np.ndarray:
	buf = io.BytesIO()
	image.save(buf, format="PNG")
	return _image_to_feature(buf.getvalue())


def precompute_product_embeddings(products_df) -> np.ndarray:
	image_urls = products_df["image"].fillna("").astype(str).tolist()
	embeddings: list[np.ndarray] = []
	for url in image_urls:
		try:
			img = Image.open(io.BytesIO(requests.get(url, timeout=15).content)).convert("RGB")
			emb = get_image_embedding(img)
		except Exception:
			# Use a zero vector if image fetch fails
			emb = np.zeros(96, dtype=np.float32)
		embeddings.append(emb)
	return np.vstack(embeddings)