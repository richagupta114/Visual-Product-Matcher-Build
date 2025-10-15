from __future__ import annotations

from typing import List, Dict

import streamlit as st


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
	"""Return a mock list of >=50 products with images and metadata."""
	return [_make_product(i) for i in range(1, 81)]