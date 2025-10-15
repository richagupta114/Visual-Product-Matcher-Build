## Visual Product Matcher

A Streamlit web app that finds visually similar products using CLIP embeddings. Users can upload an image or paste an image URL to search a catalog of 100+ products (from DummyJSON) and see the most similar items with adjustable filters.

### Demo (deploy yourself)
- Deploy to Streamlit Community Cloud: fork this repo, then create a new app pointing to `app.py`.

### Features
- Image upload and image URL input
- Live preview of the uploaded image
- Similarity search using CLIP (`clip-ViT-B-32` via `sentence-transformers`)
- Results list with thumbnails, metadata (title, category, brand, price), and similarity score
- Filters: top-K and minimum similarity threshold
- Mobile-responsive Streamlit layout
- Caching of product data and embeddings for fast responses

### Local Setup
```bash
# Create venv (optional)
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### Data Source
Products are fetched from DummyJSON (`https://dummyjson.com/products?limit=100`) and cached locally under `cache/products.json`.

### Approach (≤200 words)
This app uses CLIP image embeddings to perform visual similarity search on product images. The catalog (≥50 products) is fetched from a public API (DummyJSON) and normalized into a compact dataframe containing metadata and a primary image URL. On startup, the app precomputes (and caches) embeddings for product images using `sentence-transformers` (`clip-ViT-B-32`). For user queries, the uploaded image or image URL is embedded with the same model. Cosine similarity between the query embedding and precomputed catalog embeddings ranks results. Streamlit provides a responsive UI with an image uploader, URL input, and sidebar filters for top-K and minimum similarity. The app includes loading spinners, error handling for network/model failures, and on-disk caching to reduce cold-start latency on free hosting. The design favors simplicity and production readability: clear module boundaries (`data_ingestion.py`, `embeddings.py`, `app.py`), caching via Streamlit decorators, and minimal dependencies. The app runs fully on free tiers with actual model inference.

### Notes
- First run may download the CLIP model weights; subsequent runs use cache.
- If running on Streamlit Cloud, ensure `requirements.txt` is present.