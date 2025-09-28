# Ayursutra AI Assistant

AI-powered Ayurvedic medicine search and consultation API. Search any medicine and get real-time prices, ratings, and purchase links from multiple pharmacy websites.

## What it does:

- **AI Chat**: Ask about symptoms, get Ayurvedic medicine suggestions
- **Medicine Search**: Real-time web scraping from 10+ pharmacy websites
- **Sources**: 1mg, Netmeds, PharmEasy, Baidyanath, Dabur, Patanjali, Apollo, Zandu, Himalaya
- **Live Data**: Current prices, ratings, stock status, direct purchase links
- **API**: Perfect for Next.js/React frontends

## Setup:

1. Get Google Gemini API key (free): https://aistudio.google.com/app/apikey
2. Add to `.env`: `GOOGLE_API_KEY=your_key`
3. `pip install -r requirements.txt`
4. `python app.py`
5. API docs: `http://localhost:8000/docs`

## Example:

Search "Ashwagandha" → Get live prices from 10+ sources: Baidyanath, Dabur, Patanjali, 1mg, Netmeds, PharmEasy, Apollo, Zandu, Himalaya with ratings and purchase links.

Deploy on Railway by connecting your GitHub repo.
