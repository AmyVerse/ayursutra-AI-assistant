from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from typing import List, Optional, Dict
import json
import re
import google.generativeai as genai

# FastAPI app - API only for Next.js frontend
app = FastAPI(
    title="Ayursutra AI API", 
    version="2.0.0",
    description="API backend for Next.js Ayurvedic AI Assistant"
)

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-nextjs-app.vercel.app"],  # Add your Next.js URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
model = None

if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        model_names = [
            'models/gemini-2.5-flash',
            'models/gemini-2.5-pro', 
            'models/gemini-2.0-flash'
        ]
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                test_response = model.generate_content("Hello")
                print(f"✅ AI Model: {model_name}")
                break
            except:
                continue
                
        if not model:
            print("❌ No AI model available - using fallback")
            
    except Exception as e:
        print(f"⚠️ Gemini setup error: {e}")
        model = None

# API Models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"

class MedicineSearchRequest(BaseModel):
    medicine_name: str
    max_results: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    confidence: float

class MedicineResponse(BaseModel):
    medicines: List[Dict]
    total_found: int

# Medicine sources for web scraping
MEDICINE_SOURCES = {
    "1mg": {
        "url": "https://www.1mg.com/search/all?name={query}",
        "selectors": {
            "container": [".style__product-card___1gbex", ".medicine-unit-wrap"],
            "name": [".style__product-name___B2eik", "h3"],
            "price": [".style__price-tag___B2eik", ".price"]
        }
    },
    "netmeds": {
        "url": "https://www.netmeds.com/catalogsearch/result?q={query}",
        "selectors": {
            "container": [".product-item-info", ".category-item"],
            "name": [".product-name", "h2 a"],
            "price": [".price", ".final-price"]
        }
    }
}

async def scrape_medicines(medicine_name: str, max_results: int = 5) -> List[Dict]:
    """Scrape medicine data from pharmacy websites"""
    results = []
    search_query = medicine_name.replace(" ", "+")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10), 
        headers=headers
    ) as session:
        
        for source_name, config in MEDICINE_SOURCES.items():
            try:
                url = config["url"].format(query=search_query)
                
                async with session.get(url) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Find medicine containers
                        items = []
                        for selector in config["selectors"]["container"]:
                            items = soup.select(selector)
                            if items:
                                break
                        
                        # Extract medicine info
                        for item in items[:3]:  # Limit per source
                            try:
                                name = ""
                                for name_sel in config["selectors"]["name"]:
                                    name_elem = item.select_one(name_sel)
                                    if name_elem:
                                        name = name_elem.get_text(strip=True)
                                        break
                                
                                price = "Check website"
                                for price_sel in config["selectors"]["price"]:
                                    price_elem = item.select_one(price_sel)
                                    if price_elem:
                                        price = price_elem.get_text(strip=True)
                                        break
                                
                                if name and len(name) > 3:
                                    results.append({
                                        "name": name[:100],
                                        "price": price,
                                        "source": source_name.upper(),
                                        "availability": "Online"
                                    })
                            except:
                                continue
                                
            except Exception as e:
                print(f"Error scraping {source_name}: {e}")
                continue
    
    # Fallback if no results
    if not results:
        results.append({
            "name": f"{medicine_name.title()}",
            "price": "Contact Ayurvedic store",
            "source": "GENERAL",
            "availability": "Available"
        })
    
    return results[:max_results]

async def get_ai_response(message: str) -> str:
    """Get AI response from Gemini"""
    if not model:
        return get_fallback_response(message)
    
    try:
        prompt = f"""You are an Ayurvedic health assistant. Provide helpful advice for: {message}

Guidelines:
- Be conversational and helpful
- Suggest Ayurvedic remedies when appropriate
- Keep responses concise (150-250 words)
- Always recommend consulting practitioners for serious issues
- Use bullet points for clarity

Response:"""
        
        config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=350,
            candidate_count=1
        )
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: model.generate_content(prompt, generation_config=config)
        )
        
        return response.text
        
    except Exception as e:
        print(f"AI Error: {e}")
        return get_fallback_response(message)

def get_fallback_response(message: str) -> str:
    """Fallback responses when AI is unavailable"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ['cold', 'cough', 'fever']):
        return """🌿 **For Cold & Cough:**
• Warm honey-ginger water
• Tulsi tea 3x daily  
• Sitopaladi Churna with honey
• Rest and avoid cold foods

⚠️ Consult doctor if symptoms persist."""
        
    elif any(word in message_lower for word in ['headache', 'pain']):
        return """🧠 **For Headaches:**
• Peppermint oil on temples
• Deep breathing exercises
• Adequate sleep and hydration
• Godanti Mishran (consult practitioner)

⚠️ See doctor for severe/recurring headaches."""
        
    else:
        return """🙏 **Ayursutra AI Assistant**

I can help with:
• Health concerns and symptoms
• Ayurvedic remedy suggestions  
• Medicine information
• Lifestyle guidance

Please describe your specific health concern!"""

# API Endpoints

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for Next.js frontend"""
    try:
        ai_response = await get_ai_response(request.message)
        return ChatResponse(
            response=ai_response,
            confidence=0.9 if model else 0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/medicines/search", response_model=MedicineResponse)
async def search_medicines_endpoint(request: MedicineSearchRequest):
    """Medicine search endpoint for Next.js frontend"""
    try:
        medicines = await scrape_medicines(request.medicine_name, request.max_results)
        return MedicineResponse(
            medicines=medicines,
            total_found=len(medicines)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Medicine search error: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check for deployment"""
    return {
        "status": "healthy",
        "ai_available": model is not None,
        "version": "2.0.0"
    }

@app.get("/")
async def root():
    """API info"""
    return {
        "name": "Ayursutra AI API",
        "version": "2.0.0", 
        "description": "FastAPI backend for Next.js Ayurvedic AI Assistant",
        "endpoints": {
            "chat": "POST /api/chat",
            "medicine_search": "POST /api/medicines/search",
            "health": "GET /api/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if os.getenv("RAILWAY_ENVIRONMENT") else "127.0.0.1"
    print(f"🚀 Starting Ayursutra AI API on {host}:{port}")
    uvicorn.run("app:app", host=host, port=port, reload=False)