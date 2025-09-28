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
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

app = FastAPI(title="Ayursutra AI Assistant", version="1.0.0")

# CORS middleware for Next.js integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)
    model = genai.GenerativeModel('gemini-pro')

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"

class MedicineQuery(BaseModel):
    medicine_name: str
    max_results: Optional[int] = 10

class SymptomQuery(BaseModel):
    symptoms: str
    age: Optional[int] = None
    gender: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    suggested_medicines: Optional[List[str]] = None
    confidence: Optional[float] = None

# Ayurvedic knowledge base for better responses
AYURVEDIC_KNOWLEDGE = """
You are an expert Ayurvedic AI assistant. You have deep knowledge of:
- Ayurvedic principles (Vata, Pitta, Kapha doshas)
- Traditional herbs and medicines
- Home remedies and natural treatments
- Ayurvedic lifestyle recommendations
- Seasonal health practices

Always provide helpful, accurate information while reminding users to consult qualified practitioners for serious conditions.
"""

# Web scraping configuration
PHARMACY_SOURCES = {
    "1mg": {
        "url": "https://www.1mg.com/search/all?name={query}",
        "selectors": {
            "container": ["div[data-testid='product-card']", ".product-card", ".medicine-unit-wrap"],
            "name": ["h3", "h4", ".product-name", "[data-testid='product-name']"],
            "price": [".price", ".cost", "[data-testid='price']", ".price-box span"],
            "rating": [".rating", ".stars", "[data-testid='rating']"]
        }
    },
    "netmeds": {
        "url": "https://www.netmeds.com/catalogsearch/result?q={query}",
        "selectors": {
            "container": [".product-item", ".medicine-card", ".product-wrapper"],
            "name": [".product-name", "h3", "h4"],
            "price": [".price", ".final-price", ".offer-price"],
            "rating": [".rating", ".star-rating"]
        }
    },
    "pharmeasy": {
        "url": "https://pharmeasy.in/search/all?name={query}",
        "selectors": {
            "container": [".ProductCard_medicineUnitWrapper", ".product-card"],
            "name": [".ProductCard_medicineName", ".product-name"],
            "price": [".ProductCard_gcdDiscountContainer", ".price"],
            "rating": [".ProductCard_ratingWrapper", ".rating"]
        }
    },
    "baidyanath": {
        "url": "https://www.baidyanath.com/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card", ".grid-product"],
            "name": [".product-title", "h3", "h4", ".product-name"],
            "price": [".price", ".money", ".product-price", ".current-price"],
            "rating": [".reviews", ".rating", ".stars"]
        }
    },
    "dabur": {
        "url": "https://www.dabur.com/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card", ".search-result"],
            "name": [".product-title", ".product-name", "h3", "h4"],
            "price": [".price", ".product-price", ".cost"],
            "rating": [".rating", ".reviews", ".stars"]
        }
    },
    "patanjali": {
        "url": "https://www.patanjaliayurveda.net/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card", ".grid-item"],
            "name": [".product-title", ".product-name", "h3"],
            "price": [".price", ".product-price", ".money"],
            "rating": [".rating", ".reviews"]
        }
    },
    "apollopharmacy": {
        "url": "https://www.apollopharmacy.in/search-medicines/{query}",
        "selectors": {
            "container": [".ProductCard", ".product-card", ".medicine-card"],
            "name": [".ProductName", ".product-name", "h3"],
            "price": [".Price", ".price", ".cost"],
            "rating": [".Rating", ".rating", ".stars"]
        }
    },
    "zandu": {
        "url": "https://www.zandu.in/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card"],
            "name": [".product-title", ".product-name"],
            "price": [".price", ".product-price"],
            "rating": [".rating", ".reviews"]
        }
    },
    "himalaya": {
        "url": "https://himalayawellness.in/search?q={query}",
        "selectors": {
            "container": [".product-item", ".product-card"],
            "name": [".product-title", ".product-name"],
            "price": [".price", ".product-price"],
            "rating": [".rating", ".reviews"]
        }
    }
}

async def scrape_medicine_prices(medicine_name: str, max_results: int = 5) -> List[Dict]:
    """Scrape medicine prices from multiple sources with dynamic selectors"""
    results = []
    query = medicine_name.replace(' ', '%20')
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10),
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    ) as session:
        
        for source_name, config in list(PHARMACY_SOURCES.items())[:4]:  # Limit to 4 sources for speed
            try:
                url = config["url"].format(query=query)
                
                async with session.get(url, timeout=8) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Try different container selectors
                        medicines = []
                        for selector in config["selectors"]["container"]:
                            medicines = soup.select(selector)[:3]
                            if medicines:
                                break
                        
                        for med in medicines:
                            try:
                                # Extract name
                                name = None
                                for name_selector in config["selectors"]["name"]:
                                    name_elem = med.select_one(name_selector)
                                    if name_elem:
                                        name = name_elem.get_text(strip=True)
                                        break
                                
                                # Extract price
                                price = None
                                for price_selector in config["selectors"]["price"]:
                                    price_elem = med.select_one(price_selector)
                                    if price_elem:
                                        price = price_elem.get_text(strip=True)
                                        break
                                
                                # Extract rating if available
                                rating = "N/A"
                                for rating_selector in config["selectors"]["rating"]:
                                    rating_elem = med.select_one(rating_selector)
                                    if rating_elem:
                                        rating = rating_elem.get_text(strip=True)
                                        break
                                
                                if name and price:
                                    results.append({
                                        "Medicine": name[:100],  # Limit length
                                        "Price": price,
                                        "Source": source_name.title(),
                                        "Stock": "Check Availability",
                                        "Rating": rating,
                                        "Link": url
                                    })
                                    
                            except Exception as item_error:
                                print(f"Error extracting item from {source_name}: {item_error}")
                                continue
                                
            except Exception as source_error:
                print(f"Error scraping {source_name}: {source_error}")
                continue
    
    return results[:max_results]

# LLM Integration
async def get_ai_response(message: str, context: str = "") -> ChatResponse:
    """Get AI response using Google Gemini or fallback to rule-based system"""
    try:
        if google_api_key and model:
            # Prepare the prompt for Gemini
            full_prompt = f"""
{AYURVEDIC_KNOWLEDGE}

Context: {context}

User Query: {message}

Please provide a helpful response about Ayurvedic treatment, remedies, and medicine suggestions. 
Be informative but remind users to consult healthcare professionals for serious conditions.
If suggesting medicines, mention specific Ayurvedic herbs or formulations.
"""
            
            # Generate response using Gemini
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: model.generate_content(full_prompt)
            )
            
            ai_response = response.text
            
            # Extract suggested medicines from response
            suggested_medicines = extract_medicine_suggestions(ai_response)
            
            return ChatResponse(
                response=ai_response,
                suggested_medicines=suggested_medicines,
                confidence=0.9
            )
        else:
            # Fallback rule-based system
            return get_rule_based_response(message)
            
    except Exception as e:
        print(f"AI response error: {e}")
        return get_rule_based_response(message)

def extract_medicine_suggestions(text: str) -> List[str]:
    """Extract medicine suggestions from AI response"""
    # Extended list of common Ayurvedic medicines and herbs
    ayurvedic_medicines = [
        'Ashwagandha', 'Triphala', 'Tulsi', 'Giloy', 'Guduchi', 'Turmeric', 'Ginger', 'Neem', 
        'Brahmi', 'Shankhpushpi', 'Amla', 'Arjuna', 'Bala', 'Bhringraj', 'Chyavanprash',
        'Dashmool', 'Fenugreek', 'Garlic', 'Haritaki', 'Jatamansi', 'Kumari', 'Licorice',
        'Manjistha', 'Moringa', 'Nutmeg', 'Pippali', 'Punarnava', 'Rasayana', 'Shatavari',
        'Tagar', 'Vidanga', 'Yashtimadhu', 'Zinc', 'Aloevera', 'Cardamom', 'Cinnamon',
        'Sitopaladi Churna', 'Taleesadi Churna', 'Avipattikar Churna', 'Hingvastak Churna',
        'Godanti Mishran', 'Shirashooladi Vajra Ras', 'Panchakola Churna', 'Trikatu Churna',
        'Madana Phala', 'Saindhava Lavana', 'Honey', 'Ghee', 'Rock Salt'
    ]
    
    # Create pattern from the medicine list
    pattern = r'\b(?:' + '|'.join(re.escape(med) for med in ayurvedic_medicines) + r')\b'
    medicines = re.findall(pattern, text, re.IGNORECASE)
    return list(set(medicines))

def get_rule_based_response(message: str) -> ChatResponse:
    """Fallback rule-based response system"""
    message_lower = message.lower()
    
    # Common symptom responses
    if any(word in message_lower for word in ['cold', 'cough', 'fever']):
        return ChatResponse(
            response="For cold and cough, try these Ayurvedic remedies: 1) Warm water with honey and ginger 2) Tulsi tea 3) Turmeric milk. Consider medicines like Sitopaladi Churna or Taleesadi Churna.",
            suggested_medicines=["Sitopaladi Churna", "Taleesadi Churna", "Tulsi"],
            confidence=0.7
        )
    
    elif any(word in message_lower for word in ['headache', 'migraine']):
        return ChatResponse(
            response="For headaches, try: 1) Apply peppermint oil to temples 2) Drink ginger tea 3) Practice Pranayama. Godanti Mishran or Shirashooladi Vajra Ras may help.",
            suggested_medicines=["Godanti Mishran", "Shirashooladi Vajra Ras"],
            confidence=0.7
        )
    
    elif any(word in message_lower for word in ['digestion', 'stomach', 'acidity']):
        return ChatResponse(
            response="For digestive issues: 1) Drink warm water with lemon 2) Chew fennel seeds after meals 3) Try Triphala before bed. Consider Avipattikar Churna or Hingvastak Churna.",
            suggested_medicines=["Avipattikar Churna", "Hingvastak Churna", "Triphala"],
            confidence=0.7
        )
    
    else:
        return ChatResponse(
            response="I'm here to help with Ayurvedic health guidance. Could you please describe your symptoms or health concerns more specifically? I can suggest appropriate remedies and medicines.",
            suggested_medicines=None,
            confidence=0.5
        )
# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to Ayursutra AI Assistant",
        "version": "1.0.0",
        "status": "running",
        "ai_enabled": bool(google_api_key),
        "endpoints": ["/chat", "/medicines/search", "/medicines/list", "/symptoms/analyze", "/medicines/categories"]
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(chat_message: ChatMessage):
    """Chat with AI assistant for health guidance"""
    try:
        response = await get_ai_response(chat_message.message)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/medicines/search")
async def search_medicines_realtime(query: MedicineQuery):
    """Search medicines with real-time web scraping and knowledge base"""
    try:
        # Get dynamic medicine data (combines web scraping + knowledge base)
        all_results = await get_dynamic_medicine_data(query.medicine_name)
        
        # Separate realtime and knowledge base results
        realtime_results = [r for r in all_results if r.get("Source") != "Knowledge Base"]
        knowledge_results = [r for r in all_results if r.get("Source") == "Knowledge Base"]
        
        return {
            "query": query.medicine_name,
            "realtime_results": realtime_results[:query.max_results],
            "knowledge_base_results": knowledge_results,
            "total_found": len(all_results),
            "sources": ["Web Scraping", "Ayurvedic Knowledge Base"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/medicines/list")
async def get_all_medicines():
    """Get list of all available medicines from knowledge base"""
    all_medicines = []
    
    # Collect all medicines from categories
    for category, medicines in AYURVEDIC_MEDICINE_CATEGORIES.items():
        for medicine in medicines:
            all_medicines.append({
                "name": medicine,
                "category": category.replace("_", " ").title(),
                "type": "Ayurvedic Medicine"
            })
    
    # Remove duplicates and sort
    unique_medicines = []
    seen = set()
    for med in all_medicines:
        if med["name"] not in seen:
            unique_medicines.append(med)
            seen.add(med["name"])
    
    unique_medicines.sort(key=lambda x: x["name"])
    
    return {
        "medicines": unique_medicines,
        "categories": list(AYURVEDIC_MEDICINE_CATEGORIES.keys()),
        "total_count": len(unique_medicines),
        "category_count": len(AYURVEDIC_MEDICINE_CATEGORIES)
    }

@app.post("/symptoms/analyze")
async def analyze_symptoms(symptom_query: SymptomQuery):
    """Analyze symptoms and suggest treatments"""
    try:
        context = f"Patient details - Age: {symptom_query.age}, Gender: {symptom_query.gender}"
        response = await get_ai_response(
            f"I have these symptoms: {symptom_query.symptoms}. Please suggest Ayurvedic remedies and medicines.",
            context
        )
        
        # Get category-based suggestions
        category_suggestions = get_medicine_suggestions_by_category(symptom_query.symptoms)
        
        # Combine AI suggestions with category-based suggestions
        all_suggestions = list(set((response.suggested_medicines or []) + category_suggestions))
        
        # Get detailed information for suggested medicines
        suggested_details = []
        for med in all_suggestions[:10]:  # Limit to 10 suggestions
            try:
                med_details = await get_dynamic_medicine_data(med)
                suggested_details.extend(med_details[:2])  # Max 2 results per medicine
            except:
                # Add basic information if scraping fails
                suggested_details.append({
                    "Medicine": med,
                    "Type": "Ayurvedic Medicine",
                    "Source": "Knowledge Base",
                    "Note": "Consult practitioner for dosage"
                })
        
        return {
            "analysis": response.response,
            "suggested_medicines": all_suggestions,
            "medicine_details": suggested_details,
            "confidence": response.confidence,
            "symptom_categories": [cat for cat in AYURVEDIC_MEDICINE_CATEGORIES.keys() 
                                 if any(keyword in symptom_query.symptoms.lower() 
                                       for keyword in ["digestive", "respiratory", "immunity", "stress", "pain"])]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.get("/medicines/categories")
async def get_medicine_categories():
    """Get medicine categories and their medicines"""
    categories_with_medicines = {}
    for category, medicines in AYURVEDIC_MEDICINE_CATEGORIES.items():
        categories_with_medicines[category] = {
            "name": category.replace("_", " ").title(),
            "medicines": medicines,
            "count": len(medicines)
        }
    
    return {
        "categories": categories_with_medicines,
        "total_categories": len(AYURVEDIC_MEDICINE_CATEGORIES),
        "description": "Ayurvedic medicine categories based on traditional usage"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy", 
        "message": "Ayursutra AI Assistant is running",
        "features": [
            "Google Gemini AI Integration",
            "Real-time Medicine Search",
            "Dynamic Knowledge Base",
            "Symptom Analysis",
            "Category-based Recommendations"
        ]
    }

# Dynamic Medicine Categories Database
AYURVEDIC_MEDICINE_CATEGORIES = {
    "digestive": ["Triphala", "Avipattikar Churna", "Hingvastak Churna", "Ajwain", "Hing"],
    "respiratory": ["Sitopaladi Churna", "Taleesadi Churna", "Tulsi", "Honey", "Ginger"],
    "immunity": ["Giloy", "Guduchi", "Ashwagandha", "Chyavanprash", "Amla"],
    "stress_anxiety": ["Brahmi", "Shankhpushpi", "Jatamansi", "Ashwagandha", "Tagar"],
    "pain_inflammation": ["Turmeric", "Ginger", "Dashmool", "Nirgundi", "Mahanarayana Oil"],
    "skin_hair": ["Neem", "Manjistha", "Bhringraj", "Aloevera", "Kumkumadi Oil"],
    "women_health": ["Shatavari", "Lodhra", "Pushyanug Churna", "Kumaryasava", "Dashmool"],
    "men_health": ["Ashwagandha", "Safed Musli", "Kapikachhu", "Gokshura", "Shilajit"],
    "heart_circulation": ["Arjuna", "Garlic", "Punarnava", "Hridayarnava Ras", "Terminalia"],
    "diabetes": ["Karela", "Jamun", "Methi", "Vijaysar", "Gudmar"],
    "liver_detox": ["Bhumi Amla", "Kalmegh", "Kutki", "Punarnava", "Liv-52"],
    "joints_bones": ["Guggul", "Shallaki", "Rasna", "Nirgundi", "Yograj Guggul"]
}

def get_medicine_suggestions_by_category(symptoms: str) -> List[str]:
    """Get medicine suggestions based on symptom categories"""
    suggestions = []
    symptoms_lower = symptoms.lower()
    
    # Map symptoms to categories
    symptom_mapping = {
        "digestive": ["stomach", "acidity", "indigestion", "gas", "bloating", "constipation"],
        "respiratory": ["cough", "cold", "asthma", "breathing", "chest", "throat"],
        "immunity": ["fever", "infection", "weak", "immunity", "frequent illness"],
        "stress_anxiety": ["stress", "anxiety", "depression", "sleep", "mental", "worry"],
        "pain_inflammation": ["pain", "inflammation", "swelling", "arthritis", "headache"],
        "skin_hair": ["skin", "hair", "acne", "rash", "eczema", "dandruff"],
        "women_health": ["menstrual", "periods", "pregnancy", "fertility", "hormonal"],
        "men_health": ["stamina", "energy", "vitality", "testosterone", "strength"],
        "heart_circulation": ["heart", "blood pressure", "circulation", "cholesterol"],
        "diabetes": ["sugar", "diabetes", "blood sugar", "glucose"],
        "liver_detox": ["liver", "detox", "cleanse", "toxins", "jaundice"],
        "joints_bones": ["joint", "bone", "arthritis", "stiffness", "mobility"]
    }
    
    # Find matching categories
    for category, keywords in symptom_mapping.items():
        if any(keyword in symptoms_lower for keyword in keywords):
            suggestions.extend(AYURVEDIC_MEDICINE_CATEGORIES.get(category, []))
    
    return list(set(suggestions))[:10]  # Return unique suggestions, limit to 10

async def get_dynamic_medicine_data(medicine_name: str) -> List[Dict]:
    """Get dynamic medicine data by combining web scraping with knowledge base"""
    
    # Get real-time data from web scraping
    realtime_data = await scrape_medicine_prices(medicine_name, max_results=5)
    
    # Add knowledge base information
    knowledge_data = []
    
    # Check if medicine exists in our categories
    for category, medicines in AYURVEDIC_MEDICINE_CATEGORIES.items():
        if any(med.lower() in medicine_name.lower() for med in medicines):
            knowledge_data.append({
                "Medicine": medicine_name,
                "Category": category.replace("_", " ").title(),
                "Type": "Ayurvedic Medicine",
                "Usage": f"Traditional use for {category.replace('_', ' ')} related conditions",
                "Source": "Knowledge Base",
                "Note": "Consult an Ayurvedic practitioner for proper dosage"
            })
    
    return realtime_data + knowledge_data

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

