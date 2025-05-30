import requests
from bs4 import BeautifulSoup
import re
import time
import json

# ==========================================
# CHOOSE YOUR MISTRAL 7B IMPLEMENTATION
# ==========================================

# OPTION 1: Using Hugging Face Transformers (Local)
def mistral_inference_hf(prompt):
    """
    For local Mistral 7B using Hugging Face transformers
    Uncomment and install: pip install transformers torch accelerate
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load model and tokenizer (do this once, ideally outside the function)
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Format prompt for Mistral
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
        
    except Exception as e:
        return f"Error with HuggingFace model: {str(e)}"

# OPTION 2: Using Ollama (Local)
def mistral_inference_ollama(prompt):
    """
    For Mistral 7B running via Ollama
    Install Ollama and run: ollama pull mistral
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'mistral',
                'prompt': prompt,
                'stream': False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Ollama error: {response.status_code}"
            
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# OPTION 3: Using LM Studio API (Local)
def mistral_inference_lmstudio(prompt):
    """
    For Mistral 7B running via LM Studio
    Start LM Studio server on localhost:1234
    """
    try:
        response = requests.post(
            'http://localhost:1234/v1/chat/completions',
            headers={'Content-Type': 'application/json'},
            json={
                'model': 'mistral-7b',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 512
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"LM Studio error: {response.status_code}"
            
    except Exception as e:
        return f"Error connecting to LM Studio: {str(e)}"

# OPTION 4: Using any OpenAI-compatible API
def mistral_inference_openai_api(prompt):
    """
    For any OpenAI-compatible API endpoint
    """
    try:
        # Replace with your API endpoint and key
        API_BASE = "http://your-api-endpoint.com/v1"
        API_KEY = "your-api-key"  # If needed
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}'  # If API key required
        }
        
        response = requests.post(
            f'{API_BASE}/chat/completions',
            headers=headers,
            json={
                'model': 'mistral-7b',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.7,
                'max_tokens': 512
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"API error: {response.status_code}"
            
    except Exception as e:
        return f"Error with API: {str(e)}"

# ==========================================
# SELECT YOUR IMPLEMENTATION HERE
# ==========================================

def mistral_inference(prompt):
    """
    CHANGE THIS LINE to use your preferred implementation:
    """
    # Uncomment ONE of these based on your setup:
    
    # return mistral_inference_hf(prompt)        # For HuggingFace local
    # return mistral_inference_ollama(prompt)    # For Ollama
    # return mistral_inference_lmstudio(prompt)  # For LM Studio
    # return mistral_inference_openai_api(prompt) # For API endpoints
    
    # TEMPORARY: Remove this and uncomment one above
    # return "Please configure mistral_inference() function with your Mistral 7B setup!"
    return mistral_inference_ollama(prompt)    # For Ollama

# ==========================================
# REST OF THE RAG PIPELINE (UNCHANGED)
# ==========================================

def is_answer_limited(answer):
    """Enhanced check for limited responses - more aggressive for small models."""
    limited_phrases = [
        "not yet", "no information", "don't know", "as of now",
        "cannot answer", "unable to", "not available", "no data", 
        "no winner", "unknown", "sorry", "don't have", "insufficient", 
        "limited", "can't provide", "i'm not sure", "unclear",
        "i don't", "not certain", "not aware", "no details",
        "cannot provide", "don't possess", "lack information"
    ]
    
    answer_lower = answer.lower().strip()
    
    for phrase in limited_phrases:
        if phrase in answer_lower:
            return True
    
    if len(answer.strip()) < 30:
        return True
    
    vague_patterns = [
        "it depends", "various factors", "many reasons", "several ways",
        "different approaches", "multiple options", "it varies"
    ]
    
    for pattern in vague_patterns:
        if pattern in answer_lower and len(answer.strip()) < 100:
            return True
    
    return False

def duckduckgo_search(query, max_results=5):
    """Search DuckDuckGo and return URLs."""
    url = "https://html.duckduckgo.com/html/"
    data = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    try:
        response = requests.post(url, data=data, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for a in soup.find_all("a", class_="result__a", limit=max_results):
            href = a.get("href")
            if href and href.startswith("http"):
                results.append(href)
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
        return []

def clean_text(text):
    """Clean and normalize text."""
    return re.sub(r'\s+', ' ', text).strip()

def is_valid_paragraph(text):
    """Filter out unwanted content like ads, cookies, etc."""
    invalid_words = ["cookie", "advertisement", "privacy", "subscribe", "terms", "javascript"]
    if any(word in text.lower() for word in invalid_words):
        return False
    return len(text) > 50

def scrape_summary(url, max_paragraphs=3):
    """Scrape and summarize content from a URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        paragraphs = soup.find_all("p")
        summary = []
        
        for p in paragraphs:
            text = clean_text(p.get_text())
            if is_valid_paragraph(text):
                summary.append(text)
            if len(summary) >= max_paragraphs:
                break

        if summary:
            return "\n\n".join(summary)
        
        if soup.title:
            return clean_text(soup.title.get_text())
        
        return "No meaningful content found."
        
    except Exception as e:
        return f"Failed to scrape page: {str(e)}"

def rag_pipeline():
    """
    Enhanced RAG Pipeline for Mistral 7B with aggressive context injection
    """
    print("🤖 Mistral 7B + DuckDuckGo RAG Pipeline")
    print("=" * 60)
    print("Optimized for limited models - aggressive context injection")
    print("Type 'exit' to quit.\n")
    #
    while True:
        user_query = input("💬 Your Question: ").strip()
        if user_query.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break

        print(f"\n🔄 Processing: '{user_query}'")
        print("-" * 50)

        # STEP 1: Initial Mistral Response
        print("1️⃣ Querying Mistral 7B...")
        try:
            initial_response = mistral_inference(user_query)
            print(f"📝 Initial Response:\n{initial_response}\n")
        except Exception as e:
            print(f"❌ Error calling Mistral: {e}")
            continue

        # STEP 2: Check if we need web context
        needs_context = (
            is_answer_limited(initial_response) or 
            len(initial_response.strip()) < 100 or
            any(keyword in user_query.lower() for keyword in [
                'latest', 'recent', 'current', 'today', 'news', 'update', 
                'when', 'what happened', 'status', '2024', '2025'
            ])
        )

        if needs_context:
            print("🔍 Triggering web search for context enhancement...")
            
            # STEP 3: Search and scrape
            print("🌐 Searching DuckDuckGo...")
            urls = duckduckgo_search(user_query, max_results=3)
            
            if not urls:
                print("❌ No search results found.")
                print(f"💡 Final Answer: {initial_response}")
                continue
            
            print(f"✅ Found {len(urls)} sources")
            
            # STEP 4: Get context from multiple sources
            all_context = []
            for i, url in enumerate(urls[:2]):
                print(f"📄 Scraping source {i+1}: {url[:50]}...")
                summary = scrape_summary(url, max_paragraphs=2)
                if "Failed to scrape" not in summary and len(summary) > 50:
                    all_context.append(f"Source {i+1}: {summary}")
            
            if not all_context:
                print("❌ Failed to retrieve useful context.")
                print(f"💡 Final Answer: {initial_response}")
                continue

            combined_context = "\n\n".join(all_context)
            print(f"📋 Retrieved Context ({len(combined_context)} chars)")
            
            # STEP 5: Enhanced prompt for Mistral 7B
            print("🧠 Generating context-aware response...")
            
            enhanced_prompt = f"""CONTEXT INFORMATION:
{combined_context}

INSTRUCTION: Using the context information provided above, answer this question thoroughly:

QUESTION: {user_query}

IMPORTANT: Base your answer on the context provided. Be specific and detailed.

ANSWER:"""

            try:
                final_response = mistral_inference(enhanced_prompt)
                print(f"🎯 Enhanced Answer:\n{final_response}")
                
                # Quality check and alternative approach if needed
                if is_answer_limited(final_response) and len(final_response.strip()) < 50:
                    print("🔄 Trying simpler prompt format...")
                    
                    simple_prompt = f"""Information: {combined_context}

Question: {user_query}
Answer based on the information above:"""
                    
                    alternative_response = mistral_inference(simple_prompt)
                    print(f"🔄 Alternative Answer:\n{alternative_response}")
                
            except Exception as e:
                print(f"❌ Error in enhanced inference: {e}")
                print(f"💡 Fallback Answer: {initial_response}")

        else:
            print("✅ Initial response seems sufficient")
            print(f"💡 Final Answer: {initial_response}")

        print("\n" + "="*50)
        time.sleep(1)

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("🚀 Starting RAG Pipeline...")
    print("📋 Setup Instructions:")
    print("1. Choose your Mistral 7B implementation in mistral_inference()")
    print("2. Install required packages: pip install requests beautifulsoup4")
    print("3. Install your chosen Mistral implementation (HF, Ollama, etc.)")
    print()
    
    rag_pipeline()
    

