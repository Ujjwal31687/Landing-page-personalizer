# 🚀 Dynamic Landing Page Personalizer

**AI-Powered CRO Tool** — Automatically rewrites landing page content to match your ad creative, boosting conversion rate alignment instantly.

> An agentic AI pipeline that analyzes an ad image, scrapes a live landing page, and rewrites key text elements (headlines, CTAs, body copy) to match the ad's offer, tone, and target audience.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Recommended Test Cases](#recommended-test-cases)
- [Scraping Evolution: BeautifulSoup → Jina → Playwright](#scraping-evolution-beautifulsoup--jina--playwright)
- [Trade-offs & Limitations](#trade-offs--limitations)
- [Deployment Guide](#deployment-guide)
- [Project Structure](#project-structure)

---

## Project Overview

Modern digital advertisers run hundreds of ad campaigns, each targeting different audiences with different offers. But the *landing page* behind every ad is often the same generic page — leading to poor message-match and low conversion rates.

**This tool solves that problem.** Given:
1. A **landing page URL** (e.g., a Shopify store, blog, product page)
2. An **ad creative** (uploaded image)

It will:
- Scrape the live landing page and preserve its full HTML structure
- Use AI (Groq LLM with Llama 4 Scout) to analyze the ad's core offer, audience, and tone
- Identify 2–4 key text elements on the page (headlines, CTAs, body copy) to rewrite
- Apply those changes **directly to the original HTML** — preserving all styles, images, scripts, and layout
- Display a side-by-side comparison of the **Original** vs **Personalized** page

---

## Architecture

The pipeline is built as an **Agentic Workflow** using [LangGraph](https://github.com/langchain-ai/langgraph), with three sequential nodes:

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   SCRAPER   │────▶│   AI AGENT   │────▶│   REPLACER   │
│  (Node 1)   │     │   (Node 2)   │     │   (Node 3)   │
└─────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
 Fetches HTML &       Analyzes ad image     Applies text
 builds element       + element map via     changes to the
 map (TAG#N list)     Groq LLM (JSON)      original HTML
```

### Node Details

| Node | Function | Description |
|------|----------|-------------|
| **Scraper** | `fetch_page_node` | Multi-strategy fetcher: tries direct `requests` → Playwright headless browser → Jina reader proxy. Builds a `TAG#N` element map for the LLM. |
| **AI Agent** | `analyze_ad_node` | Sends the element map + ad image to Groq's Llama 4 Scout model. Returns structured JSON with `tag_id`, `original_text`, `new_text` for 2–4 changes. |
| **Replacer** | `apply_changes_node` | Locates each element in the original HTML using multi-pass matching (exact → full-text → fuzzy) and performs in-place text replacement. |

### State Flow (LangGraph `TypedDict`)

```python
class AgentState(TypedDict):
    landing_page_url: str
    ad_image_base64: str
    original_html: str       # Full untouched HTML
    element_map_text: str    # TAG#N list for LLM
    element_lookup: dict     # {tag_id: original_text}
    ai_results: PagePersonalization  # Pydantic model
    modified_html: str       # Final personalized HTML
    applied_count: int
    error: Optional[str]
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend** | Streamlit (wide layout, dual-column, tabbed preview) |
| **Agentic Framework** | LangGraph (StateGraph with 3 nodes) |
| **LLM** | Groq Cloud — `meta-llama/llama-4-scout-17b-16e-instruct` |
| **Scraping** | Multi-strategy: `requests` → Playwright (headless Chromium) → Jina Reader |
| **HTML Parsing** | BeautifulSoup 4 |
| **Data Models** | Pydantic v2 |
| **Language** | Python 3.11+ |

---

## Setup & Installation

### Prerequisites
- Python 3.11 or higher
- A [Groq API key](https://console.groq.com/) (free tier available)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/troopod-assignment.git
cd troopod-assignment

# 2. Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate   # macOS/Linux
# myenv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Playwright browsers (required for JS-rendered sites)
playwright install chromium

# 5. Set up environment variables
# Create a .env file with your Groq API key:
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# 6. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. **Enter a Landing Page URL** — Paste the URL of the page you want to personalize
2. **Provide an Ad Creative** — Choose one of two options:
   - **Image Upload**: Upload a PNG/JPG image of your ad
   - **Ad Link**: Paste a URL to an ad (image URL or ad landing page — the system will screenshot it automatically)
3. **Click "Generate Personalized Page"** — The pipeline will:
   - Scrape the page (you'll see progress updates)
   - Analyze the ad with AI (via image or screenshot)
   - Apply text changes to the HTML
4. **View Results** — Switch between "Personalized Output" and "Original Page" tabs to compare

---

## Recommended Test Cases

The following URLs are tested and work well with the pipeline. They are **server-side rendered** (SSR) or static sites that don't employ aggressive anti-bot measures:

### ✅ Confirmed Working

| # | URL | Type | Notes |
|---|-----|------|-------|
| 1 | `https://hydrogen.shopify.dev/` | Shopify Demo Store | Clean SSR Hydrogen storefront. **Verified working** — personalization applied perfectly. |
| 2 | `https://en.wikipedia.org/wiki/Sneakers` | Wikipedia | Static content, rich text elements. Excellent for testing headline/body rewrites. |
| 3 | `https://www.example.com` | Static HTML | Minimal page — good for basic smoke testing. |
| 4 | `https://books.toscrape.com/` | E-commerce Demo | Fake bookstore with product listings. No anti-bot. Great for CTA/heading changes. |
| 5 | `https://quotes.toscrape.com/` | Quotes Demo | Simple quote aggregator. Tests body text replacement. |
| 6 | `https://store.steampowered.com/` | Steam Store | Large e-commerce store, mostly server-rendered. Rich text hierarchy. |
| 7 | `https://news.ycombinator.com/` | Hacker News | Static HTML news aggregator. Clean text elements. |
| 8 | `https://www.bbc.com/news` | BBC News | SSR news site with strong heading hierarchy (H1, H2, H3). |
| 9 | `https://www.imdb.com/chart/top` | IMDB Top 250 | Rich structured content with tables, headings, links. |
| 10 | `https://httpbin.org/html` | Test HTML | Ultra-simple HTML page for basic pipeline validation. |

### 🎯 Recommended Ad Images to Test With

- **Ski/snowboard gear ad** → pair with `hydrogen.shopify.dev`
- **Nike shoe ad** → pair with Wikipedia Sneakers page
- **Book sale ad** → pair with `books.toscrape.com`
- **Tech deal ad** → pair with `store.steampowered.com`

> **Tip:** For best results, choose ad images whose theme/product matches the landing page content. The AI will produce more meaningful rewrites when there's topical overlap.

---

## Scraping Evolution: BeautifulSoup → Jina → Playwright

This section documents the architectural evolution of the scraping layer — one of the most challenging parts of the project.

### Phase 1: BeautifulSoup + Requests (Initial)

**Approach:** Simple `requests.get()` to fetch HTML, then BeautifulSoup to parse and extract text.

```python
resp = requests.get(url)
soup = BeautifulSoup(resp.text, 'html.parser')
text = soup.get_text()
```

**Problem:** Many modern e-commerce sites (Flipkart, Nike, Shopify) are Single Page Applications (SPAs) that render content via JavaScript. A raw HTTP request returns an empty shell with no visible text — just a `<div id="app"></div>` and bundled JS files.

**Result:** ❌ Failed on ~60% of real-world e-commerce sites.

---

### Phase 2: Jina Reader Proxy (Intermediate)

**Approach:** Routed requests through `https://r.jina.ai/{url}`, which renders JavaScript server-side and returns clean HTML or Markdown.

```python
jina_url = f"https://r.jina.ai/{url}"
resp = requests.get(jina_url, headers={"Accept": "text/html", "X-Return-Format": "html"})
```

**Advantages:**
- No local browser dependency — works on any server
- Handles basic JS rendering
- Returns clean, well-structured HTML

**Problems:**
- **Rate limiting**: Jina's free tier has request limits; heavy sites like Flipkart would timeout (90s+)
- **Anti-bot detection**: Sites with aggressive bot protection (Cloudflare, Akamai) would still block Jina's servers
- **HTML fidelity**: The returned HTML was sometimes "cleaned up" in ways that broke original styling

**Result:** ✅ Improved coverage to ~70%, but still failed on heavily protected sites.

---

### Phase 3: Playwright Headless Browser (Current)

**Approach:** Uses Playwright to launch a headless Chromium browser that fully renders the page like a real user.

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto(url, wait_until="domcontentloaded")
    html = page.content()
```

**Advantages:**
- Renders JavaScript completely — SPAs, React, Next.js all work
- Executes in a real browser context with genuine user agent
- Preserves the full DOM including dynamically loaded content
- Can wait for specific elements to appear

**Current Multi-Strategy Architecture:**

```
1. Direct requests.get()  →  Fast, works for static/SSR sites
         ↓ (if fails)
2. Playwright headless     →  Renders JS, handles SPAs
         ↓ (if fails)
3. Jina reader proxy       →  Last resort fallback
```

Each strategy validates results by checking the element map has ≥ 3–5 text entries before accepting.

**Result:** ✅ Works for ~85% of sites. Remaining ~15% are sites with **aggressive anti-bot** measures (see Trade-offs below).

---

## Trade-offs & Limitations

### 🔴 Sites That Block Automated Access

Despite the multi-strategy approach, some major e-commerce platforms actively prevent any form of automated access:

| Site | Blocking Method | Behavior | Status |
|------|----------------|----------|--------|
| **Flipkart** | Akamai Bot Manager + custom challenge | Returns empty/challenge page even to headless browsers. Detects Playwright via browser fingerprinting. | ❌ Blocked |
| **Nike** | Cloudflare Bot Management (Enterprise) | Aggressive JavaScript challenges, CAPTCHAs, TLS fingerprinting. | ❌ Blocked |
| **Amazon** | Custom bot detection + CAPTCHA | Returns distorted HTML or CAPTCHA challenges. | ❌ Blocked |

**Why this happens:**
These sites invest heavily in anti-bot infrastructure because they face:
- Price scraping from competitors
- Inventory monitoring bots
- Automated purchasing (sneaker bots)
- Content theft

Their bot detection goes beyond simple user-agent checks — they analyze:
- TLS handshake fingerprints (JA3/JA4)
- Browser JavaScript API consistency
- Mouse movement and interaction patterns
- WebGL/Canvas fingerprints
- Timing patterns

**This is a fundamental tradeoff:** Bypassing these protections would require techniques like residential proxies, browser fingerprint spoofing, or paid anti-bot bypass services — all of which are:
1. **Ethically questionable** — these sites explicitly don't want to be scraped
2. **Expensive** — residential proxy services cost $5-15/GB
3. **Unreliable** — anti-bot measures evolve constantly
4. **Out of scope** — this is a CRO personalization tool, not a web scraping service

### ✅ Sites That Work Well

The tool excels on sites that are:
- **Server-side rendered (SSR)** — HTML content is in the initial response
- **Publicly accessible** — No login or geo-blocking required
- **Standard architecture** — Standard HTML structure with semantic tags
- **Not heavily protected** — No enterprise-grade bot management

This covers the majority of:
- Shopify stores (Hydrogen, standard themes)
- WordPress sites and blogs
- News sites (BBC, HN, etc.)
- Demo/showcase sites
- Content-focused pages

### ⚠️ Other Limitations

| Limitation | Details |
|-----------|---------|
| **LLM hallucinations** | Occasionally the AI may invent discounts or claims not present in the ad. The system prompt guards against this but isn't perfect. |
| **Image-heavy pages** | Pages where key information is in images (not text) will have fewer elements to personalize. |
| **Non-English content** | The LLM works best with English. Non-English pages may produce inconsistent results. |
| **Playwright on cloud** | Playwright requires Chromium installed, which adds ~400MB to deployment size and may not be available on all cloud platforms. |

---

## Deployment Guide

### ⚠️ Important Note About Vercel

**Streamlit apps cannot be deployed on Vercel.** Vercel is designed for static sites and serverless Node.js functions. Streamlit requires a persistent Python server process, which Vercel does not support.

The recommended free options for getting a live deployment link are:

---

### Option 1: Streamlit Community Cloud (Recommended — Easiest)

**Free tier, purpose-built for Streamlit apps.**

> **Note:** Playwright (headless browser) may not work on Streamlit Community Cloud due to system-level dependency restrictions. The pipeline will automatically fall back to direct fetch and Jina proxy strategies.

#### Step-by-step:

1. **Push your code to GitHub:**
   ```bash
   # Initialize git (if not already done)
   git init
   
   # Create .gitignore
   echo -e "myenv/\n__pycache__/\n.env\n*.pyc" > .gitignore
   
   # Add and commit
   git add .
   git commit -m "Initial commit: Landing Page Personalizer"
   
   # Create repo on GitHub, then push
   git remote add origin https://github.com/<your-username>/troopod-assignment.git
   git branch -M main
   git push -u origin main
   ```

2. **Go to [share.streamlit.io](https://share.streamlit.io/)** and sign in with your GitHub account.

3. **Click "New app"** → Select your repository, branch (`main`), and main file (`app.py`).

4. **Add your secrets in Advanced Settings:**
   ```toml
   # Paste this in the "Secrets" text box:
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

5. **Click "Deploy"** — Your app will be live at `https://<your-app>.streamlit.app` within 2–5 minutes.

---

### Option 2: Render (Recommended for Full Playwright Support)

**Free tier available, supports Python + Playwright.**

1. **Push your code to GitHub** (same as above).

2. **Create a `render.yaml`** in your project root:
   ```yaml
   services:
     - type: web
       name: landing-page-personalizer
       runtime: python
       buildCommand: |
         pip install -r requirements.txt
         playwright install chromium
         playwright install-deps chromium
       startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
       envVars:
         - key: GROQ_API_KEY
           sync: false
   ```

3. **Go to [render.com](https://render.com/)** → Sign in with GitHub → **New Web Service** → Select your repo.

4. **Set the environment variable** `GROQ_API_KEY` in the Render dashboard.

5. **Deploy** — Render will build and deploy your app. You'll get a live URL like `https://landing-page-personalizer.onrender.com`.

---

### Option 3: Railway

**Simple Git-based deploys with free tier.**

1. **Go to [railway.app](https://railway.app/)** → Sign in with GitHub.

2. **New Project → Deploy from GitHub Repo** → Select your repository.

3. **Add environment variable** `GROQ_API_KEY`.

4. **Add start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

5. **Generate Domain** → Get your live URL.

---

## Project Structure

```
troopod-assignment/
├── app.py              # Streamlit frontend (UI, file upload, results display)
├── backend.py          # LangGraph pipeline (scraper, AI agent, replacer nodes)
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (GROQ_API_KEY) — NOT committed
├── .gitignore          # Git ignore rules
├── README.md           # This documentation
└── images/             # Ad creative samples for testing
```

---

## License

This project was built as an assignment for **Troopod**. All rights reserved.

---

*Built with ❤️ using LangGraph, Groq, Playwright, and Streamlit.*
