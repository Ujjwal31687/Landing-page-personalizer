import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup, NavigableString
import base64
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Literal, TypedDict, Optional, Dict, Any
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

load_dotenv()

# ---------------------------------------------------------------------------
# PYDANTIC MODELS  (used internally after JSON parse – NOT for tool-use)
# ---------------------------------------------------------------------------

class SingleChange(BaseModel):
    tag_id: str = Field(description="Tag identifier like H1#1, P#3, A#5")
    original_text: str = Field(description="Exact original text from the element map")
    new_text: str = Field(description="Rewritten text to match the ad")
    element_type: str = Field(default="text", description="headline/subheadline/cta/body")


class PagePersonalization(BaseModel):
    changes: List[SingleChange] = Field(default_factory=list)
    ad_core_offer: str = Field(default="")
    target_audience: str = Field(default="")


# ---------------------------------------------------------------------------
# LANGGRAPH STATE
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    landing_page_url: str
    ad_image_base64: str
    ad_url: str                 # URL to ad creative (alternative to image upload)
    original_html: str          # Full untouched HTML
    raw_html: str               # backward compat alias
    clean_text: str
    element_map_text: str       # Human-readable TAG#N list for LLM
    element_lookup: dict        # {tag_id: original_text} for apply step
    soup_html: str              # HTML that was parsed (same as original_html)
    ai_results: Optional[PagePersonalization]
    modified_html: str
    applied_count: int
    error: Optional[str]


# ---------------------------------------------------------------------------
# LLM SETUP
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Conversion Rate Optimization (CRO) AI.

You will receive:
1. A TEXT LIST of visible elements on a landing page. Each line is formatted as:
   TAG#N: "visible text"
   For example: H1#1: "Welcome to our store"
2. An ad image.

Your task:
- Look at the ad image and understand its core offer, tone, and target audience.
- Pick 2-4 elements from the text list that should be rewritten to match the ad.
- Focus on headlines (H1, H2), subheadlines (H3, P), and CTA buttons (A, BUTTON).

RESPOND WITH ONLY A JSON BLOCK in this exact format (no other text before or after):

```json
{
  "ad_core_offer": "one sentence describing the ad's main offer",
  "target_audience": "who the ad targets",
  "changes": [
    {
      "tag_id": "H1#1",
      "original_text": "exact text from the list above",
      "new_text": "rewritten text matching the ad",
      "element_type": "headline"
    }
  ]
}
```

CRITICAL RULES:
- tag_id MUST be copied exactly from the text list (e.g. H1#1, P#3).
- original_text MUST be copied exactly from the text list.
- Do NOT invent discounts, prices, or claims not shown in the ad.
- Do NOT change brand names, logos, or navigation items.
- Keep new_text concise and punchy.
- Return ONLY the JSON block, nothing else."""


llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ---------------------------------------------------------------------------
# HELPER: Fetch page using Playwright (headless browser, renders JS)
# ---------------------------------------------------------------------------

def fetch_with_playwright(url: str, timeout_ms: int = 30000) -> str:
    """
    Use Playwright to open the URL in a headless Chromium browser,
    wait for the page to fully render (including JS), and return the HTML.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
            java_script_enabled=True,
        )
        page = context.new_page()

        try:
            # Navigate and wait for network to settle
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            # Wait for body to have meaningful content
            # Try to wait for at least some text to appear
            try:
                page.wait_for_function(
                    """() => {
                        const body = document.body;
                        if (!body) return false;
                        const text = body.innerText || '';
                        return text.trim().length > 100;
                    }""",
                    timeout=15000
                )
            except Exception:
                # Even if the wait times out, we'll proceed with whatever we have
                pass

            # Small extra wait to let lazy-loaded content appear
            page.wait_for_timeout(3000)

            html = page.content()
            return html

        finally:
            browser.close()


# ---------------------------------------------------------------------------
# HELPER: Screenshot a URL and return base64 image (for ad link input)
# ---------------------------------------------------------------------------

def screenshot_url_to_base64(url: str, timeout_ms: int = 30000) -> str:
    """
    Use Playwright to navigate to a URL, take a screenshot,
    and return it as a base64-encoded JPEG string.
    Used for converting ad link URLs into images for the vision LLM.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 720},
        )
        page = context.new_page()

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_timeout(3000)  # Let content render

            # Take screenshot as bytes
            screenshot_bytes = page.screenshot(type="jpeg", quality=85, full_page=False)
            return base64.b64encode(screenshot_bytes).decode('utf-8')

        finally:
            browser.close()


# ---------------------------------------------------------------------------
# HELPER: Build simple TAG#N element map
# ---------------------------------------------------------------------------

CONTENT_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'button',
                'span', 'li', 'td', 'th', 'label', 'figcaption', 'blockquote',
                'strong', 'em', 'b', 'i', 'small', 'div'}

SKIP_TAGS = {'script', 'style', 'meta', 'link', 'noscript', 'svg', 'path',
             'iframe', 'img', 'br', 'hr', 'input', 'textarea', 'select',
             'option', 'head', 'title'}


def build_element_map(html: str, max_entries: int = 80):
    """
    Parse HTML and return:
      - text_map: string of 'TAG#N: "text"' lines for the LLM
      - lookup: dict {tag_id: original_text} for the apply step
      - soup: the parsed BeautifulSoup object (reused later)
    """
    soup = BeautifulSoup(html, 'html.parser')
    lines = []
    lookup = {}       # tag_id -> original_text
    seen_texts = set()
    tag_counts = {}   # tag_name -> running counter

    for elem in soup.find_all(True):
        if elem.name in SKIP_TAGS:
            continue
        if elem.name not in CONTENT_TAGS:
            continue

        # Get DIRECT text only (not from nested children) to avoid duplicates
        direct_texts = []
        for child in elem.children:
            if isinstance(child, NavigableString):
                t = child.strip()
                if t:
                    direct_texts.append(t)
        direct_text = ' '.join(direct_texts)

        # Fallback: if no direct text but element has text, use get_text for leaves
        if not direct_text:
            child_tags = [c for c in elem.children if hasattr(c, 'name') and c.name]
            if not child_tags:
                direct_text = elem.get_text(strip=True)

        if not direct_text or len(direct_text) < 3:
            continue

        # Skip duplicates
        text_key = direct_text[:200].lower().strip()
        if text_key in seen_texts:
            continue
        seen_texts.add(text_key)

        # Skip large parent divs (their children will be captured)
        child_tags = [c for c in elem.children if hasattr(c, 'name') and c.name]
        if len(child_tags) > 3 and elem.name == 'div':
            continue

        # Assign TAG#N identifier
        tag_upper = elem.name.upper()
        tag_counts[tag_upper] = tag_counts.get(tag_upper, 0) + 1
        tag_id = f"{tag_upper}#{tag_counts[tag_upper]}"

        display_text = direct_text[:200]
        lines.append(f'{tag_id}: "{display_text}"')
        lookup[tag_id] = display_text

        if len(lines) >= max_entries:
            break

    text_map = '\n'.join(lines)
    return text_map, lookup, soup


# ---------------------------------------------------------------------------
# HELPER: Parse JSON from LLM response (robust)
# ---------------------------------------------------------------------------

def parse_llm_json(response_text: str) -> dict:
    """Extract and parse JSON from LLM response, handling code fences etc."""

    # Try to find JSON in code block first
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find raw JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0).strip()
        else:
            raise ValueError(f"No JSON found in LLM response: {response_text[:300]}")

    return json.loads(json_str)


# ---------------------------------------------------------------------------
# LANGGRAPH NODES
# ---------------------------------------------------------------------------

def fetch_page_node(state: AgentState) -> AgentState:
    """
    Fetch the landing page HTML with a multi-strategy approach:
      1. Quick direct fetch with requests (works for static HTML sites)
      2. Playwright headless browser (works for JS-rendered SPAs like Flipkart)
    """
    url = state["landing_page_url"]

    # Full browser-like headers
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    original_html = None
    used_strategy = ""

    # ── Strategy 1: Direct fetch (fast, works for static sites) ──
    try:
        print("🔄 Trying direct fetch...")
        resp = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        if resp.status_code == 200 and len(resp.text) > 500:
            # Quick check: does the HTML actually have visible text?
            test_map, _, _ = build_element_map(resp.text)
            entry_count = len(test_map.splitlines()) if test_map else 0
            if entry_count >= 5:
                original_html = resp.text
                used_strategy = "direct"
                print(f"✅ Direct fetch succeeded ({len(original_html)} chars, {entry_count} text elements)")
            else:
                print(f"⚠️ Direct fetch got HTML but only {entry_count} text elements — page likely needs JS rendering")
        else:
            print(f"⚠️ Direct fetch failed: status {resp.status_code}, body {len(resp.text)} chars")
    except Exception as e:
        print(f"⚠️ Direct fetch failed: {e}")

    # ── Strategy 2: Playwright headless browser (handles JS-rendered SPAs) ──
    if original_html is None:
        try:
            print("🔄 Using Playwright headless browser (renders JavaScript)...")
            playwright_html = fetch_with_playwright(url, timeout_ms=45000)
            if playwright_html and len(playwright_html) > 500:
                test_map, _, _ = build_element_map(playwright_html)
                entry_count = len(test_map.splitlines()) if test_map else 0
                if entry_count >= 3:
                    original_html = playwright_html
                    used_strategy = "playwright"
                    print(f"✅ Playwright fetch succeeded ({len(original_html)} chars, {entry_count} text elements)")
                else:
                    print(f"⚠️ Playwright fetched HTML but only {entry_count} text elements")
            else:
                print(f"⚠️ Playwright returned insufficient HTML ({len(playwright_html) if playwright_html else 0} chars)")
        except Exception as e:
            print(f"⚠️ Playwright fetch failed: {e}")

    # ── Strategy 3: Jina reader proxy as last resort ──
    if original_html is None:
        try:
            print("🔄 Falling back to Jina reader proxy...")
            jina_url = f"https://r.jina.ai/{url}"
            jina_headers = {
                "Accept": "text/html",
                "X-Return-Format": "html",
            }
            resp_jina = requests.get(jina_url, headers=jina_headers, timeout=90)
            resp_jina.raise_for_status()
            if len(resp_jina.text) > 500:
                test_map, _, _ = build_element_map(resp_jina.text)
                entry_count = len(test_map.splitlines()) if test_map else 0
                if entry_count >= 3:
                    original_html = resp_jina.text
                    used_strategy = "jina"
                    print(f"✅ Jina fetch succeeded ({len(original_html)} chars, {entry_count} text elements)")
                else:
                    print(f"⚠️ Jina returned HTML but only {entry_count} text elements")
        except Exception as e:
            print(f"⚠️ Jina fetch failed: {e}")

    if original_html is None:
        return {"error": f"Scraping failed: All strategies (direct, Playwright, Jina) failed to fetch meaningful content from {url}. The site may be aggressively blocking automated access."}

    # Inject <base> tag so relative URLs resolve correctly
    if "<head" in original_html.lower():
        original_html = re.sub(
            r'(<head[^>]*>)',
            rf'\1\n<base href="{url}">',
            original_html,
            count=1,
            flags=re.IGNORECASE
        )

    # ── Build the element map ──
    element_map_text, element_lookup, soup = build_element_map(original_html)
    entry_count = len(element_map_text.splitlines())
    print(f"📋 Built element map with {entry_count} entries (strategy: {used_strategy})")

    if entry_count == 0:
        return {"error": "Scraping failed: fetched HTML but found no visible text elements after all strategies."}

    return {
        "original_html": original_html,
        "raw_html": original_html,
        "soup_html": original_html,
        "element_map_text": element_map_text,
        "element_lookup": element_lookup,
        "clean_text": element_map_text,  # backward compat
    }


def analyze_ad_node(state: AgentState) -> AgentState:
    """Send ad image + element list to LLM, parse raw JSON response."""
    if state.get("error"):
        return state

    # --- Resolve ad image: either from base64 upload or by screenshotting the ad URL ---
    ad_base64 = state.get('ad_image_base64', '')

    if not ad_base64 and state.get('ad_url'):
        try:
            print(f"📸 Screenshotting ad URL: {state['ad_url']}")
            ad_base64 = screenshot_url_to_base64(state['ad_url'])
            print(f"✅ Ad screenshot captured ({len(ad_base64)} chars base64)")
        except Exception as e:
            return {"error": f"Failed to screenshot ad URL: {e}"}

    if not ad_base64:
        return {"error": "No ad creative provided — upload an image or enter an ad link URL."}

    image_data_uri = f"data:image/jpeg;base64,{ad_base64}"
    element_map_text = state.get("element_map_text", "")

    # Trim if very long (shouldn't be, since max 80 entries)
    trimmed = element_map_text[:8000]

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        f"Here is the TEXT LIST of visible elements on the landing page:\n\n"
                        f"{trimmed}\n\n"
                        f"Now analyze the ad image below and return your JSON response. "
                        f"Pick 2-4 elements from the list above to rewrite."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri}
                }
            ]
        )
    ]

    try:
        print("🤖 Calling LLM (raw JSON mode)...")
        response = llm.invoke(messages)
        response_text = response.content
        print(f"📝 LLM response length: {len(response_text)} chars")

        # Parse the JSON from the response
        parsed = parse_llm_json(response_text)

        # Build PagePersonalization from parsed JSON
        changes = []
        for c in parsed.get("changes", []):
            changes.append(SingleChange(
                tag_id=c.get("tag_id", ""),
                original_text=c.get("original_text", ""),
                new_text=c.get("new_text", ""),
                element_type=c.get("element_type", "text"),
            ))

        result = PagePersonalization(
            changes=changes,
            ad_core_offer=parsed.get("ad_core_offer", ""),
            target_audience=parsed.get("target_audience", ""),
        )

        print(f"✅ Parsed {len(result.changes)} changes from LLM")
        return {"ai_results": result}

    except json.JSONDecodeError as e:
        return {"error": f"AI returned invalid JSON: {e}. Raw response: {response_text[:500]}"}
    except Exception as e:
        return {"error": f"AI Analysis failed: {str(e)}"}


def apply_changes_node(state: AgentState) -> AgentState:
    """Apply text changes by finding original text in HTML and replacing it."""
    if state.get("error") or not state.get("ai_results"):
        return state

    html = state["original_html"]
    changes = state["ai_results"].changes

    soup = BeautifulSoup(html, 'html.parser')
    successful = 0

    # Only search within these tags — never touch structural/container tags
    ALLOWED_APPLY_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'a', 'button',
                          'span', 'li', 'td', 'th', 'label', 'figcaption', 'blockquote',
                          'strong', 'em', 'b', 'i', 'small'}
    # Note: 'div' is intentionally excluded from apply to avoid replacing huge containers

    for change in changes:
        if not change.original_text or not change.new_text:
            continue

        applied = False
        original_clean = re.sub(r'\s+', ' ', change.original_text.strip())

        # Collect all candidate elements (only from ALLOWED tags)
        candidates = []
        for elem in soup.find_all(list(ALLOWED_APPLY_TAGS)):
            elem_direct = _get_direct_text(elem)
            elem_full = elem.get_text(separator=' ', strip=True)
            candidates.append((elem, elem_direct, elem_full))

        # ── Pass 1: Try EXACT match on direct text ──
        for elem, elem_direct, elem_full in candidates:
            if not elem_direct:
                continue
            direct_clean = re.sub(r'\s+', ' ', elem_direct.strip())
            if direct_clean.lower() == original_clean.lower():
                _replace_text_in_element(elem, change.new_text)
                successful += 1
                applied = True
                print(f"  ✅ Applied [{change.tag_id}]: <{elem.name}> → \"{change.new_text[:60]}\"")
                break

        # ── Pass 2: Try EXACT match on full text (for leaf elements) ──
        if not applied:
            for elem, elem_direct, elem_full in candidates:
                if not elem_full:
                    continue
                # Only match leaf elements (no child tags) to avoid containers
                child_tags = [c for c in elem.children if hasattr(c, 'name') and c.name]
                if len(child_tags) > 1:
                    continue
                full_clean = re.sub(r'\s+', ' ', elem_full.strip())
                if full_clean.lower() == original_clean.lower():
                    _replace_text_in_element(elem, change.new_text)
                    successful += 1
                    applied = True
                    print(f"  ✅ Applied [{change.tag_id}]: <{elem.name}> → \"{change.new_text[:60]}\"")
                    break

        # ── Pass 3: Fuzzy match on direct text (handles minor differences) ──
        if not applied:
            for elem, elem_direct, elem_full in candidates:
                if not elem_direct:
                    continue
                direct_clean = re.sub(r'\s+', ' ', elem_direct.strip())
                if _fuzzy_match(original_clean, direct_clean):
                    _replace_text_in_element(elem, change.new_text)
                    successful += 1
                    applied = True
                    print(f"  ✅ Applied [{change.tag_id}]: <{elem.name}> → \"{change.new_text[:60]}\"")
                    break

        if not applied:
            print(f"  ❌ Could not find element for [{change.tag_id}]: \"{change.original_text[:60]}\"")

    print(f"\n📊 Applied {successful}/{len(changes)} changes total")
    return {"modified_html": str(soup), "applied_count": successful}


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _get_direct_text(elem) -> str:
    """Get only the direct text of an element, not from nested children."""
    parts = []
    for child in elem.children:
        if isinstance(child, NavigableString):
            t = child.strip()
            if t:
                parts.append(t)
    return ' '.join(parts)


def _fuzzy_match(needle: str, haystack: str) -> bool:
    """Check if needle text approximately matches haystack text."""
    n = re.sub(r'\s+', ' ', needle.strip().lower())
    h = re.sub(r'\s+', ' ', haystack.strip().lower())

    if not n or not h:
        return False

    # Exact or contained
    if n == h or n in h or h in n:
        return True

    # Word overlap for longer texts
    n_words = set(n.split())
    h_words = set(h.split())
    if len(n_words) >= 3:
        overlap = len(n_words & h_words) / len(n_words)
        return overlap >= 0.8

    return False


def _replace_text_in_element(elem, new_text: str):
    """Replace ALL text content of an element with new_text, preserving tag structure minimally."""
    child_tags = [c for c in elem.children if hasattr(c, 'name') and c.name]

    if not child_tags:
        # Leaf element — direct replacement
        elem.string = new_text
        return

    # Has child tags — replace all text nodes with new text in the first text node,
    # and clear subsequent text nodes
    first_replaced = False
    for child in list(elem.children):
        if isinstance(child, NavigableString) and child.strip():
            if not first_replaced:
                child.replace_with(new_text)
                first_replaced = True
            else:
                child.replace_with('')

    # If no text nodes were found (text is inside children), clear and set
    if not first_replaced:
        elem.clear()
        elem.string = new_text


# ---------------------------------------------------------------------------
# COMPILE LANGGRAPH
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("scraper", fetch_page_node)
workflow.add_node("ai_agent", analyze_ad_node)
workflow.add_node("replacer", apply_changes_node)

workflow.add_edge(START, "scraper")
workflow.add_edge("scraper", "ai_agent")
workflow.add_edge("ai_agent", "replacer")
workflow.add_edge("replacer", END)

app_workflow = workflow.compile()


# ---------------------------------------------------------------------------
# TEST BLOCK
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("--- Running LangGraph pipeline test ---")

    url = "https://en.wikipedia.org/wiki/Sneakers"
    ad_image_path = "/Users/ujjwalsharma/Desktop/troopod-assignment/nike-shoes-instagram-ad-design-template-37360e3838b6a6ae35ee728e0fdd97fe_screen.jpg"
    with open(ad_image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode("utf-8")

    initial_state = {
        "landing_page_url": url,
        "ad_image_base64": base64_img,
        "original_html": "",
        "raw_html": "",
        "clean_text": "",
        "element_map_text": "",
        "element_lookup": {},
        "soup_html": "",
        "ai_results": None,
        "modified_html": "",
        "applied_count": 0,
        "error": None
    }

    print("Invoking LangGraph App Workflow...")
    final_state = app_workflow.invoke(initial_state)

    if final_state.get("error"):
        print(f"\n❌ Workflow failed: {final_state['error']}")
    else:
        results = final_state['ai_results']
        print(f"\n🎯 Core Offer: {results.ad_core_offer}")
        print(f"👥 Target Audience: {results.target_audience}")
        print(f"\n📝 Changes ({len(results.changes)}):")
        for i, c in enumerate(results.changes, 1):
            print(f"  [{i}] {c.element_type.upper()}")
            print(f"      Tag ID:      {c.tag_id}")
            print(f"      Original:    '{c.original_text[:80]}'")
            print(f"      New text:    '{c.new_text[:80]}'")

        print(f"\n✅ Applied {final_state['applied_count']}/{len(results.changes)} changes to HTML")