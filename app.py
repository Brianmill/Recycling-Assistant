import base64
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import cv2
import numpy as np
import pdfplumber
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request, send_from_directory
from geopy.geocoders import Nominatim
from ultralytics import YOLO

from trash_detector import (
    TrackState,
    assign_tracks,
    detector_recycle_probability,
    get_recyclability,
    load_class_thresholds,
    load_recyclability_map,
    fuse_decision,
    verifier_recycle_probability,
)
import traceback
import logging


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "runs/detect/runs/detect/taco_basic_fixed/weights/best.pt"
VERIFIER_PATH = BASE_DIR / "yolov8n-cls.pt"
MAPPING_PATH = BASE_DIR / "recyclability_map.json"
THRESHOLD_PATH = BASE_DIR / "class_thresholds.json"

MATERIALS = ["paper", "plastic", "glass", "metal"]

ITEM_TO_MATERIAL: Dict[str, str] = {
    "paper": "paper",
    "mixed paper": "paper",
    "office paper": "paper",
    "books": "paper",
    "cardboard": "paper",
    "corrugated cardboard": "paper",
    "newspaper": "paper",
    "magazines": "paper",
    "junk mail": "paper",
    "milk jugs": "plastic",
    "beverage bottles": "plastic",
    "detergent bottles": "plastic",
    "plastic bottles": "plastic",
    "plastic tubs": "plastic",
    "plastic containers": "plastic",
    "metal cans": "metal",
    "aluminum cans": "metal",
    "steel cans": "metal",
    "tin cans": "metal",
    "steel (tin) cans": "metal",
    "aluminum foil": "metal",
    "aluminum plates": "metal",
    "beverage cartons": "paper",
    "cartons": "paper",
    "glass bottles": "glass",
    "glass jars": "glass",
    "glass bottles and jars": "glass",
    "glass": "glass",
    "metal containers": "metal",
    "plastic #1": "plastic",
    "plastic #2": "plastic",
    "plastic #5": "plastic",
}

STATE_ABBREVIATIONS: Dict[str, str] = {
    "new jersey": "nj",
    "new york": "ny",
    "pennsylvania": "pa",
    "california": "ca",
    "texas": "tx",
    "florida": "fl",
}


@dataclass
class GuidanceResult:
    location: str
    sources: List[str]
    allowed: Set[str]
    disallowed: Set[str]
    allowed_items: Set[str]
    disallowed_items: Set[str]
    notes: List[str]


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
model = YOLO(str(MODEL_PATH))
verifier_model = YOLO(str(VERIFIER_PATH)) if VERIFIER_PATH.exists() else None
mapping = load_recyclability_map(MAPPING_PATH)
class_thresholds = load_class_thresholds(THRESHOLD_PATH)
geolocator = Nominatim(user_agent="recycling-assistant")
guidance_cache: Dict[str, GuidanceResult] = {}
tracks: Dict[int, TrackState] = {}
frame_index = 0
TRACKER_IOU_THRESHOLD = 0.35
TRACKER_MAX_MISSED = 12
RECYCLE_ACCEPT = 0.62
TRASH_ACCEPT = 0.40
UNKNOWN_FRAMES = 4


@app.route("/images/<path:filename>")
def template_image(filename: str):
    return send_from_directory(BASE_DIR / "templates" / "images", filename)


def _mapping_key_variants(value: str) -> Set[str]:
    cleaned = value.strip().lower()
    if not cleaned:
        return set()

    compact = re.sub(r"\s+", " ", cleaned)
    variants = {
        compact,
        compact.replace(" ", "_"),
        compact.replace(" ", "-"),
    }

    if compact.startswith("plastic #"):
        variants.add("plastic")

    return {v for v in variants if v}


def apply_guidance_to_mapping(guidance: GuidanceResult, mapping_path: Path = MAPPING_PATH) -> Dict[str, object]:
    with mapping_path.open("r", encoding="utf-8") as f:
        current_mapping = json.load(f)

    normalized_mapping: Dict[str, str] = {
        str(k).strip().lower(): str(v).strip().lower() for k, v in current_mapping.items()
    }

    updates: Dict[str, str] = {}
    applied_keys: Set[str] = set()

    for entry in guidance.allowed:
        for key in _mapping_key_variants(entry):
            updates[key] = "recyclable"
            applied_keys.add(key)

    for entry in guidance.allowed_items:
        for key in _mapping_key_variants(entry):
            updates[key] = "recyclable"
            applied_keys.add(key)

    for entry in guidance.disallowed:
        for key in _mapping_key_variants(entry):
            updates[key] = "not_recyclable"
            applied_keys.add(key)

    for entry in guidance.disallowed_items:
        for key in _mapping_key_variants(entry):
            updates[key] = "not_recyclable"
            applied_keys.add(key)

    normalized_mapping.update(updates)
    if "default" not in normalized_mapping:
        normalized_mapping["default"] = "unknown"

    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(normalized_mapping, f, indent=2)

    mapping.clear()
    mapping.update(normalized_mapping)

    return {
        "updated_count": len(applied_keys),
        "updated_keys": sorted(applied_keys),
        "mapping_path": str(mapping_path),
    }


def _decode_data_url_image(image_data_url: str) -> np.ndarray:
    if "," not in image_data_url:
        raise ValueError("Invalid image payload")

    _, encoded = image_data_url.split(",", 1)
    decoded = base64.b64decode(encoded)
    image_np = np.frombuffer(decoded, dtype=np.uint8)
    frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image")

    return frame


def _build_location_context(location: str) -> Dict[str, str]:
    context: Dict[str, str] = {
        "location": location,
        "city": "",
        "county": "",
        "state": "",
        "country": "",
    }

    try:
        place = geolocator.geocode(location, language="en", addressdetails=True, timeout=8)
        if place and place.raw:
            address = place.raw.get("address", {})
            context["city"] = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("municipality")
                or ""
            )
            context["county"] = address.get("county") or ""
            context["state"] = address.get("state") or ""
            context["country"] = address.get("country") or ""
    except Exception:
        pass

    return context


def _search_recycling_pages(location: str, context: Dict[str, str], limit: int = 6) -> List[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    city = context.get("city", "")
    county = context.get("county", "")
    state = context.get("state", "")

    locality_hint = " ".join(part for part in [city, county, state] if part).strip() or location
    queries = [
        f"{locality_hint} recycling regulations",
        f"{locality_hint} recycling rules",
        f"{locality_hint} recycling ordinance",
        f"{locality_hint} what can be recycled",
    ]
    collected: List[str] = []

    def _normalize_search_href(href: str) -> Optional[str]:
        if "bing.com/ck/a" in href:
            parsed = urlparse(href)
            encoded_target = parse_qs(parsed.query).get("u", [""])[0]
            if encoded_target:
                try:
                    payload = encoded_target[2:] if encoded_target.startswith("a1") else encoded_target
                    payload = unquote(payload)
                    padding = "=" * (-len(payload) % 4)
                    decoded = base64.urlsafe_b64decode(payload + padding).decode("utf-8", errors="ignore")
                    if decoded.startswith("http"):
                        return decoded
                except Exception:
                    pass

        if href.startswith("http"):
            return href

        if "uddg=" in href:
            parsed = urlparse(href)
            encoded_target = parse_qs(parsed.query).get("uddg", [""])[0]
            if encoded_target:
                return unquote(encoded_target)

        return None

    for query in queries:
        logging.info("Search query: %s", query)
        # Prefer RSS results because they are stable and easier to parse than dynamic HTML.
        rss_response = requests.get(
            "https://www.bing.com/search",
            params={"q": query, "format": "rss"},
            timeout=12,
            headers=headers,
        )
        logging.info("Bing RSS request: %s", rss_response.url)
        rss_response.raise_for_status()
        try:
            root = ET.fromstring(rss_response.text)
        except ET.ParseError:
            root = None

        items = root.findall("./channel/item") if root is not None else []
        for item in items:
            href = (item.findtext("link") or "").strip()
            signal_text = " ".join(
                [
                    (item.findtext("title") or "").strip(),
                    (item.findtext("description") or "").strip(),
                ]
            )

            if not href or not href.startswith("http"):
                continue

            if href not in collected:
                collected.append(href)
    # Fallback: Bing HTML search when RSS yields insufficient results.
    if len(collected) < limit:
        for query in queries:
            logging.info("Bing HTML fallback query: %s", query)
            response = requests.get(
                "https://www.bing.com/search",
                params={"q": query},
                timeout=12,
                headers=headers,
            )
            logging.info("Bing HTML request: %s", response.url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            for anchor in soup.select("li.b_algo h2 a"):
                href = anchor.get("href")
                if not href:
                    continue

                normalized = _normalize_search_href(href)
                if not normalized:
                    continue

                if normalized not in collected:
                    collected.append(normalized)

    blocked_fragments = [
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "linkedin.com",
        "pinterest.com",
        "wikipedia.org",
        "amazon.com",
        "etsy.com",
        "ebay.com",
        "store",
        "shop",
        "shopping",
        "supply",
        "outdoor",
        "equipment",
        "gear",
    ]

    preferred_fragments = [
        ".gov",
        ".us",
        "township",
        "town",
        "borough",
        "city",
        "municipal",
        "county",
        "publicworks",
        "recycl",
        "solidwaste",
        "waste",
    ]

    scored: List[Tuple[int, str]] = []
    for url in collected:
        lowered = url.lower()
        if any(fragment in lowered for fragment in blocked_fragments):
            continue

        score = 0
        if "recycl" in lowered or "waste" in lowered or "trash" in lowered or "garbage" in lowered:
            score += 2
        if any(fragment in lowered for fragment in preferred_fragments):
            score += 2
        if urlparse(url).netloc.endswith(".gov") or urlparse(url).netloc.endswith(".us"):
            score += 3
        if any(term in lowered for term in ["collection", "guideline", "ordinance", "rules", "accepted", "program"]):
            score += 1

        if score <= 0:
            logging.info("Search result dropped as low relevance: %s", url)
            continue

        scored.append((score, url))

    scored.sort(key=lambda item: item[0], reverse=True)
    ranked = [url for _, url in scored][:limit]
    for url in ranked:
        logging.info("Search result selected for crawl: %s", url)
    return ranked


def _extract_recycling_signals(text: str) -> Tuple[Set[str], Set[str], List[str]]:
    lowered = re.sub(r"\s+", " ", text.lower())

    allowed: Set[str] = set()
    disallowed: Set[str] = set()
    notes: List[str] = []

    for material in MATERIALS:
        positive_patterns = [
            rf"{material}[^.\n]{{0,100}}(accepted|recyclable|can be recycled|place in recycling)",
            rf"(accepted|recyclable|can be recycled|place in recycling)[^.\n]{{0,100}}{material}",
        ]
        negative_patterns = [
            rf"{material}[^.\n]{{0,100}}(not accepted for recycling|do not recycle|not recyclable)",
            rf"(not accepted for recycling|do not recycle|not recyclable)[^.\n]{{0,100}}{material}",
        ]

        matched_positive = any(re.search(pattern, lowered) for pattern in positive_patterns)
        matched_negative = any(re.search(pattern, lowered) for pattern in negative_patterns)

        if matched_negative:
            disallowed.add(material)
        elif matched_positive:
            allowed.add(material)

    return allowed, disallowed, notes


def _extract_item_signals(text: str) -> Tuple[Set[str], Set[str], Set[str], Set[str], List[str]]:
    lowered = re.sub(r"\s+", " ", text.lower())

    allowed_materials: Set[str] = set()
    disallowed_materials: Set[str] = set()
    allowed_items: Set[str] = set()
    disallowed_items: Set[str] = set()
    notes: List[str] = []

    plastic_number_positive = re.search(r"(plastics?|resin).{0,40}(#?1|#?2|#?5)", lowered)
    if plastic_number_positive:
        allowed_materials.add("plastic")
        allowed_items.add("plastic #1/#2/#5")

    for item, material in ITEM_TO_MATERIAL.items():
        pos_patterns = [
            rf"{re.escape(item)}[^.\n]{{0,110}}(accepted|recyclable|can be recycled|place in recycling|yes)",
            rf"(accepted|recyclable|can be recycled|place in recycling|yes)[^.\n]{{0,110}}{re.escape(item)}",
        ]
        neg_patterns = [
            rf"{re.escape(item)}[^.\n]{{0,110}}(not accepted for recycling|do not recycle|not recyclable)",
            rf"(not accepted for recycling|do not recycle|not recyclable)[^.\n]{{0,110}}{re.escape(item)}",
        ]

        has_pos = any(re.search(pattern, lowered) for pattern in pos_patterns)
        has_neg = any(re.search(pattern, lowered) for pattern in neg_patterns)

        if has_neg:
            disallowed_items.add(item)
            disallowed_materials.add(material)
        elif has_pos:
            allowed_items.add(item)
            allowed_materials.add(material)

    # Capture list-style sections where accepted items are enumerated after a trigger phrase.
    # Also handle pages where materials are listed under a recycling section heading without
    # per-item "accepted" keywords (e.g., NJ "Mandated Materials to Recycle" pages).
    trigger_patterns = [
        r"required to recycle[^.\n:]*[:\-]?",
        r"accepted (materials|items|recyclables)[^.\n:]*[:\-]?",
        r"can be recycled[^.\n:]*[:\-]?",
        r"materials? mandated[^.\n:]*",
        r"mandated (materials?|to recycle|to be source)[^.\n:]*",
        r"source separated[^.\n:]*recycl[^.\n:]*",
        r"dual.stream recycl[a-z]*[^.\n:]*",
        r"curbside recycl[a-z]* collection[^.\n:]*",
        r"recycling (collection|program|materials?|items?)[^.\n:]*[:\-]?",
        r"comingled container[^.\n:]*",
    ]
    for pattern in trigger_patterns:
        for match in re.finditer(pattern, lowered):
            snippet = lowered[match.end(): match.end() + 800]
            for item, material in ITEM_TO_MATERIAL.items():
                if item in snippet:
                    allowed_items.add(item)
                    allowed_materials.add(material)

            for material in MATERIALS:
                if material in snippet:
                    allowed_materials.add(material)

            if "plastic" in snippet and re.search(r"#?1|#?2|#?5", snippet):
                allowed_items.add("plastic #1/#2/#5")
                allowed_materials.add("plastic")

    if allowed_items:
        # only keep the longest item listed to avoid overlaps (e.g., "plastic bottles" vs "plastic")
        notes.append("Accepted items: " + ", ".join(sorted(allowed_items)[:10]))
    if disallowed_items:
        notes.append("Restricted items: " + ", ".join(sorted(disallowed_items)[:10]))

    # Resolve conflicts in favor of accepted signals when both appear.
    disallowed_items -= allowed_items
    disallowed_materials -= allowed_materials

    return allowed_materials, disallowed_materials, allowed_items, disallowed_items, notes


def _split_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    raw_sentences = re.split(r"(?<=[.!?])\s+|\s*[\r\n]+\s*", cleaned)
    sentences = [sentence.strip(" \t-•;:") for sentence in raw_sentences]
    return [sentence for sentence in sentences if len(sentence) > 20]


def _score_regulation_sentence(sentence: str, location_tokens: Set[str]) -> Tuple[int, Set[str], Set[str]]:
    lowered = sentence.lower()
    score = 0
    allowed_hits: Set[str] = set()
    disallowed_hits: Set[str] = set()

    regulatory_terms = [
        "recycle",
        "recycling",
        "accepted",
        "accepts",
        "allowed",
        "prohibited",
        "prohibit",
        "not accepted",
        "do not recycle",
        "not recyclable",
        "ordinance",
        "guideline",
        "regulation",
        "policy",
        "rule",
        "ban",
        "curbside",
        "municipal",
        "county",
    ]
    if any(term in lowered for term in regulatory_terms):
        score += 3

    for token in location_tokens:
        if token and token in lowered:
            score += 1

    for material in MATERIALS:
        if material in lowered:
            score += 2
            if re.search(rf"{material}[^\n]{{0,80}}(accepted|allowed|recyclable|can be recycled)", lowered):
                allowed_hits.add(material)
                score += 2
            if re.search(rf"{material}[^\n]{{0,80}}(not accepted|do not recycle|not recyclable|prohibited)", lowered):
                disallowed_hits.add(material)
                score += 2

    for item, material in ITEM_TO_MATERIAL.items():
        if item in lowered:
            score += 2
            if re.search(rf"{re.escape(item)}[^\n]{{0,110}}(accepted|allowed|recyclable|can be recycled|yes)", lowered):
                allowed_hits.add(material)
                score += 2
            if re.search(rf"{re.escape(item)}[^\n]{{0,110}}(not accepted|do not recycle|not recyclable|prohibited)", lowered):
                disallowed_hits.add(material)
                score += 2

    if re.search(r"\b(must|shall|should|may not|must not|do not|not)\b", lowered):
        score += 1

    return score, allowed_hits, disallowed_hits


def _extract_regulations_with_nlp(text: str, context: Dict[str, str], max_sentences: int = 10) -> Tuple[Set[str], Set[str], Set[str], Set[str], List[str]]:
    sentences = _split_sentences(text)
    if not sentences:
        return set(), set(), set(), set(), []

    location_tokens = {
        token.strip().lower()
        for token in [context.get("city", ""), context.get("county", ""), context.get("state", ""), context.get("country", "")]
        if token and len(token.strip()) > 2
    }

    scored: List[Tuple[int, str, Set[str], Set[str]]] = []
    for sentence in sentences:
        score, allowed_hits, disallowed_hits = _score_regulation_sentence(sentence, location_tokens)
        if score >= 4:
            scored.append((score, sentence, allowed_hits, disallowed_hits))

    scored.sort(key=lambda item: item[0], reverse=True)

    allowed_materials: Set[str] = set()
    disallowed_materials: Set[str] = set()
    allowed_items: Set[str] = set()
    disallowed_items: Set[str] = set()
    notes: List[str] = []

    for _, sentence, allowed_hits, disallowed_hits in scored[:max_sentences]:
        allowed_materials.update(allowed_hits)
        disallowed_materials.update(disallowed_hits)

        lowered = sentence.lower()
        for item, material in ITEM_TO_MATERIAL.items():
            if item in lowered:
                if material in allowed_hits or re.search(rf"{re.escape(item)}[^\n]{{0,110}}(accepted|allowed|recyclable|can be recycled|yes)", lowered):
                    allowed_items.add(item)
                    allowed_materials.add(material)
                if material in disallowed_hits or re.search(rf"{re.escape(item)}[^\n]{{0,110}}(not accepted|do not recycle|not recyclable|prohibited)", lowered):
                    disallowed_items.add(item)
                    disallowed_materials.add(material)

        for material in MATERIALS:
            if material in lowered:
                if re.search(rf"{material}[^\n]{{0,100}}(accepted|allowed|recyclable|can be recycled)", lowered):
                    allowed_materials.add(material)
                if re.search(rf"{material}[^\n]{{0,100}}(not accepted|do not recycle|not recyclable|prohibited)", lowered):
                    disallowed_materials.add(material)

    disallowed_items -= allowed_items
    disallowed_materials -= allowed_materials

    return allowed_materials, disallowed_materials, allowed_items, disallowed_items, notes


def _fetch_page_text(url: str) -> Tuple[str, List[str]]:
    """Returns (page_text, list_of_absolute_anchor_hrefs)."""
    # If the caller passed something that isn't a URL, skip fetching.
    if not isinstance(url, str) or not url.lower().startswith("http"):
        logging.info("_fetch_page_text: non-URL input received, skipping: %s", repr(url))
        return "", []
    logging.info("Fetching page: %s", url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    # Check if URL points to a PDF
    is_pdf = url.lower().endswith(".pdf")

    try:
        response = requests.get(url, timeout=12, headers=headers)
        logging.info("Fetched %s with status %s", response.url, response.status_code)
        if response.status_code < 400:
            # Handle PDF extraction
            if is_pdf:
                try:
                    pdf_file = io.BytesIO(response.content)
                    pdf_text_parts: List[str] = []
                    with pdfplumber.open(pdf_file) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                pdf_text_parts.append(page_text)
                    pdf_text = " ".join(pdf_text_parts)
                    logging.info("Successfully extracted text from PDF: %s (length: %d)", url, len(pdf_text))
                    return pdf_text, []
                except Exception as pdf_err:
                    logging.warning("Failed to extract PDF text from %s: %s", url, pdf_err)
                    return "", []
            else:
                # Handle HTML extraction
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract anchor hrefs before stripping tags
                links: List[str] = []
                for a in soup.select("a[href]"):
                    href = a.get("href", "")
                    if href and not href.startswith(("#", "mailto:", "javascript:")):
                        links.append(urljoin(url, href))

                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()

                return " ".join(soup.stripped_strings), links
    except Exception as exc:
        logging.warning("Failed to fetch %s: %s", url, exc)

    parsed = urlparse(url)
    # Only attempt the jina.ai mirror when the parsed netloc looks valid (contains a dot)
    # and it's not a PDF (PDFs should not be mirrored)
    if not is_pdf and parsed.netloc and "." in parsed.netloc:
        mirror_url = f"https://r.jina.ai/http://{parsed.netloc}{parsed.path}"
        if parsed.query:
            mirror_url = f"{mirror_url}?{parsed.query}"

        try:
            logging.info("Trying jina mirror: %s", mirror_url)
            mirror_response = requests.get(mirror_url, timeout=15, headers=headers)
            logging.info("Mirror response: %s %s", mirror_response.url, mirror_response.status_code)
            mirror_response.raise_for_status()
            return mirror_response.text, []
        except Exception as mirror_err:
            logging.warning("Jina mirror fallback failed for %s: %s", url, mirror_err)

    logging.info("_fetch_page_text: could not fetch or parse content from: %s", url)
    return "", []

def get_location_guidance(location: str, source_url: Optional[str] = None) -> GuidanceResult:
    cache_key = f"{location.strip().lower()}::{(source_url or '').strip().lower()}"
    if cache_key in guidance_cache:
        return guidance_cache[cache_key]

    allowed: Set[str] = set()
    disallowed: Set[str] = set()
    allowed_items: Set[str] = set()
    disallowed_items: Set[str] = set()
    notes: List[str] = []
    sources: List[str] = []
    context = _build_location_context(location)

    urls_to_scan: List[str] = []
    if source_url:
        urls_to_scan = [source_url]
    else:
        try:
            urls_to_scan = _search_recycling_pages(location=location, context=context)
        except Exception as exc:
            notes.append(f"Search step failed: {exc}")

    if not source_url:
        merged: List[str] = []
        for candidate in urls_to_scan:
            if candidate not in merged:
                merged.append(candidate)
        urls_to_scan = merged
    logging.info("Crawl seed URLs for %s: %s", location, urls_to_scan)

    queue: List[str] = urls_to_scan[:20]
    seen: Set[str] = set()
    processed_count = 0
    follow_pattern = re.compile(r"recycl|solid\s*waste|hazardous|mcia|public\s*works", re.IGNORECASE)

    while queue and processed_count < 10:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        logging.info("Crawl visit #%d: %s", processed_count + 1, url)

        # Decode Granicus/CivicPlus external splash redirects (e.g. mercercounty.org)
        # instead of wasting a processing slot on a "you are leaving" interstitial page.
        if "____isexternal=true" in url or "?splash=" in url or "&splash=" in url:
            parsed_splash = urlparse(url)
            splash_target = parse_qs(parsed_splash.query).get("splash", [""])[0]
            if splash_target:
                real_url = unquote(splash_target)
                if real_url.startswith("http") and real_url not in seen and real_url not in queue:
                    queue.insert(0, real_url)
                # Also enqueue the domain root so that if the specific target 404s
                # (e.g. old Granicus links pointing to mcianj.org/content/119/),
                # we still crawl the authority site home page and follow recycling links.
                parsed_real = urlparse(real_url)
                if parsed_real.netloc:
                    root_url = f"{parsed_real.scheme}://{parsed_real.netloc}"
                    if root_url not in seen and root_url not in queue:
                        queue.append(root_url)
            continue

        processed_count += 1

        try:
            text, page_links = _fetch_page_text(url)
            page_allowed, page_disallowed, page_notes = _extract_recycling_signals(text)
            (
                item_allowed_materials,
                item_disallowed_materials,
                page_allowed_items,
                page_disallowed_items,
                item_notes,
            ) = _extract_item_signals(text)
            (
                nlp_allowed_materials,
                nlp_disallowed_materials,
                nlp_allowed_items,
                nlp_disallowed_items,
                nlp_notes,
            ) = _extract_regulations_with_nlp(text, context=context)

            allowed.update(page_allowed)
            allowed.update(item_allowed_materials)
            allowed.update(nlp_allowed_materials)
            disallowed.update(page_disallowed)
            disallowed.update(item_disallowed_materials)
            disallowed.update(nlp_disallowed_materials)
            allowed_items.update(page_allowed_items)
            allowed_items.update(nlp_allowed_items)
            disallowed_items.update(page_disallowed_items)
            disallowed_items.update(nlp_disallowed_items)
            notes.extend(page_notes)
            notes.extend(item_notes)
            notes.extend(nlp_notes)
            sources.append(url)

            # Follow recycling-relevant anchor links found in the page HTML.
            for link_url in page_links:
                if link_url in seen or link_url in queue:
                    continue
                is_mcia_sec_link = "mcianj.org/index.asp?SEC=" in link_url
                if follow_pattern.search(link_url) or is_mcia_sec_link:
                    logging.info("Queueing discovered link: %s", link_url)
                    queue.append(link_url)

            # Also follow absolute URLs embedded in page text (e.g., jina-mirrored pages).
            for discovered in re.findall(r"https?://[^\s\)\]]+", text):
                candidate = discovered.strip().rstrip(".,")
                if candidate in seen or candidate in queue:
                    continue
                if follow_pattern.search(candidate):
                    logging.info("Queueing text-discovered link: %s", candidate)
                    queue.append(candidate)
        except Exception as exc:
            notes.append(f"Could not process {url}: {exc}")

    if not sources and source_url:
        notes.append("Could not read the provided guidance URL.")
    if not sources and not source_url:
        notes.append("No guidance pages were fully readable from automatic search; try adding an official city or county recycling URL in the optional field.")
        try:
            fallback_url = "https://www.epa.gov/recycle"
            text, page_links = _fetch_page_text(fallback_url)
            if text:
                page_allowed, page_disallowed, page_notes = _extract_recycling_signals(text)
                (
                    item_allowed_materials,
                    item_disallowed_materials,
                    page_allowed_items,
                    page_disallowed_items,
                    item_notes,
                ) = _extract_item_signals(text)
                (
                    nlp_allowed_materials,
                    nlp_disallowed_materials,
                    nlp_allowed_items,
                    nlp_disallowed_items,
                    nlp_notes,
                ) = _extract_regulations_with_nlp(text, context=context)

                allowed.update(page_allowed)
                allowed.update(item_allowed_materials)
                allowed.update(nlp_allowed_materials)
                disallowed.update(page_disallowed)
                disallowed.update(item_disallowed_materials)
                disallowed.update(nlp_disallowed_materials)
                allowed_items.update(page_allowed_items)
                allowed_items.update(nlp_allowed_items)
                disallowed_items.update(page_disallowed_items)
                disallowed_items.update(nlp_disallowed_items)
                notes.extend(page_notes)
                notes.extend(item_notes)
                notes.extend(nlp_notes)
                sources.append(fallback_url)
                for link_url in page_links:
                    if link_url not in sources and link_url not in queue:
                        queue.append(link_url)
        except Exception as exc:
            notes.append(f"General recycling guidance fallback failed: {exc}")

    # Resolve cross-page conflicts in favor of accepted signals.
    disallowed_items -= allowed_items
    disallowed -= allowed

    # Keep exactly one consolidated accepted-items note.
    notes = [note for note in notes if not note.startswith("Accepted items:")]
    if allowed_items:
        notes.append("Accepted items: " + ", ".join(sorted(allowed_items)))

    # Add a single summary note if nothing could be extracted
    if not allowed and not allowed_items and not disallowed and not disallowed_items:
        notes.append("No clear material-specific rules were extracted from fetched pages.")

    # Deduplicate notes while preserving order
    seen_notes: Set[str] = set()
    unique_notes: List[str] = []
    for note in notes:
        if note not in seen_notes:
            seen_notes.add(note)
            unique_notes.append(note)

    result = GuidanceResult(
        location=location,
        sources=sources,
        allowed=allowed,
        disallowed=disallowed,
        allowed_items=allowed_items,
        disallowed_items=disallowed_items,
        notes=unique_notes[:12],
    )
    guidance_cache[cache_key] = result
    return result


def _apply_local_override(label: str, base_status: str, guidance: Optional[GuidanceResult]) -> Tuple[str, str]:
    material = label.strip().lower()

    if not guidance:
        return base_status, "No local rules loaded."

    if material in guidance.disallowed:
        return "not_recyclable", f"Local rule says {material} is not accepted."

    if material in guidance.allowed and base_status in {"unknown", "not_recyclable"}:
        return "recyclable", f"Local rule says {material} is accepted."

    return base_status, "No local override applied."


def _resolve_location_text(lat: Optional[float], lon: Optional[float], location: Optional[str]) -> str:
    if location and location.strip():
        return location.strip()

    if lat is None or lon is None:
        return ""

    try:
        place = geolocator.reverse((lat, lon), language="en", exactly_one=True, timeout=8)
        if place is None:
            return ""
        address = place.raw.get("address", {})
        city = address.get("city") or address.get("town") or address.get("village") or ""
        state = address.get("state") or ""
        country = address.get("country") or ""
        parts = [part for part in [city, state, country] if part]
        return ", ".join(parts)
    except Exception:
        return ""


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/resolve-location")
def resolve_location():
    payload = request.get_json(force=True)
    lat = payload.get("lat")
    lon = payload.get("lon")
    location = payload.get("location")

    location_text = _resolve_location_text(lat=lat, lon=lon, location=location)
    if not location_text:
        return jsonify({"ok": False, "error": "Could not resolve location."}), 400

    return jsonify({"ok": True, "location": location_text})


@app.post("/api/guidelines")
def guidelines():
    payload = request.get_json(force=True)
    location = (payload.get("location") or "").strip()
    source_url = (payload.get("source_url") or "").strip() or None

    if not location:
        return jsonify({"ok": False, "error": "Location is required."}), 400

    guidance = get_location_guidance(location=location, source_url=source_url)
    mapping_update = apply_guidance_to_mapping(guidance)
    return jsonify(
        {
            "ok": True,
            "guidance": {
                "location": guidance.location,
                "sources": guidance.sources,
                "allowed": sorted(guidance.allowed),
                "disallowed": sorted(guidance.disallowed),
                "allowed_items": sorted(guidance.allowed_items),
                "disallowed_items": sorted(guidance.disallowed_items),
                "notes": guidance.notes,
            },
            "mapping_update": mapping_update,
        }
    )


@app.post("/api/detect-frame")
def detect_frame():
    global frame_index
    frame_index += 1
    payload = request.get_json(force=True)
    logging.info("/api/detect-frame called, payload keys: %s", list(payload.keys()) if isinstance(payload, dict) else str(type(payload)))
    image_data_url = payload.get("image")
    location = (payload.get("location") or "").strip()
    source_url = (payload.get("source_url") or "").strip() or None

    if not image_data_url:
        return jsonify({"ok": False, "error": "Image is required."}), 400

    try:
        frame = _decode_data_url_image(image_data_url)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    # Debug logging to help diagnose client/server issues
    try:
        logging.info(f"/api/detect-frame called, frame shape: {frame.shape}")
    except Exception:
        logging.info("/api/detect-frame called, could not read frame shape")

    guidance: Optional[GuidanceResult] = None
    if location:
        guidance = get_location_guidance(location=location, source_url=source_url)

    try:
        results = model.predict(source=frame, conf=0.3, verbose=False)
    except Exception as exc:
        tb = traceback.format_exc()
        logging.error("Model predict failed: %s", tb)
        return jsonify({"ok": False, "error": "Model prediction failed", "detail": str(exc)}), 500

    # Detailed detection logging for debugging
    try:
        if not results:
            logging.info("Model.predict returned no results")
        else:
            logging.info("Model.predict returned %d result(s)", len(results))
            result = results[0]
            boxes_count = len(result.boxes) if hasattr(result, 'boxes') else 0
            logging.info("Result[0] contains %d boxes", boxes_count)
            for i, box in enumerate(getattr(result, 'boxes', [])):
                try:
                    cls_id = int(box.cls.item())
                    det_conf = float(box.conf.item())
                    coords = [float(v) for v in box.xyxy[0].tolist()]
                    label = str(result.names.get(cls_id, str(cls_id)))
                    threshold = class_thresholds.get(label.strip().lower(), 0.35)
                    logging.info(
                        "Box %d: cls=%s id=%d conf=%.3f coords=%s threshold=%.3f",
                        i,
                        label,
                        cls_id,
                        det_conf,
                        coords,
                        threshold,
                    )
                    if det_conf < threshold:
                        logging.info("Box %d filtered out by threshold (%.3f < %.3f)", i, det_conf, threshold)
                except Exception:
                    logging.exception("Error logging box %d details", i)
    except Exception:
        logging.exception("Error while logging detection results")
    detections: List[Dict[str, object]] = []
    detection_rows: List[Dict[str, object]] = []

    if results:
        result = results[0]
        for box in result.boxes:
            cls_id = int(box.cls.item())
            det_conf = float(box.conf.item())
            label = str(result.names.get(cls_id, str(cls_id)))
            threshold = class_thresholds.get(label.strip().lower(), 0.35)

            if det_conf < threshold:
                continue

            detection_rows.append(
                {
                    "label": label,
                    "conf": det_conf,
                    "box": [float(v) for v in box.xyxy[0].tolist()],
                }
            )

    assign_tracks(
        tracks=tracks,
        detections=detection_rows,
        frame_index=frame_index,
        iou_threshold=TRACKER_IOU_THRESHOLD,
        max_missed_frames=TRACKER_MAX_MISSED,
    )

    for det in detection_rows:
        label = str(det["label"])
        det_conf = float(det["conf"])
        xyxy = det["box"]
        track_id = int(det["track_id"])
        track = tracks[track_id]

        base_status = get_recyclability(label, mapping)
        detector_prob = detector_recycle_probability(base_status, det_conf)
        verifier_prob = verifier_recycle_probability(verifier_model, frame, xyxy)
        track_status, track_prob = fuse_decision(
            track=track,
            detector_prob=detector_prob,
            verifier_prob=verifier_prob,
            recyclable_accept=RECYCLE_ACCEPT,
            trash_accept=TRASH_ACCEPT,
            unknown_frames=UNKNOWN_FRAMES,
        )
        final_status, reason = _apply_local_override(label, track_status, guidance)

        detections.append(
            {
                "label": label,
                "confidence": round(det_conf, 3),
                "box": [round(float(v), 1) for v in xyxy],
                "track_id": track_id,
                "detector_recycle_score": round(detector_prob, 3),
                "verifier_recycle_score": round(verifier_prob, 3),
                "track_status": track_status,
                "track_recycle_score": round(track_prob, 3),
                "base_status": base_status,
                "final_status": final_status,
                "local_reason": reason,
            }
        )

    summary = "unknown"
    if any(d["final_status"] == "not_recyclable" for d in detections):
        summary = "trash"
    elif detections and all(d["final_status"] == "recyclable" for d in detections):
        summary = "recycle"

    return jsonify(
        {
            "ok": True,
            "summary": summary,
            "detections": detections,
            "guidance": None
            if guidance is None
            else {
                "location": guidance.location,
                "sources": guidance.sources,
                "allowed": sorted(guidance.allowed),
                "disallowed": sorted(guidance.disallowed),
                "allowed_items": sorted(guidance.allowed_items),
                "disallowed_items": sorted(guidance.disallowed_items),
                "notes": guidance.notes,
            },
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
