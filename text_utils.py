import re
import unicodedata

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    # Unicode normalize and lowercase
    s = unicodedata.normalize("NFKC", s).lower()
    # Replace fancy punctuation with plain ASCII
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("–", "-")
    # Remove URLs and emails
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    # Keep alphanumerics, whitespace, and a small set of punctuation
    s = re.sub(r"[^a-z0-9\s,\.!?\-\'\"]", " ", s)
    # Collapse repeated whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s