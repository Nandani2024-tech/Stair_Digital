import re

raw = "The repo rate was held steady at 6.50%\n [Page 2 | Section: RBI continues with hawkish policy stance;\n held the repo rate steady at 6.50%]."

def strip_citation_tags(text: str) -> str:
    clean = re.sub(r'\[Page\s+\d+[^\]]*\]', '', text, flags=re.IGNORECASE)
    return re.sub(r'\s{2,}', ' ', clean).strip()

print("RAW:", repr(raw))
print("CLEAN:", repr(strip_citation_tags(raw)))
