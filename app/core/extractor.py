import re

def extract_intelligence(text: str):
    upi_pattern = r"[a-zA-Z0-9.\-_]{2,}@[a-zA-Z]{2,}"
    bank_pattern = r"\b\d{9,18}\b"
    url_pattern = r"https?://[^\s]+"

    upi_ids = re.findall(upi_pattern, text)
    bank_accounts = re.findall(bank_pattern, text)
    links = re.findall(url_pattern, text)

    return {
        "upi_id": upi_ids[0] if upi_ids else None,
        "bank_account": bank_accounts[0] if bank_accounts else None,
        "phishing_links": links
    }
