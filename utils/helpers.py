import os
import time
from functools import wraps

def load_api_key(key_name="ANTHROPIC_API_KEY"):
    key = os.environ.get(key_name)
    if not key:
        raise EnvironmentError(f"Missing API key: {key_name}\nSet it with: export {key_name}='your-key'")
    return key

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"[timer] {func.__name__} took {time.time()-start:.2f}s")
        return result
    return wrapper

def truncate_text(text, max_chars=200, suffix="..."):
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix

def format_confidence(score):
    pct = score * 100
    label = "High" if pct >= 80 else "Medium" if pct >= 50 else "Low"
    return f"{pct:.0f}% ({label})"

def load_dotenv():
    if not os.path.exists(".env"):
        return
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"\''))