def safe_divide(a, b):
    try:
        return a // b if isinstance(a, int) and isinstance(b, int) and a % b == 0 else a / b
    except Exception:
        return 0