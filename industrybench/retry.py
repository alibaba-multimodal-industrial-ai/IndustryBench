import time


def retry_call(fn, max_retries=3, retry_interval=5.0, label=""):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            result = fn()
            if result.get("success"):
                return result
            last_error = result.get("error", "unknown error")
        except Exception as e:
            last_error = str(e)
        if attempt < max_retries:
            wait = retry_interval * attempt
            if label:
                print(f"  [RETRY] {label} attempt {attempt} failed: {last_error}, retrying in {wait}s...")
            time.sleep(wait)
    return {"success": False, "error": f"Failed after {max_retries} attempts: {last_error}"}
