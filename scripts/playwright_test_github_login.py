from playwright.sync_api import sync_playwright
import os

storage_state_path = os.path.join(os.path.dirname(__file__), "auth.json")
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context(storage_state=storage_state_path)
    try:
        page = context.new_page()
        page.goto("https://github.com/")
        page.wait_for_load_state("load", timeout=10000)  # Wait for basic load, not networkidle
        if page.query_selector("text=Sign in"):
            print("NOT AUTHENTICATED")
        else:
            print("AUTHENTICATED")
        input("Press Enter to close the browser...")
    except Exception as e:
        print(f"Error during authentication check: {e}")
    finally:
        context.close()
        browser.close()