from playwright.sync_api import sync_playwright
import os

profile_dir = os.path.abspath("./.pw-profile")
with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=False
    )
    page = browser.new_page()
    page.goto("https://github.com/")
    page.wait_for_load_state("networkidle")
    if page.query_selector("text=Sign in"):
        print("NOT AUTHENTICATED")
    else:
        print("AUTHENTICATED")
    browser.close()