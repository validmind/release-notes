from playwright.sync_api import sync_playwright
import os

profile_dir = os.path.abspath("./.pw-profile")
with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        user_data_dir=profile_dir,
        headless=False
    )
    page = browser.new_page()
    page.goto("https://github.com/login")
    input("Log in to GitHub in the browser window, then press Enter here...")
    page.goto("https://github.com/user-attachments/assets/6f2ae03c-cb9d-4ee1-bde7-016cde360fe5")
    page.screenshot(path="debug.png")
    input("Check debug.png, then press Enter to close...")
    browser.close()