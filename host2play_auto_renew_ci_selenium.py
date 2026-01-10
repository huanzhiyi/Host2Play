"""host2play_auto_renew_ci_selenium.py

GitHub Actions 专用 Selenium 续期脚本：
- 复用 host2play_auto_renew_local.py 的 reCAPTCHA + YOLO 解题逻辑（已在本地验证可用）
- 增加 Cloudflare 等待/判定（避免其实没放行就继续）
- 所有配置从环境变量读取（RENEW_URL / TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID）
- 只有当检测到“续期成功”文本时，才写出 host2play_renew_success.png 并退出 0
- 否则写出 host2play_error.png 并 exit(1)，避免 Actions 出现“假成功”

注意：GitHub Actions 的数据中心 IP 对 Cloudflare 风险很高，可能仍不稳定。
"""

from __future__ import annotations

import os
import sys
import re
import time
import json
import random
import shutil
import logging
from typing import Optional, Tuple, List

import requests
import numpy as np
import cv2
from PIL import Image

# YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# Selenium (沿用 local 的 seleniumwire.undetected_chromedriver)
try:
    import seleniumwire.undetected_chromedriver as webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
except Exception as e:
    print(f"❌ Selenium import failed: {e}")
    raise

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# ENV
RENEW_URL = os.environ.get('RENEW_URL')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
CI = os.environ.get('CI') == 'true'

MODEL_PATH = os.environ.get('YOLO_MODEL_PATH', 'model.onnx')
MODEL_DOWNLOAD_URLS = [
    os.environ.get('YOLO_MODEL_URL_1', 'https://media.githubusercontent.com/media/DannyLuna17/RecaptchaV2-IA-Solver/main/model.onnx'),
    os.environ.get('YOLO_MODEL_URL_2', 'https://github.com/DannyLuna17/RecaptchaV2-IA-Solver/raw/main/model.onnx'),
]

VERBOSE = True


def send_telegram(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Telegram send failed: {e}")


def human_like_delay(a: float = 0.4, b: float = 1.2) -> None:
    time.sleep(random.uniform(a, b))


def random_delay(mu: float = 0.8, sigma: float = 0.4) -> None:
    d = random.gauss(mu, sigma)
    d = max(0.3, min(2.0, d))
    time.sleep(d)


def download_yolo_model() -> bool:
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1_000_000:
        logger.info(f"✅ model exists: {MODEL_PATH} ({os.path.getsize(MODEL_PATH)/1024/1024:.1f} MB)")
        return True

    for idx, url in enumerate(MODEL_DOWNLOAD_URLS, 1):
        if not url:
            continue
        try:
            logger.info(f"📥 downloading model ({idx}/{len(MODEL_DOWNLOAD_URLS)}): {url}")
            r = requests.get(url, stream=True, timeout=300)
            r.raise_for_status()
            tmp = MODEL_PATH + '.tmp'
            with open(tmp, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(tmp) < 1_000_000:
                logger.warning(f"model too small: {os.path.getsize(tmp)} bytes")
                os.remove(tmp)
                continue
            os.replace(tmp, MODEL_PATH)
            logger.info(f"✅ model downloaded: {MODEL_PATH}")
            return True
        except Exception as e:
            logger.warning(f"model download failed: {e}")
            continue

    return False


def save_screenshot(driver, path: str) -> None:
    try:
        driver.save_screenshot(path)
        logger.info(f"📸 screenshot saved: {path}")
    except Exception as e:
        logger.warning(f"screenshot failed: {e}")


def is_cloudflare_challenge(driver) -> bool:
    """检测是否处于 Cloudflare challenge 页面"""
    try:
        title = (driver.title or '').lower()
        url = (driver.current_url or '').lower()
        if 'just a moment' in title or 'checking your browser' in title:
            return True
        if 'cf-chl' in url or '/cdn-cgi/' in url:
            return True
        # 常见 DOM
        dom_text = driver.page_source.lower()
        if 'cf-browser-verification' in dom_text or 'challenge-platform' in dom_text:
            return True
    except Exception:
        return False
    return False


def wait_cloudflare(driver, timeout_sec: int = 60) -> bool:
    """等待 Cloudflare 自动放行。返回 True 表示看起来已放行。"""
    start = time.time()
    last_shot = 0
    while time.time() - start < timeout_sec:
        if not is_cloudflare_challenge(driver):
            return True
        if time.time() - last_shot > 10:
            save_screenshot(driver, 'host2play_cloudflare_wait.png')
            last_shot = time.time()
        time.sleep(2)
    return not is_cloudflare_challenge(driver)


def get_all_captcha_img_urls(driver) -> List[str]:
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, '//div[@id="rc-imageselect-target"]//img')))
    return [img.get_attribute('src') for img in images]


def normalize_image(path: str, size: Tuple[int, int]) -> None:
    img = Image.open(path).convert('RGB')
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
        img.save(path)


def screenshot_grid(driver, grid_size: int) -> bool:
    """Selenium 截图验证码网格到 0.png，避免 payload URL 410 Gone"""
    try:
        grid = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'rc-imageselect-target'))
        )
        grid.screenshot('0.png')
        expected = (300, 300) if grid_size == 3 else (450, 450)
        normalize_image('0.png', expected)
        logger.info(f"✅ grid screenshot saved: 0.png -> {expected}")
        return True
    except Exception as e:
        logger.error(f"❌ grid screenshot failed: {e}")
        return False


def screenshot_tile(driver, index: int, grid_size: int) -> bool:
    try:
        cell = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{index}]'))
        )
        cell.screenshot(f'{index}.png')
        expected = (100, 100) if grid_size == 3 else (112, 112)
        normalize_image(f'{index}.png', expected)
        return True
    except Exception as e:
        logger.warning(f"tile screenshot failed {index}: {e}")
        return False


def paste_new_img_on_main_img(main: np.ndarray, new: np.ndarray, loc: int) -> None:
    paste = np.copy(main)
    row = (loc - 1) // 3
    col = (loc - 1) % 3
    start_row, end_row = row * 100, (row + 1) * 100
    start_col, end_col = col * 100, (col + 1) * 100
    paste[start_row:end_row, start_col:end_col] = new
    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite('0.png', paste)


def get_target_num(driver) -> int:
    """从 challenge 文本推断目标类型。与 local 类似。"""
    try:
        text = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//div[@id="rc-imageselect"]//strong'))).text
    except Exception:
        return 1000

    text_l = text.lower()
    mapping = {
        'bicycle': 1,
        'bus': 5,
        'boat': 8,
        'car': 2,
        'hydrant': 10,
        'motorcycle': 3,
        'traffic': 9,
    }
    for k, v in mapping.items():
        if k in text_l:
            return v
    return 1000


def dynamic_and_selection_solver(target_num: int, model) -> List[int]:
    try:
        if not os.path.exists('0.png'):
            return []
        image = np.asarray(Image.open('0.png'))
        result = model.predict(image, task='detect', verbose=False)
        boxes = result[0].boxes.data
        target_index = []
        for idx, cls in enumerate(result[0].boxes.cls):
            if int(cls) == int(target_num):
                target_index.append(idx)

        answers = []
        for i in target_index:
            x1, y1, x2, y2 = [int(v) for v in boxes[i][:4]]
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            row = yc // 100
            col = xc // 100
            answers.append(int(row * 3 + col + 1))
        return list(set(answers))
    except Exception as e:
        logger.error(f"✗ IA solve failed: {e}")
        return []


def get_occupied_cells(vertices: List[int]) -> List[int]:
    occupied_cells = set()
    rows, cols = zip(*[((v - 1) // 4, (v - 1) % 4) for v in vertices])
    for i in range(min(rows), max(rows) + 1):
        for j in range(min(cols), max(cols) + 1):
            occupied_cells.add(4 * i + j + 1)
    return sorted(list(occupied_cells))


def square_solver(target_num: int, model) -> List[int]:
    try:
        if not os.path.exists('0.png'):
            return []
        image = np.asarray(Image.open('0.png'))
        result = model.predict(image, task='detect', verbose=False)
        boxes = result[0].boxes.data
        target_index = []
        for idx, cls in enumerate(result[0].boxes.cls):
            if int(cls) == int(target_num):
                target_index.append(idx)

        answers: List[int] = []
        for i in target_index:
            x1, y1, x4, y4 = [int(v) for v in boxes[i][:4]]
            x2, y2 = x4, y1
            x3, y3 = x1, y4
            xys = [x1, y1, x2, y2, x3, y3, x4, y4]
            four_cells = []
            for j in range(4):
                x = xys[j * 2]
                y = xys[j * 2 + 1]
                if x < 112.5 and y < 112.5: four_cells.append(1)
                if 112.5 < x < 225 and y < 112.5: four_cells.append(2)
                if 225 < x < 337.5 and y < 112.5: four_cells.append(3)
                if 337.5 < x <= 450 and y < 112.5: four_cells.append(4)
                if x < 112.5 and 112.5 < y < 225: four_cells.append(5)
                if 112.5 < x < 225 and 112.5 < y < 225: four_cells.append(6)
                if 225 < x < 337.5 and 112.5 < y < 225: four_cells.append(7)
                if 337.5 < x <= 450 and 112.5 < y < 225: four_cells.append(8)
                if x < 112.5 and 225 < y < 337.5: four_cells.append(9)
                if 112.5 < x < 225 and 225 < y < 337.5: four_cells.append(10)
                if 225 < x < 337.5 and 225 < y < 337.5: four_cells.append(11)
                if 337.5 < x <= 450 and 225 < y < 337.5: four_cells.append(12)
                if x < 112.5 and 337.5 < y <= 450: four_cells.append(13)
                if 112.5 < x < 225 and 337.5 < y <= 450: four_cells.append(14)
                if 225 < x < 337.5 and 337.5 < y <= 450: four_cells.append(15)
                if 337.5 < x <= 450 and 337.5 < y <= 450: four_cells.append(16)

            for ans in get_occupied_cells(four_cells):
                answers.append(ans)

        return sorted(list(set(answers)))
    except Exception as e:
        logger.error(f"✗ square_solver failed: {e}")
        return []


def solve_recaptcha_ia(driver, model, max_attempts: int = 8) -> bool:
    """尽量贴近 local 的 solve_recaptcha_ia：如果检测到 410 Gone，可进一步改为 element 截图。"""
    try:
        driver.switch_to.default_content()
        iframe1 = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]')))
        driver.switch_to.frame(iframe1)
        checkbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//div[@class="recaptcha-checkbox-border"]')))
        human_like_delay(0.3, 0.8)
        checkbox.click()

        driver.switch_to.default_content()
        iframe2 = WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.XPATH, '//iframe[contains(@title, "challenge")]')))
        driver.switch_to.frame(iframe2)

        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            logger.info(f"  🧩 reCAPTCHA attempt {attempt}/{max_attempts}")

            reload_attempts = 0
            max_reload_attempts = 5
            captcha = None
            answers: List[int] = []

            while reload_attempts < max_reload_attempts:
                reload_attempts += 1
                try:
                    reload_btn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'recaptcha-reload-button')))
                    title_wrapper = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, 'rc-imageselect')))
                except Exception as e:
                    logger.info(f"    locate challenge elements failed: {e}")
                    time.sleep(2)
                    continue

                target_num = get_target_num(driver)
                title_text = title_wrapper.text or ''

                if target_num == 1000:
                    logger.info("    skip unsupported target, reload")
                    random_delay()
                    reload_btn.click()
                    time.sleep(2)
                    continue

                if 'squares' in title_text.lower():
                    logger.info("    detected 4x4 squares")
                    # 关键：不要下载 payload URL（会 410 Gone），直接截图网格
                    if not screenshot_grid(driver, grid_size=4):
                        reload_btn.click()
                        time.sleep(2)
                        continue

                    answers = square_solver(target_num, model)
                    if 1 <= len(answers) < 16:
                        captcha = 'squares'
                        break
                    logger.info("    square result not reasonable, reload")
                    try:
                        shutil.copy('0.png', f'failed_square_{attempt}_{reload_attempts}.png')
                    except Exception:
                        pass
                    reload_btn.click()
                    time.sleep(2)
                    continue

                if 'none' in title_text.lower():
                    logger.info("    detected 3x3 dynamic")
                    # 关键：不要下载 payload URL（会 410 Gone），直接截图网格
                    if not screenshot_grid(driver, grid_size=3):
                        reload_btn.click()
                        time.sleep(2)
                        continue
                    answers = dynamic_and_selection_solver(target_num, model)
                    if len(answers) >= 1:
                        captcha = 'dynamic'
                        break
                    logger.info(f"    3x3 insufficient: {len(answers)}, reload")
                    reload_btn.click()
                    time.sleep(2)
                    continue

                # selection 3x3
                logger.info("    detected 3x3 selection")
                # 关键：不要下载 payload URL（会 410 Gone），直接截图网格
                if not screenshot_grid(driver, grid_size=3):
                    reload_btn.click()
                    time.sleep(2)
                    continue

                answers = dynamic_and_selection_solver(target_num, model)
                if len(answers) >= 1:
                    captcha = 'selection'
                    break
                logger.info(f"    selection insufficient: {len(answers)}, reload")
                reload_btn.click()
                time.sleep(2)

            if not captcha:
                logger.info("    failed to decide captcha type in this attempt")
                continue

            # click answers
            if captcha == 'squares':
                for a in answers:
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{a}]'))).click()
                    random_delay(mu=0.6, sigma=0.3)

            elif captcha == 'selection':
                for a in answers:
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{a}]'))).click()
                    random_delay(mu=0.6, sigma=0.3)

            else:  # dynamic
                for a in answers:
                    WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{a}]'))).click()
                    random_delay(mu=0.6, sigma=0.3)

            human_like_delay(1.5, 2.5)
            verify = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, 'recaptcha-verify-button')))
            human_like_delay(0.8, 1.5)
            verify.click()

            human_like_delay(3, 4)

            # check solved
            try:
                driver.switch_to.default_content()
                # checkbox checked
                try:
                    iframe1 = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]')))
                    driver.switch_to.frame(iframe1)
                    WebDriverWait(driver, 3).until(EC.presence_of_element_located((By.XPATH, '//span[contains(@aria-checked, "true")]')))
                    driver.switch_to.default_content()
                    logger.info("✅ reCAPTCHA solved (checkbox checked)")
                    return True
                except Exception:
                    driver.switch_to.default_content()

                # challenge iframe hidden
                try:
                    challenge_iframe = driver.find_element(By.XPATH, '//iframe[contains(@title, "challenge")]')
                    if not challenge_iframe.is_displayed():
                        logger.info("✅ reCAPTCHA solved (challenge hidden)")
                        return True
                except Exception:
                    logger.info("✅ reCAPTCHA solved (challenge missing)")
                    return True

                # not solved, continue
                iframe2 = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//iframe[contains(@title, "challenge")]')))
                driver.switch_to.frame(iframe2)
                logger.info("    not solved, retry")
            except Exception as e:
                logger.info(f"    check solved error: {e}")
                # maybe solved
                return True

        return False
    except Exception as e:
        logger.error(f"solve_recaptcha_ia fatal: {e}")
        return False


def renew_success_criteria(driver) -> bool:
    """严格成功判定：必须看到 success/renewed 等文本。"""
    try:
        body_text = driver.find_element(By.TAG_NAME, 'body').text.lower()
        return ('success' in body_text) or ('renewed' in body_text)
    except Exception:
        return False


def main() -> int:
    if not RENEW_URL:
        logger.error('❌ RENEW_URL not set')
        return 2

    if not YOLO_AVAILABLE:
        logger.error('❌ ultralytics/YOLO not available')
        return 2

    if not download_yolo_model():
        logger.error('❌ model.onnx not available')
        return 2

    logger.info(f"✅ loading YOLO model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task='detect')

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--allow-insecure-localhost')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--lang=en-US')
    try:
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    except Exception:
        pass

    # CI 上建议 headless；如遇 CF 过不去，可改为 xvfb-run + 非 headless
    if CI:
        chrome_options.add_argument('--headless=new')
        chrome_options.add_argument('--window-size=1920,1080')

    seleniumwire_options = {
        'no_proxy': 'localhost,127.0.0.1',
        'disable_encoding': True,
        'verify_ssl': False,
        'suppress_connection_errors': True,
        'disable_capture': False,
    }

    driver = None
    try:
        logger.info('🚀 starting browser...')
        driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=seleniumwire_options)
        driver.scopes = ['.*google.com/recaptcha.*']

        logger.info('🌐 opening renew url...')
        driver.get(RENEW_URL)
        time.sleep(3)

        # Cloudflare wait
        if is_cloudflare_challenge(driver):
            logger.info('⏳ Cloudflare challenge detected, waiting...')
            ok = wait_cloudflare(driver, timeout_sec=90)
            if not ok:
                save_screenshot(driver, 'host2play_error_cloudflare.png')
                raise RuntimeError('Cloudflare challenge not passed')
            logger.info('✅ Cloudflare seems passed')

        # inject anti webdriver
        try:
            driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                if (!window.chrome) { window.chrome = {}; }
                if (!window.chrome.runtime) { window.chrome.runtime = {}; }
            """)
        except Exception:
            pass

        # wait recaptcha script
        logger.info('⏳ waiting grecaptcha...')
        for _ in range(25):
            try:
                ready = driver.execute_script("return typeof grecaptcha !== 'undefined' && typeof grecaptcha.render === 'function';")
                if ready:
                    break
            except Exception:
                pass
            time.sleep(1)

        # if recaptcha exists already, solve
        driver.switch_to.default_content()
        recaptcha_exists = False
        try:
            iframe = driver.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]')
            recaptcha_exists = iframe.is_displayed()
        except Exception:
            recaptcha_exists = False

        if recaptcha_exists:
            logger.info('🧩 solving existing reCAPTCHA on page...')
            ok = solve_recaptcha_ia(driver, model, max_attempts=8)
            if not ok:
                save_screenshot(driver, 'host2play_error_recaptcha_page.png')
                raise RuntimeError('reCAPTCHA solve failed (page)')

        # click Renew server
        driver.switch_to.default_content()
        time.sleep(2)
        logger.info("🔘 clicking 'Renew' button...")
        selectors = [
            "//button[contains(text(), 'Renew server')]",
            "//button[contains(text(), 'Renew')]",
            "//a[contains(text(), 'Renew server')]",
            "//a[contains(text(), 'Renew')]",
            "//input[@value='Renew server']",
            "//input[@value='Renew']",
            "//button[@type='submit']",
            "//input[@type='submit']",
        ]
        renew_button = None
        for sel in selectors:
            try:
                renew_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, sel)))
                break
            except Exception:
                continue
        if renew_button:
            driver.execute_script('arguments[0].click();', renew_button)
        else:
            save_screenshot(driver, 'host2play_error_no_renew_button.png')
            raise RuntimeError('Renew button not found')

        time.sleep(2)

        # modal recaptcha
        driver.switch_to.default_content()
        try:
            iframe = driver.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]')
            if iframe.is_displayed():
                logger.info('🧩 solving modal reCAPTCHA...')
                ok = solve_recaptcha_ia(driver, model, max_attempts=8)
                if not ok:
                    save_screenshot(driver, 'host2play_error_recaptcha_modal.png')
                    raise RuntimeError('reCAPTCHA solve failed (modal)')
        except Exception:
            pass

        # click modal Renew
        driver.switch_to.default_content()
        time.sleep(1.5)
        logger.info("🔘 clicking modal Renew button...")
        modal_selectors = [
            "//div[contains(@class, 'modal')]//button[contains(text(), 'Renew') and not(contains(text(), 'server'))]",
            "//div[contains(@class, 'dialog')]//button[contains(text(), 'Renew') and not(contains(text(), 'server'))]",
            "//div[contains(@class, 'popup')]//button[contains(text(), 'Renew') and not(contains(text(), 'server'))]",
            "//div[contains(@role, 'dialog')]//button[contains(text(), 'Renew') and not(contains(text(), 'server'))]",
            "//div[contains(@class, 'swal')]//button[contains(text(), 'Renew')]",
            "//div[contains(@class, 'swal')]//button[contains(text(), 'Confirm')]",
            "//div[contains(@class, 'modal')]//button[contains(text(), 'Confirm')]",
            "//div[contains(@class, 'modal')]//button[@type='submit']",
        ]
        modal_button = None
        for sel in modal_selectors:
            try:
                modal_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, sel)))
                break
            except Exception:
                continue
        if modal_button:
            driver.execute_script('arguments[0].click();', modal_button)
        else:
            # JS fallback like local
            js = """
                var modalSelectors = ['.modal', '.dialog', '.popup', '[role="dialog"]', '.swal2-container', '.swal-modal'];
                var modal = null;
                for (var i = 0; i < modalSelectors.length; i++) {
                    var modals = document.querySelectorAll(modalSelectors[i]);
                    for (var j = 0; j < modals.length; j++) {
                        if (modals[j].offsetParent !== null) { modal = modals[j]; break; }
                    }
                    if (modal) break;
                }
                if (modal) {
                    var buttons = modal.querySelectorAll('button, a, input[type="submit"]');
                    for (var i = 0; i < buttons.length; i++) {
                        var text = (buttons[i].textContent || buttons[i].value || '').toLowerCase();
                        if (text.includes('renew') && !text.includes('server')) { buttons[i].click(); return 'Clicked'; }
                        if (text.includes('confirm') || text.includes('yes') || text.includes('ok')) { buttons[i].click(); return 'Clicked confirm'; }
                    }
                }
                return 'No modal button';
            """
            res = driver.execute_script(js)
            logger.info(f"modal JS result: {res}")

        time.sleep(3)

        # strict success check
        if renew_success_criteria(driver):
            save_screenshot(driver, 'host2play_renew_success.png')
            send_telegram(f"✅ Host2Play renew success\n{time.strftime('%Y-%m-%d %H:%M:%S')}")
            return 0

        save_screenshot(driver, 'host2play_error.png')
        send_telegram(f"❌ Host2Play renew failed (no success text)\n{time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 1

    except Exception as e:
        logger.error(f"❌ fatal: {e}")
        if driver:
            save_screenshot(driver, 'host2play_error.png')
        send_telegram(f"❌ Host2Play renew failed\n{e}")
        return 1

    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == '__main__':
    raise SystemExit(main())
