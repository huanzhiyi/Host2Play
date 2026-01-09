"""
Host2Play 自动续期脚本 - 使用 RecaptchaV2-IA-Solver
访问续期页面，点击 Renew server，通过 reCAPTCHA 验证后点击窗体中的 Renew

基于成功的 host2play_with_ia_solver.py
"""
import os
import sys
import shutil
from time import sleep
import re
import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import seleniumwire.undetected_chromedriver as webdriver


# 配置
MODEL_PATH = os.environ.get('MODEL_PATH', 'model.onnx')  # 模型文件在当前目录
VERBOSE = os.environ.get('VERBOSE', 'true').lower() == 'true'
RENEW_URL = os.environ.get('RENEW_URL')  # 必须通过环境变量提供
HEADLESS = os.environ.get('HEADLESS', 'false').lower() == 'true'  # GitHub Actions 需要 headless
SCREENSHOT_PATH = os.environ.get('SCREENSHOT_PATH', 'host2play_renew_success.png')

# Telegram 配置
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
ENABLE_TELEGRAM = TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID


def random_delay(mu=0.3, sigma=0.1):
    """随机延迟模拟人类行为"""
    delay = np.random.normal(mu, sigma)
    delay = max(0.1, delay)
    sleep(delay)


def human_like_delay(min_time=0.5, max_time=1.5):
    """更自然的随机延迟"""
    sleep(np.random.uniform(min_time, max_time))


def send_telegram_message(message, parse_mode='HTML'):
    """发送 Telegram 通知"""
    if not ENABLE_TELEGRAM:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': parse_mode
        }
        response = requests.post(url, data=data, timeout=10)
        if response.status_code == 200:
            print("✓ Telegram 通知发送成功")
            return True
        else:
            print(f"⚠ Telegram 通知发送失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠ Telegram 通知发送异常: {e}")
        return False


def send_telegram_photo(photo_path, caption=''):
    """发送 Telegram 图片"""
    if not ENABLE_TELEGRAM:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        
        with open(photo_path, 'rb') as photo:
            files = {'photo': photo}
            data = {
                'chat_id': TELEGRAM_CHAT_ID,
                'caption': caption,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=data, files=files, timeout=30)
        
        if response.status_code == 200:
            print("✓ Telegram 截图发送成功")
            return True
        else:
            print(f"⚠ Telegram 截图发送失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠ Telegram 截图发送异常: {e}")
        return False


def download_img(name, url):
    """下载图片"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        with open(f'{name}.png', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        return True
    except Exception as e:
        print(f"✗ 图片下载失败 {name}: {e}")
        return False


def get_target_num(driver):
    """获取验证目标类别编号"""
    target_mappings = {
        "bicycle": 1,
        "bus": 5,
        "boat": 8,
        "car": 2,
        "hydrant": 10,
        "motorcycle": 3,
        "traffic": 9
    }
    
    target = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//div[@id="rc-imageselect"]//strong')))
    
    for term, value in target_mappings.items():
        if re.search(term, target.text):
            return value
    
    return 1000


def dynamic_and_selection_solver(target_num, verbose, model):
    """解决 3x3 网格验证（动态和一次性选择）"""
    try:
        if not os.path.exists("0.png"):
            if verbose: print("  ✗ 图片文件不存在: 0.png")
            return []
        
        image = Image.open("0.png")
        image = np.asarray(image)
        result = model.predict(image, task="detect", verbose=False, conf=0.25)
        
        # 获取目标索引
        target_index = []
        for i, num in enumerate(result[0].boxes.cls):
            if num == target_num:
                target_index.append(i)
        
        if verbose and len(target_index) > 0:
            print(f"    检测到 {len(target_index)} 个目标物体")
        
        # 计算答案位置
        answers = []
        boxes = result[0].boxes.data
        for i in target_index:
            target_box = boxes[i]
            x1, y1 = int(target_box[0]), int(target_box[1])
            x2, y2 = int(target_box[2]), int(target_box[3])
            
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            
            row = yc // 100
            col = xc // 100
            answer = int(row * 3 + col + 1)
            answers.append(answer)
        
        return list(set(answers))
    except Exception as e:
        if verbose: print(f"  ✗ 图片识别失败: {e}")
        return []


def get_occupied_cells(vertices):
    """获取被占用的单元格（4x4 网格）"""
    occupied_cells = set()
    rows, cols = zip(*[((v-1)//4, (v-1) % 4) for v in vertices])
    
    for i in range(min(rows), max(rows)+1):
        for j in range(min(cols), max(cols)+1):
            occupied_cells.add(4*i + j + 1)
    
    return sorted(list(occupied_cells))


def square_solver(target_num, verbose, model):
    """解决 4x4 方格验证"""
    try:
        if not os.path.exists("0.png"):
            if verbose: print("  ✗ 图片文件不存在: 0.png")
            return []
        
        image = Image.open("0.png")
        image = np.asarray(image)
        result = model.predict(image, task="detect", verbose=False, conf=0.25)
        boxes = result[0].boxes.data
        
        target_index = []
        for i, num in enumerate(result[0].boxes.cls):
            if num == target_num:
                target_index.append(i)
        
        if verbose and len(target_index) > 0:
            print(f"    检测到 {len(target_index)} 个目标物体")
        
        answers = []
        for i in target_index:
            target_box = boxes[i]
            x1, y1 = int(target_box[0]), int(target_box[1])
            x4, y4 = int(target_box[2]), int(target_box[3])
            x2, y2 = x4, y1
            x3, y3 = x1, y4
            xys = [x1, y1, x2, y2, x3, y3, x4, y4]
            
            four_cells = []
            for j in range(4):
                x = xys[j*2]
                y = xys[(j*2)+1]
                
                # 4x4 网格坐标映射
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
            
            answer = get_occupied_cells(four_cells)
            for ans in answer:
                answers.append(ans)
        
        return sorted(list(set(answers)))
    except Exception as e:
        if verbose: print(f"  ✗ 图片识别失败: {e}")
        return []


def get_all_captcha_img_urls(driver):
    """获取所有验证码图片 URL"""
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, '//div[@id="rc-imageselect-target"]//img')))
    
    img_urls = []
    for img in images:
        img_urls.append(img.get_attribute("src"))
    
    return img_urls


def get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, driver):
    """获取动态验证码的新图片 URL"""
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.XPATH, '//div[@id="rc-imageselect-target"]//img')))
    img_urls = []
    
    for img in images:
        try:
            img_urls.append(img.get_attribute("src"))
        except:
            return False, img_urls
    
    # 检查是否有新图片
    index_common = []
    for answer in answers:
        if img_urls[answer-1] == before_img_urls[answer-1]:
            index_common.append(answer)
    
    if len(index_common) >= 1:
        return False, img_urls
    else:
        return True, img_urls


def paste_new_img_on_main_img(main, new, loc):
    """将新图片粘贴到主图片上"""
    paste = np.copy(main)
    
    row = (loc - 1) // 3
    col = (loc - 1) % 3
    
    start_row, end_row = row * 100, (row + 1) * 100
    start_col, end_col = col * 100, (col + 1) * 100
    
    paste[start_row:end_row, start_col:end_col] = new
    
    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite('0.png', paste)


def solve_recaptcha_ia(driver, verbose=True, max_attempts=5):
    """使用 IA 模型解决 reCAPTCHA"""
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"✗ 模型文件不存在: {MODEL_PATH}")
        print("  请确保已下载模型文件到 tmp_rovodev_recaptcha_ia/model.onnx")
        return False
    
    print(f"\n✓ 加载 YOLO 模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")
    
    try:
        # 切换到 checkbox iframe
        driver.switch_to.default_content()
        recaptcha_iframe1 = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]')))
        driver.switch_to.frame(recaptcha_iframe1)
        
        # 点击 checkbox
        print("✓ 点击 reCAPTCHA checkbox...")
        checkbox = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//div[@class="recaptcha-checkbox-border"]')))
        human_like_delay(0.3, 0.8)
        checkbox.click()
        
        # 切换到图片验证 iframe
        driver.switch_to.default_content()
        recaptcha_iframe2 = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, '//iframe[contains(@title, "challenge")]')))
        driver.switch_to.frame(recaptcha_iframe2)
        
        print("✓ 开始识别验证码...")
        
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            if verbose: print(f"\n  尝试 {attempt}/{max_attempts}...")
            
            try:
                reload_attempts = 0
                max_reload_attempts = 5  # 恢复原始配置
                
                while reload_attempts < max_reload_attempts:
                    reload_attempts += 1
                    
                    try:
                        reload = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.ID, 'recaptcha-reload-button')))
                        title_wrapper = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.ID, 'rc-imageselect')))
                    except Exception as e:
                        if verbose: print(f"  定位元素失败: {e}")
                        sleep(2)
                        continue
                    
                    try:
                        target_num = get_target_num(driver)
                    except Exception as e:
                        if verbose: print(f"  获取目标类型失败: {e}")
                        sleep(2)
                        reload.click()
                        sleep(2)
                        continue
                    
                    if target_num == 1000:
                        if verbose: print("  跳过不支持的类型...")
                        random_delay()
                        reload.click()
                        sleep(2)
                    elif "squares" in title_wrapper.text:
                        if verbose: print("  检测到 4x4 方格验证...")
                        try:
                            img_urls = get_all_captcha_img_urls(driver)
                            if not download_img(0, img_urls[0]):
                                reload.click()
                                sleep(2)
                                continue
                        except Exception as e:
                            if verbose: print(f"  获取图片URL失败: {e}")
                            reload.click()
                            sleep(2)
                            continue
                        answers = square_solver(target_num, verbose, model)
                        if len(answers) >= 1 and len(answers) < 16:
                            captcha = "squares"
                            break
                        else:
                            reload.click()
                            sleep(2)
                    elif "none" in title_wrapper.text:
                        if verbose: print("  检测到 3x3 动态验证...")
                        try:
                            img_urls = get_all_captcha_img_urls(driver)
                            if not download_img(0, img_urls[0]):
                                reload.click()
                                sleep(2)
                                continue
                        except Exception as e:
                            if verbose: print(f"  获取图片URL失败: {e}")
                            reload.click()
                            sleep(2)
                            continue
                        answers = dynamic_and_selection_solver(target_num, verbose, model)
                        if len(answers) >= 1:
                            captcha = "dynamic"
                            break
                        else:
                            if verbose: print("    未检测到足够的目标，重新加载...")
                            reload.click()
                            sleep(2)
                    else:
                        if verbose: print("  检测到 3x3 一次性选择验证...")
                        try:
                            img_urls = get_all_captcha_img_urls(driver)
                            if not download_img(0, img_urls[0]):
                                reload.click()
                                sleep(2)
                                continue
                        except Exception as e:
                            if verbose: print(f"  获取图片URL失败: {e}")
                            reload.click()
                            sleep(2)
                            continue
                        answers = dynamic_and_selection_solver(target_num, verbose, model)
                        if len(answers) >= 1:
                            captcha = "selection"
                            break
                        else:
                            if verbose: print("    未检测到足够的目标，重新加载...")
                            reload.click()
                            sleep(2)
                    
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.XPATH, '(//div[@id="rc-imageselect-target"]//td)[1]')))
                    except Exception as e:
                        if verbose: print(f"  等待验证码加载失败: {e}")
                        if reload_attempts < max_reload_attempts:
                            continue
                        else:
                            break
                
                if reload_attempts >= max_reload_attempts:
                    if verbose: print("  重载次数过多，跳过此轮...")
                    continue
                
                if verbose: print(f"  ✓ 识别到的答案位置: {answers}")
                if verbose: print(f"  验证类型: {captcha}")
                
                # 处理动态验证码
                if captcha == "dynamic":
                    for answer in answers:
                        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                            (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()
                        random_delay(mu=0.5, sigma=0.2)  # 恢复原始成功配置
                    
                    dynamic_rounds = 0
                    max_dynamic_rounds = 10
                    
                    while dynamic_rounds < max_dynamic_rounds:
                        dynamic_rounds += 1
                        if verbose: print(f"    动态验证轮次 {dynamic_rounds}/{max_dynamic_rounds}")
                        
                        before_img_urls = img_urls
                        new_img_wait_count = 0
                        max_new_img_wait = 30
                        
                        while new_img_wait_count < max_new_img_wait:
                            new_img_wait_count += 1
                            sleep(0.2)
                            is_new, img_urls = get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, driver)
                            if is_new:
                                break
                        
                        if new_img_wait_count >= max_new_img_wait:
                            if verbose: print("    等待新图片超时，跳出动态验证")
                            break
                        
                        new_img_index_urls = [answer-1 for answer in answers]
                        
                        for index in new_img_index_urls:
                            if not download_img(index+1, img_urls[index]):
                                if verbose: print("    图片下载失败，跳出动态验证")
                                break
                        
                        for answer in answers:
                            try:
                                main_img = Image.open("0.png")
                                new_img = Image.open(f"{answer}.png")
                                paste_new_img_on_main_img(main_img, new_img, answer)
                            except Exception as e:
                                if verbose: print(f"    图片处理失败: {e}")
                                break
                        
                        answers = dynamic_and_selection_solver(target_num, verbose, model)
                        
                        if len(answers) >= 1:
                            for answer in answers:
                                WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                                    (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()
                                random_delay(mu=0.5, sigma=0.1)  # 恢复原始配置
                        else:
                            if verbose: print("    未识别到更多目标，结束动态验证")
                            break
                
                # 处理一次性选择或方格验证
                elif captcha == "selection" or captcha == "squares":
                    for answer in answers:
                        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                            (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()
                        random_delay(mu=0.8, sigma=0.3)  # 恢复之前成功的配置
                
                # 点击验证按钮
                human_like_delay(1.5, 2.5)  # 使用更自然的随机延迟
                verify = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "recaptcha-verify-button")))
                human_like_delay(0.8, 1.5)  # 点击前停顿
                verify.click()
                
                # 等待验证结果
                human_like_delay(3, 4)  # 使用随机延迟
                
                # 检查是否通过
                try:
                    driver.switch_to.default_content()
                    
                    # 方法1: 检查 checkbox 是否被勾选
                    try:
                        recaptcha_iframe1 = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]')))
                        driver.switch_to.frame(recaptcha_iframe1)
                        
                        WebDriverWait(driver, 3).until(
                            EC.presence_of_element_located((By.XPATH, '//span[contains(@aria-checked, "true")]')))
                        
                        if verbose: print("✓✓✓ reCAPTCHA 验证成功（checkbox已勾选）！")
                        driver.switch_to.default_content()
                        return True
                    except:
                        driver.switch_to.default_content()
                    
                    # 方法2: 检查挑战框是否消失或隐藏
                    try:
                        challenge_iframe = driver.find_element(By.XPATH, '//iframe[contains(@title, "challenge")]')
                        if not challenge_iframe.is_displayed():
                            if verbose: print("✓✓✓ reCAPTCHA 验证成功（挑战框已隐藏）！")
                            return True
                    except:
                        if verbose: print("✓✓✓ reCAPTCHA 验证成功（找不到挑战框）！")
                        return True
                    
                    # 验证未通过，继续下一轮
                    recaptcha_iframe2 = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.XPATH, '//iframe[contains(@title, "challenge")]')))
                    driver.switch_to.frame(recaptcha_iframe2)
                    if verbose: print("  验证未通过，重试...")
                    
                except Exception as check_error:
                    if verbose: print(f"  检查验证结果时出错: {check_error}")
                    try:
                        driver.switch_to.default_content()
                        recaptcha_iframe2 = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, '//iframe[contains(@title, "challenge")]')))
                        driver.switch_to.frame(recaptcha_iframe2)
                        if verbose: print("  重新定位到挑战框，继续...")
                    except:
                        if verbose: print("✓✓✓ reCAPTCHA 可能已验证成功（无法定位挑战框）")
                        driver.switch_to.default_content()
                        return True
            
            except Exception as e:
                if verbose: print(f"  本轮尝试失败: {e}")
                if attempt >= max_attempts:
                    print(f"✗ 达到最大尝试次数 ({max_attempts})，验证失败")
                    return False
                else:
                    if verbose: print("  准备下一轮尝试...")
                    try:
                        driver.switch_to.default_content()
                        recaptcha_iframe2 = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, '//iframe[contains(@title, "challenge")]')))
                        driver.switch_to.frame(recaptcha_iframe2)
                    except:
                        if verbose: print("  无法重新定位到验证框，尝试重新开始...")
                        return False
    
    except Exception as e:
        print(f"✗ reCAPTCHA 解决失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def renew_host2play_server():
    """续期 Host2Play 服务器"""
    
    print("=" * 60)
    print("Host2Play 自动续期 - RecaptchaV2-IA-Solver")
    print("=" * 60)
    
    # 检查必需配置
    if not RENEW_URL:
        error_msg = "✗ 错误: RENEW_URL 环境变量未设置"
        print(error_msg)
        if ENABLE_TELEGRAM:
            send_telegram_message(f"❌ <b>Host2Play 续期失败</b>\n\n{error_msg}")
        return
    
    print(f"续期 URL: {RENEW_URL}")
    
    # 发送开始通知
    if ENABLE_TELEGRAM:
        send_telegram_message("🔄 <b>Host2Play 自动续期开始</b>\n\n正在启动浏览器...")
    
    # 配置 Chrome 选项
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--ignore-certificate-errors')
    chrome_options.add_argument('--ignore-certificate-errors-spki-list')
    chrome_options.add_argument('--ignore-ssl-errors')
    chrome_options.add_argument('--allow-insecure-localhost')
    chrome_options.add_argument('--disable-web-security')
    chrome_options.add_argument('--lang=en-US')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    
    # GitHub Actions 或 CI 环境需要的选项
    if HEADLESS or os.environ.get('CI'):
        chrome_options.add_argument('--headless=new')
        print("✓ 使用 headless 模式")
    
    # 尝试禁用自动化特征（如果版本支持）
    try:
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    except:
        pass
    
    seleniumwire_options = {
        'no_proxy': 'localhost,127.0.0.1',
        'disable_encoding': True,
        'verify_ssl': False,
        'suppress_connection_errors': True,
        'disable_capture': False
    }
    
    # 初始化浏览器
    print("\n启动浏览器...")
    driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=seleniumwire_options)
    driver.scopes = ['.*google.com/recaptcha.*']
    
    try:
        # 访问续期页面
        print("\n访问续期页面...")
        driver.get(RENEW_URL)
        sleep(3)
        
        # 页面加载后注入反检测脚本
        try:
            driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                if (!window.chrome) { window.chrome = {}; }
                if (!window.chrome.runtime) { window.chrome.runtime = {}; }
            """)
            print("✓ 已注入反检测脚本")
        except:
            pass
        
        # 不再保存初始截图
        
        # 等待 reCAPTCHA 脚本加载（优化版：更快检测）
        print("\n等待 reCAPTCHA 脚本加载...")
        max_wait = 20  # 减少最大等待时间
        for i in range(max_wait):
            try:
                grecaptcha_ready = driver.execute_script("""
                    return typeof grecaptcha !== 'undefined' && 
                           typeof grecaptcha.render === 'function';
                """)
                if grecaptcha_ready:
                    print(f"✓ reCAPTCHA 脚本已加载（{i+1}秒）")
                    break
            except:
                pass
            
            if i == max_wait - 1:
                print(f"⚠ 等待 {max_wait} 秒后 reCAPTCHA 脚本仍未加载")
            else:
                sleep(0.5)  # 减少检查间隔，更快响应
        
        sleep(1)  # 减少额外等待
        
        # 先检查页面上是否已经有 reCAPTCHA
        print("\n检查页面上是否有 reCAPTCHA...")
        recaptcha_exists = False
        try:
            recaptcha_iframe = driver.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]')
            if recaptcha_iframe.is_displayed():
                print("✓ 页面上已有 reCAPTCHA，先解决验证码")
                recaptcha_exists = True
        except:
            print("  页面上暂无 reCAPTCHA")
        
        # 如果页面上已有 reCAPTCHA，先解决它
        if recaptcha_exists:
            print("\n解决页面上的 reCAPTCHA...")
            success = solve_recaptcha_ia(driver, verbose=VERBOSE)
            
            if not success:
                print("\n⚠ 自动识别未完成，请手动完成验证...")
                print("等待 60 秒...")
                sleep(60)
        
        # 查找并点击 "Renew" 按钮
        print("\n查找并点击 'Renew' 按钮...")
        driver.switch_to.default_content()
        sleep(2)
        
        try:
            # 尝试多种可能的选择器
            renew_button = None
            selectors = [
                "//button[contains(text(), 'Renew server')]",
                "//button[contains(text(), 'Renew')]",
                "//a[contains(text(), 'Renew server')]",
                "//a[contains(text(), 'Renew')]",
                "//input[@value='Renew server']",
                "//input[@value='Renew']",
                "//button[@type='submit']",
                "//input[@type='submit']"
            ]
            
            for selector in selectors:
                try:
                    renew_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector)))
                    print(f"✓ 找到 Renew 按钮: {selector}")
                    break
                except:
                    continue
            
            if renew_button is None:
                print("⚠ 无法找到 'Renew' 按钮，尝试使用 JavaScript...")
                # 尝试通过 JavaScript 查找并点击
                js_code = """
                var buttons = document.querySelectorAll('button, a, input[type="submit"]');
                for (var i = 0; i < buttons.length; i++) {
                    var text = buttons[i].textContent || buttons[i].value || '';
                    if (text.toLowerCase().includes('renew')) {
                        buttons[i].click();
                        return 'Clicked: ' + text;
                    }
                }
                return 'No button found';
                """
                result = driver.execute_script(js_code)
                print(f"  JavaScript 结果: {result}")
                
                if 'No button found' in result:
                    print("\n✗ 无法找到 Renew 按钮")
                    print("  尝试查找所有按钮...")
                    buttons = driver.find_elements(By.TAG_NAME, 'button')
                    for i, btn in enumerate(buttons):
                        try:
                            print(f"    按钮 {i+1}: {btn.text}")
                        except:
                            pass
                    
                    print("\n  请手动点击 Renew 按钮...")
                    sleep(30)
            else:
                # 使用 JavaScript 点击，避免被遮挡
                driver.execute_script("arguments[0].click();", renew_button)
                print("✓ 已点击 Renew 按钮")
            
            # 等待弹窗出现（优化：减少等待）
            print("\n等待弹窗和 reCAPTCHA 加载...")
            sleep(2)  # 减少初始等待
            
            # 等待弹窗中的 reCAPTCHA 渲染（优化：更快检测）
            print("等待弹窗中的 reCAPTCHA 渲染...")
            recaptcha_rendered = False
            for i in range(15):  # 减少最大等待次数
                try:
                    recaptcha_iframe = driver.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]')
                    if recaptcha_iframe.is_displayed():
                        print(f"✓ reCAPTCHA 已渲染（等待 {i+1} 秒）")
                        recaptcha_rendered = True
                        break
                except:
                    pass
                
                sleep(0.5)  # 减少检查间隔，更快响应
            
            if not recaptcha_rendered:
                print("⚠ reCAPTCHA 未渲染，可能需要手动刷新或等待")
                print("  尝试强制触发 reCAPTCHA 渲染...")
                
                # 尝试手动触发 grecaptcha.render
                try:
                    driver.execute_script("""
                        // 查找 reCAPTCHA 容器
                        var containers = document.querySelectorAll('[data-sitekey], .g-recaptcha');
                        if (containers.length > 0 && typeof grecaptcha !== 'undefined') {
                            try {
                                grecaptcha.render(containers[0], {
                                    'sitekey': containers[0].getAttribute('data-sitekey')
                                });
                            } catch(e) {
                                console.log('Manual render failed:', e);
                            }
                        }
                    """)
                    sleep(1.5)  # 减少等待
                    
                    # 再次检查
                    recaptcha_iframe = driver.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]')
                    if recaptcha_iframe.is_displayed():
                        print("✓ 手动触发成功，reCAPTCHA 已渲染")
                        recaptcha_rendered = True
                except Exception as e:
                    print(f"  手动触发失败: {e}")
            
            sleep(1)  # 减少等待
            
        except Exception as e:
            print(f"✗ 点击 Renew 按钮失败: {e}")
        
        # 检查是否出现 reCAPTCHA（无论之前是否存在）
        print("\n检查是否需要解决 reCAPTCHA...")
        sleep(1)  # 减少等待
        
        try:
            recaptcha_iframe = driver.find_element(By.XPATH, '//iframe[@title="reCAPTCHA"]')
            if recaptcha_iframe.is_displayed() and not recaptcha_exists:
                print("✓ 弹窗中出现了 reCAPTCHA，开始解决...")
                
                success = solve_recaptcha_ia(driver, verbose=VERBOSE)
                
                if not success:
                    print("\n⚠ 自动识别未完成，请手动完成验证...")
                    print("等待 60 秒...")
                    sleep(60)
        except:
            print("  无需解决 reCAPTCHA 或已通过验证")
        
        # 验证通过后，点击弹窗内的 Renew 按钮（不是页面上的 Renew server）
        print("\n查找并点击弹窗内的 'Renew' 按钮...")
        driver.switch_to.default_content()
        sleep(1.5)  # 减少等待
        
        try:
            # 专门查找弹窗内的 Renew 按钮，排除 Renew server
            modal_button_selectors = [
                "//div[contains(@class, 'modal')]//button[contains(text(), 'Renew') and not(contains(text(), 'server'))]",
                "//div[contains(@class, 'dialog')]//button[contains(text(), 'Renew') and not(contains(text(), 'server'))]",
                "//div[contains(@class, 'popup')]//button[contains(text(), 'Renew') and not(contains(text(), 'server'))]",
                "//div[contains(@role, 'dialog')]//button[contains(text(), 'Renew') and not(contains(text(), 'server'))]",
                "//div[contains(@class, 'swal')]//button[contains(text(), 'Renew')]",
                "//div[contains(@class, 'swal')]//button[contains(text(), 'Confirm')]",
                "//div[contains(@class, 'modal')]//button[contains(text(), 'Confirm')]",
                "//div[contains(@class, 'modal')]//button[@type='submit']"
            ]
            
            modal_button = None
            for selector in modal_button_selectors:
                try:
                    modal_button = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector)))
                    print(f"✓ 找到弹窗内的 Renew 按钮: {selector}")
                    break
                except:
                    continue
            
            if modal_button is None:
                print("⚠ 标准选择器未找到弹窗按钮，使用 JavaScript 查找...")
                # JavaScript 专门在弹窗内查找
                js_code = """
                // 查找弹窗容器
                var modalSelectors = ['.modal', '.dialog', '.popup', '[role="dialog"]', '.swal2-container', '.swal-modal'];
                var modal = null;
                
                for (var i = 0; i < modalSelectors.length; i++) {
                    var modals = document.querySelectorAll(modalSelectors[i]);
                    for (var j = 0; j < modals.length; j++) {
                        if (modals[j].offsetParent !== null) {  // 可见的弹窗
                            modal = modals[j];
                            break;
                        }
                    }
                    if (modal) break;
                }
                
                if (modal) {
                    // 在弹窗内查找按钮，排除 "Renew server"
                    var buttons = modal.querySelectorAll('button, a, input[type="submit"]');
                    for (var i = 0; i < buttons.length; i++) {
                        var text = (buttons[i].textContent || buttons[i].value || '').toLowerCase();
                        // 只匹配 "renew" 但不包含 "server"
                        if (text.includes('renew') && !text.includes('server')) {
                            buttons[i].click();
                            return 'Clicked modal Renew: ' + buttons[i].textContent;
                        }
                        if (text.includes('confirm') || text.includes('yes') || text.includes('ok')) {
                            buttons[i].click();
                            return 'Clicked modal confirm: ' + buttons[i].textContent;
                        }
                    }
                    return 'Modal found but no Renew button (buttons: ' + buttons.length + ')';
                } else {
                    return 'No modal found';
                }
                """
                result = driver.execute_script(js_code)
                print(f"  JavaScript 结果: {result}")
                
                if 'Clicked' in result:
                    print("✓ 使用 JavaScript 成功点击弹窗内的 Renew 按钮")
                else:
                    print("✗ 无法找到弹窗内的 Renew 按钮")
                    print("  请手动点击弹窗内的 Renew 按钮...")
                    sleep(30)
            else:
                # 使用 JavaScript 点击，避免被遮挡
                driver.execute_script("arguments[0].click();", modal_button)
                print("✓ 已点击弹窗内的 Renew 按钮")
            
            sleep(2)  # 减少等待
            
        except Exception as e:
            print(f"✗ 点击弹窗 Renew 按钮失败: {e}")
        
        sleep(3)
        
        # 检查结果
        print(f"\n当前 URL: {driver.current_url}")
        
        # 等待页面加载完成
        print("\n等待页面加载完成...")
        human_like_delay(3, 5)
        
        # 检查页面是否有成功提示
        success = False
        try:
            page_text = driver.find_element(By.TAG_NAME, 'body').text
            if 'success' in page_text.lower() or 'renewed' in page_text.lower():
                print("✓✓✓ 续期成功！")
                success = True
                
                # 等待页面完全加载（检查加载状态）
                print("等待页面完全加载...")
                
                # 方法1: 检查文档就绪状态
                for i in range(10):
                    try:
                        ready_state = driver.execute_script("return document.readyState")
                        if ready_state == "complete":
                            print(f"✓ 文档就绪状态: complete（检查 {i+1} 次）")
                            break
                    except:
                        pass
                    sleep(0.5)
                
                # 方法2: 检查是否有加载指示器
                for i in range(10):
                    try:
                        loading_elements = driver.find_elements(By.XPATH, 
                            "//*[contains(@class, 'loading') or contains(@class, 'spinner') or contains(@class, 'loader') or contains(text(), 'Loading') or contains(text(), '加载中')]")
                        if not loading_elements or not any(elem.is_displayed() for elem in loading_elements):
                            print(f"✓ 无加载指示器（检查 {i+1} 次）")
                            break
                    except:
                        pass
                    sleep(0.5)
                
                # 额外等待确保所有内容渲染完成
                print("额外等待确保内容完全渲染...")
                human_like_delay(3, 5)
                
                # 只有成功时才保存截图
                driver.save_screenshot(SCREENSHOT_PATH)
                print(f"✓ 已保存成功截图: {SCREENSHOT_PATH}")
                
                # 发送 Telegram 成功通知
                if ENABLE_TELEGRAM:
                    from datetime import datetime
                    success_msg = (
                        "✅ <b>Host2Play 续期成功！</b>\n\n"
                        f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"🔗 URL: {RENEW_URL[:50]}..."
                    )
                    send_telegram_message(success_msg)
                    
                    # 发送截图
                    if os.path.exists(SCREENSHOT_PATH):
                        send_telegram_photo(SCREENSHOT_PATH, "📸 续期成功截图")
            else:
                print("⚠ 请检查页面确认续期是否成功")
        except:
            print("⚠ 无法检查续期结果，请手动确认")
        
        # 如果没有成功，发送失败通知
        if not success and ENABLE_TELEGRAM:
            from datetime import datetime
            failure_msg = (
                "⚠️ <b>Host2Play 续期状态未知</b>\n\n"
                f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔗 URL: {RENEW_URL[:50]}...\n\n"
                "请手动检查续期结果"
            )
            send_telegram_message(failure_msg)
        
        print("\n浏览器将保持打开 10 秒...")
        sleep(10)
        
    except Exception as e:
        print(f"\n✗ 续期失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 发送失败通知
        if ENABLE_TELEGRAM:
            from datetime import datetime
            error_msg = (
                "❌ <b>Host2Play 续期失败！</b>\n\n"
                f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔗 URL: {RENEW_URL[:50] if RENEW_URL else 'N/A'}...\n"
                f"❗ 错误: {str(e)[:100]}"
            )
            send_telegram_message(error_msg)
    finally:
        print("\n关闭浏览器...")
        driver.quit()
        
        # 清理临时图片
        for i in range(17):
            try:
                os.remove(f"{i}.png")
            except:
                pass


if __name__ == "__main__":
    try:
        renew_host2play_server()
        print("\n✓ 脚本执行完成")
    except Exception as e:
        print(f"\n✗ 脚本执行失败: {e}")
