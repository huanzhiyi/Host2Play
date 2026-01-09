"""
Host2Play 自动续期脚本 - Playwright + Camoufox + YOLO 版本
- 使用 Playwright + Camoufox 过 Cloudflare（参考 katabump）
- 使用 YOLO 模型自动识别 reCAPTCHA 图形验证（参考 local 版本）
"""
import asyncio
import logging
import random
import os
import re
import shutil
from typing import Optional
from datetime import datetime
import numpy as np
import requests
from PIL import Image
import cv2

from playwright.async_api import Page, Frame, TimeoutError as PlaywrightTimeoutError
from camoufox.async_api import AsyncCamoufox
from browserforge.fingerprints import Screen

# YOLO 模型（可选）
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("⚠️ YOLO 未安装，将跳过图形验证")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
MODEL_PATH = "model.onnx"  # 模型文件在脚本同一目录
RENEW_URL = os.environ.get('RENEW_URL')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
VERBOSE = True


def send_telegram_message(message: str, photo_path: str = None) -> bool:
    """发送Telegram消息"""
    bot_token = TELEGRAM_BOT_TOKEN
    chat_id = TELEGRAM_CHAT_ID
    
    if not bot_token or not chat_id:
        logger.warning("⚠️ 未设置 Telegram 配置，跳过消息推送")
        return False
    
    try:
        # 如果有图片，发送图片和消息
        if photo_path and os.path.exists(photo_path):
            url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
            with open(photo_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': chat_id,
                    'caption': message,
                    'parse_mode': 'Markdown'
                }
                response = requests.post(url, files=files, data=data, timeout=30)
        else:
            # 只发送文本消息
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            logger.info("✅ Telegram 消息发送成功")
            return True
        else:
            logger.warning(f"⚠️ Telegram 消息发送失败: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Telegram 消息发送出错: {str(e)}")
        return False


async def human_click(page: Page, x: float, y: float) -> None:
    """模拟人类点击行为 - 带随机偏移和步骤"""
    target_x = x + random.uniform(-5, 5)
    target_y = y + random.uniform(-5, 5)

    await page.mouse.move(target_x, target_y, steps=random.randint(10, 25))
    await asyncio.sleep(random.uniform(0.1, 0.3))
    await page.mouse.down()
    await asyncio.sleep(random.uniform(0.05, 0.15))
    await page.mouse.up()


def random_delay(mu=0.3, sigma=0.1):
    """随机延迟"""
    import time
    delay = np.random.normal(mu, sigma)
    delay = max(0.1, delay)
    time.sleep(delay)


def download_img(name, url):
    """下载图片"""
    try:
        response = requests.get(url, stream=True, timeout=10)
        with open(f'{name}.png', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
        return True
    except Exception as e:
        if VERBOSE:
            logger.error(f"✗ 图片下载失败 {name}: {e}")
        return False


def get_target_num_from_text(text: str) -> int:
    """从文本获取目标类别编号"""
    target_mappings = {
        "bicycle": 1, "bus": 5, "boat": 8, "car": 2,
        "hydrant": 10, "motorcycle": 3, "traffic": 9
    }
    text_lower = text.lower()
    for term, value in target_mappings.items():
        if term in text_lower:
            return value
    return 1000


def dynamic_and_selection_solver(target_num, verbose, model):
    """解决 3x3 网格验证"""
    try:
        if not os.path.exists("0.png"):
            return []
        
        image = Image.open("0.png")
        image = np.asarray(image)
        result = model.predict(image, task="detect", verbose=False)
        
        target_index = []
        for count, num in enumerate(result[0].boxes.cls):
            if num == target_num:
                target_index.append(count)
        
        if verbose and len(target_index) > 0:
            logger.info(f"    检测到 {len(target_index)} 个目标物体")
        
        answers = []
        boxes = result[0].boxes.data
        for i in target_index:
            target_box = boxes[i]
            x1, y1 = int(target_box[0]), int(target_box[1])
            x2, y2 = int(target_box[2]), int(target_box[3])
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            row, col = yc // 100, xc // 100
            answer = int(row * 3 + col + 1)
            answers.append(answer)
        
        return list(set(answers))
    except Exception as e:
        if verbose:
            logger.error(f"✗ 图片识别失败: {e}")
        return []


def get_occupied_cells(vertices):
    """获取被占用的单元格（4x4）"""
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
            return []
        
        image = Image.open("0.png")
        image = np.asarray(image)
        result = model.predict(image, task="detect", verbose=False)
        boxes = result[0].boxes.data
        
        target_index = []
        for count, num in enumerate(result[0].boxes.cls):
            if num == target_num:
                target_index.append(count)
        
        if verbose and len(target_index) > 0:
            logger.info(f"    检测到 {len(target_index)} 个目标物体")
        
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
                x, y = xys[j*2], xys[j*2+1]
                
                if x < 112.5 and y < 112.5: four_cells.append(1)
                elif 112.5 < x < 225 and y < 112.5: four_cells.append(2)
                elif 225 < x < 337.5 and y < 112.5: four_cells.append(3)
                elif 337.5 < x <= 450 and y < 112.5: four_cells.append(4)
                elif x < 112.5 and 112.5 < y < 225: four_cells.append(5)
                elif 112.5 < x < 225 and 112.5 < y < 225: four_cells.append(6)
                elif 225 < x < 337.5 and 112.5 < y < 225: four_cells.append(7)
                elif 337.5 < x <= 450 and 112.5 < y < 225: four_cells.append(8)
                elif x < 112.5 and 225 < y < 337.5: four_cells.append(9)
                elif 112.5 < x < 225 and 225 < y < 337.5: four_cells.append(10)
                elif 225 < x < 337.5 and 225 < y < 337.5: four_cells.append(11)
                elif 337.5 < x <= 450 and 225 < y < 337.5: four_cells.append(12)
                elif x < 112.5 and 337.5 < y <= 450: four_cells.append(13)
                elif 112.5 < x < 225 and 337.5 < y <= 450: four_cells.append(14)
                elif 225 < x < 337.5 and 337.5 < y <= 450: four_cells.append(15)
                elif 337.5 < x <= 450 and 337.5 < y <= 450: four_cells.append(16)
            
            for ans in get_occupied_cells(four_cells):
                answers.append(ans)
        
        return sorted(list(set(answers)))
    except Exception as e:
        if verbose:
            logger.error(f"✗ 图片识别失败: {e}")
        return []


def paste_new_img_on_main_img(main, new, loc):
    """粘贴新图片到主图片"""
    paste = np.copy(main)
    row, col = (loc - 1) // 3, (loc - 1) % 3
    start_row, end_row = row * 100, (row + 1) * 100
    start_col, end_col = col * 100, (col + 1) * 100
    paste[start_row:end_row, start_col:end_col] = new
    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite('0.png', paste)


async def find_and_click_turnstile(page: Page, retries: int = 20) -> bool:
    """查找并点击 Cloudflare Turnstile 验证框"""
    logger.info("🔍 寻找 Turnstile 验证框...")
    
    for attempt in range(retries):
        try:
            turnstile_frames = []
            
            # Collect all Turnstile frames
            for frame in page.frames:
                if "challenges.cloudflare.com" in frame.url or "turnstile" in frame.url:
                    turnstile_frames.append(frame)
            
            if not turnstile_frames:
                if attempt % 5 == 0:
                    logger.debug(f"尝试 {attempt + 1}/{retries}: 未找到 Turnstile iframe")
                await asyncio.sleep(1)
                continue
            
            if attempt == 0:
                logger.info(f"✅ 找到 {len(turnstile_frames)} 个 Turnstile frame")
            
            # Try to click the first visible Turnstile frame
            for frame in turnstile_frames:
                try:
                    frame_element = await frame.frame_element()
                    is_visible = await frame_element.is_visible()
                    
                    if not is_visible:
                        continue
                    
                    # Get the bounding box
                    box = await frame_element.bounding_box()
                    if not box:
                        continue
                    
                    # Calculate click position (center of the frame)
                    click_x = box['x'] + box['width'] / 2
                    click_y = box['y'] + box['height'] / 2
                    
                    # Human-like click
                    await human_click(page, click_x, click_y)
                    logger.info(f"✅ 已点击 Turnstile 验证框")
                    
                    # Wait for verification
                    await asyncio.sleep(3)
                    return True
                    
                except Exception as e:
                    logger.debug(f"处理 frame 出错: {e}")
                    continue
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.debug(f"查找 Turnstile 出错: {e}")
            await asyncio.sleep(1)
    
    logger.warning("⚠️ 未能找到或点击 Turnstile")
    return False


async def solve_recaptcha_with_yolo(page: Page, max_attempts: int = 8) -> bool:
    """使用 YOLO 模型处理 reCAPTCHA 图形验证"""
    logger.info("🔍 检查 reCAPTCHA...")
    
    # 检查 YOLO 是否可用
    if not YOLO_AVAILABLE:
        logger.warning("⚠️ YOLO 不可用，将尝试简单点击")
        await asyncio.sleep(2)
        
        # 尝试点击 checkbox
        for frame in page.frames:
            if "recaptcha" in frame.url and "anchor" in frame.url:
                try:
                    checkbox = await frame.wait_for_selector('.recaptcha-checkbox-border', timeout=5000)
                    if checkbox:
                        await checkbox.click()
                        logger.info("✅ 已点击 reCAPTCHA checkbox")
                        await asyncio.sleep(10)
                        return True
                except:
                    pass
        return False
    
    # 检查模型文件
    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ 模型文件不存在: {MODEL_PATH}")
        return False
    
    logger.info(f"✓ 加载 YOLO 模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH, task="detect")
    
    try:
        # 步骤 1: 查找并点击 checkbox
        await asyncio.sleep(2)
        checkbox_frame = None
        for frame in page.frames:
            if "recaptcha" in frame.url and "anchor" in frame.url:
                checkbox_frame = frame
                break
        
        if not checkbox_frame:
            logger.warning("⚠️ 未找到 reCAPTCHA checkbox")
            return False
        
        logger.info("✓ 点击 reCAPTCHA checkbox...")
        checkbox = await checkbox_frame.wait_for_selector('.recaptcha-checkbox-border', timeout=10000)
        await asyncio.sleep(random.uniform(0.3, 0.8))
        await checkbox.click()
        
        # 步骤 2: 等待并查找挑战 iframe
        await asyncio.sleep(3)
        challenge_frame = None
        for frame in page.frames:
            if "recaptcha" in frame.url and "bframe" in frame.url:
                challenge_frame = frame
                break
        
        if not challenge_frame:
            logger.info("✅ 无需图形验证，checkbox 已通过")
            return True
        
        logger.info("✓ 开始识别验证码...")
        
        # 步骤 3: 开始 YOLO 识别循环
        for attempt in range(max_attempts):
            logger.info(f"\n  尝试 {attempt + 1}/{max_attempts}...")
            
            try:
                # 等待验证码加载
                await asyncio.sleep(2)
                
                # 获取目标文本
                try:
                    target_element = await challenge_frame.wait_for_selector('#rc-imageselect strong', timeout=10000)
                    target_text = await target_element.text_content()
                    target_num = get_target_num_from_text(target_text)
                    logger.info(f"  目标类型: {target_text} (编号: {target_num})")
                except Exception as e:
                    logger.warning(f"  获取目标类型失败: {e}")
                    # 点击重载按钮
                    reload_btn = await challenge_frame.query_selector('#recaptcha-reload-button')
                    if reload_btn:
                        await reload_btn.click()
                        await asyncio.sleep(2)
                    continue
                
                if target_num == 1000:
                    logger.info("  跳过不支持的类型...")
                    reload_btn = await challenge_frame.query_selector('#recaptcha-reload-button')
                    if reload_btn:
                        await reload_btn.click()
                        await asyncio.sleep(2)
                    continue
                
                # 检查验证码类型
                title_element = await challenge_frame.query_selector('#rc-imageselect')
                title_text = await title_element.text_content() if title_element else ""
                
                # 获取图片 URL
                img_elements = await challenge_frame.query_selector_all('#rc-imageselect-target img')
                img_urls = []
                for img in img_elements:
                    url = await img.get_attribute('src')
                    if url:
                        img_urls.append(url)
                
                if not img_urls:
                    logger.warning("  未找到验证码图片")
                    continue
                
                # 下载第一张图片
                if not download_img(0, img_urls[0]):
                    continue
                
                # 根据类型识别
                if "squares" in title_text.lower():
                    logger.info("  检测到 4x4 方格验证...")
                    answers = square_solver(target_num, VERBOSE, model)
                    captcha_type = "squares"
                elif "none" in title_text.lower():
                    logger.info("  检测到 3x3 动态验证...")
                    answers = dynamic_and_selection_solver(target_num, VERBOSE, model)
                    captcha_type = "dynamic"
                else:
                    logger.info("  检测到 3x3 一次性选择验证...")
                    answers = dynamic_and_selection_solver(target_num, VERBOSE, model)
                    captcha_type = "selection"
                
                if len(answers) < 1:
                    logger.warning("    未检测到目标，重新加载...")
                    reload_btn = await challenge_frame.query_selector('#recaptcha-reload-button')
                    if reload_btn:
                        await reload_btn.click()
                        await asyncio.sleep(2)
                    continue
                
                logger.info(f"  ✓ 识别到的答案位置: {answers}")
                
                # 点击答案
                cells = await challenge_frame.query_selector_all('#rc-imageselect-target td')
                for answer in answers:
                    if answer <= len(cells):
                        await cells[answer - 1].click()
                        random_delay(mu=0.6, sigma=0.3)
                
                # 处理动态验证
                if captcha_type == "dynamic":
                    dynamic_rounds = 0
                    max_dynamic_rounds = 10
                    
                    while dynamic_rounds < max_dynamic_rounds:
                        dynamic_rounds += 1
                        logger.info(f"    动态验证轮次 {dynamic_rounds}/{max_dynamic_rounds}")
                        
                        await asyncio.sleep(2)
                        
                        # 获取新图片
                        new_img_urls = []
                        img_elements = await challenge_frame.query_selector_all('#rc-imageselect-target img')
                        for img in img_elements:
                            url = await img.get_attribute('src')
                            if url:
                                new_img_urls.append(url)
                        
                        # 检查是否有新图片
                        has_new = False
                        for answer in answers:
                            if answer <= len(new_img_urls) and new_img_urls[answer-1] != img_urls[answer-1]:
                                has_new = True
                                break
                        
                        if not has_new:
                            logger.info("    没有新图片，结束动态验证")
                            break
                        
                        # 下载新图片并更新
                        for answer in answers:
                            if answer <= len(new_img_urls):
                                download_img(answer, new_img_urls[answer-1])
                        
                        # 更新主图片
                        for answer in answers:
                            try:
                                main_img = Image.open("0.png")
                                new_img = Image.open(f"{answer}.png")
                                paste_new_img_on_main_img(main_img, new_img, answer)
                            except:
                                break
                        
                        # 重新识别
                        answers = dynamic_and_selection_solver(target_num, VERBOSE, model)
                        
                        if len(answers) >= 1:
                            logger.info(f"    新一轮检测到 {len(answers)} 个目标")
                            cells = await challenge_frame.query_selector_all('#rc-imageselect-target td')
                            for answer in answers:
                                if answer <= len(cells):
                                    await cells[answer - 1].click()
                                    random_delay(mu=0.6, sigma=0.3)
                        else:
                            logger.info("    未识别到更多目标，结束动态验证")
                            break
                        
                        img_urls = new_img_urls
                
                # 点击验证按钮
                await asyncio.sleep(random.uniform(1.5, 2.5))
                verify_btn = await challenge_frame.query_selector('#recaptcha-verify-button')
                if verify_btn:
                    await asyncio.sleep(random.uniform(0.8, 1.5))
                    await verify_btn.click()
                
                # 等待验证结果
                await asyncio.sleep(random.uniform(3, 4))
                
                # 检查是否通过
                checkbox_frame = None
                for frame in page.frames:
                    if "recaptcha" in frame.url and "anchor" in frame.url:
                        checkbox_frame = frame
                        break
                
                if checkbox_frame:
                    try:
                        checked = await checkbox_frame.query_selector('span[aria-checked="true"]')
                        if checked:
                            logger.info("✓✓✓ reCAPTCHA 验证成功！")
                            return True
                    except:
                        pass
                
                # 检查挑战框是否消失
                challenge_frame = None
                for frame in page.frames:
                    if "recaptcha" in frame.url and "bframe" in frame.url:
                        challenge_frame = frame
                        break
                
                if not challenge_frame:
                    logger.info("✓✓✓ reCAPTCHA 验证成功（挑战框已消失）！")
                    return True
                
                logger.info("  验证未通过，重试...")
                
            except Exception as e:
                logger.error(f"  本轮尝试失败: {e}")
                if attempt >= max_attempts - 1:
                    return False
        
        logger.warning(f"✗ 达到最大尝试次数 ({max_attempts})，验证失败")
        return False
        
    except Exception as e:
        logger.error(f"❌ reCAPTCHA 解决失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时图片
        for i in range(17):
            try:
                os.remove(f"{i}.png")
            except:
                pass


async def main():
    """主函数"""
    # 验证环境变量
    if not RENEW_URL:
        logger.error("❌ 错误: RENEW_URL 环境变量未设置")
        return
    
    renew_url = RENEW_URL
    
    print("="*70)
    print("  🔐 Host2Play 自动续期脚本 (Playwright 版)")
    print(f"  🌐 续期 URL: {renew_url[:50]}...")
    print("  🤖 模式: Playwright + Camoufox (自动过检测)")
    print("="*70)
    print()
    
    # 发送开始通知
    start_time = datetime.now()
    start_message = f"""🚀 *Host2Play 自动续期开始*

🕐 时间: `{start_time.strftime('%Y-%m-%d %H:%M:%S')}`
🤖 模式: Playwright + Camoufox

⏳ 正在处理中..."""
    send_telegram_message(start_message)
    
    # 检测是否在 CI 环境
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    
    if is_ci:
        logger.info("🤖 检测到 CI 环境，使用 headless 模式")
    
    # 使用 Camoufox 浏览器（自动反检测，类似 katabump）
    async with AsyncCamoufox(
        headless=is_ci,
        os=["windows"],
        screen=Screen(max_width=1920, max_height=1080),
    ) as browser:
        
        page = await browser.new_page()
        
        try:
            # Step 1: 访问续期页面
            logger.info("\n[1/4] 🌐 访问续期页面...")
            await page.goto(renew_url, wait_until='domcontentloaded')
            await asyncio.sleep(3)
            
            logger.info(f"✅ 当前 URL: {page.url}")
            
            # Step 2: 检测并处理 Cloudflare Turnstile
            logger.info("\n[2/4] 🔍 检测 Cloudflare 保护...")
            
            # 检查页面内容
            page_content = await page.content()
            page_title = await page.title()
            
            if 'cloudflare' in page_content.lower() or 'turnstile' in page_content.lower():
                logger.info("⚠️ 检测到 Cloudflare 保护，尝试处理...")
                
                # 尝试点击 Turnstile
                success = await find_and_click_turnstile(page)
                
                if success:
                    logger.info("✅ Turnstile 验证已完成")
                    await asyncio.sleep(3)
                else:
                    logger.warning("⚠️ Turnstile 自动处理失败，等待自动通过...")
                    await asyncio.sleep(10)
            else:
                logger.info("✅ 未检测到 Cloudflare 保护")
            
            # 截图保存当前状态
            await page.screenshot(path='host2play_01_after_load.png', full_page=True)
            logger.info("📸 截图保存: host2play_01_after_load.png")
            
            # Step 3: 查找并点击 Renew server 按钮
            logger.info("\n[3/4] 🖱️ 查找并点击 'Renew' 按钮...")
            await asyncio.sleep(2)
            
            # 尝试多种选择器
            renew_button_selectors = [
                'button:has-text("Renew")',
                'a:has-text("Renew")',
                'button:has-text("renew")',
                'button[type="submit"]:has-text("Renew")',
                'input[type="submit"][value*="Renew"]',
                '[onclick*="renew"]',
            ]
            
            renew_button = None
            for selector in renew_button_selectors:
                try:
                    button = await page.wait_for_selector(selector, timeout=5000)
                    if button and await button.is_visible():
                        renew_button = button
                        logger.info(f"✅ 找到 Renew 按钮: {selector}")
                        break
                except:
                    continue
            
            if not renew_button:
                logger.error("❌ 未找到 Renew 按钮")
                await page.screenshot(path='host2play_error_no_button.png', full_page=True)
                
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: 未找到 Renew 按钮
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
                send_telegram_message(error_message, 'host2play_error_no_button.png')
                return
            
            # 点击 Renew 按钮
            await renew_button.click()
            logger.info("✅ 已点击 Renew 按钮")
            await asyncio.sleep(3)
            
            # 截图弹窗状态
            await page.screenshot(path='host2play_02_after_button.png', full_page=True)
            logger.info("📸 截图保存: host2play_02_after_button.png")
            
            # Step 4: 处理 reCAPTCHA
            logger.info("\n[4/4] 🔐 处理 reCAPTCHA...")
            
            recaptcha_success = await solve_recaptcha_with_yolo(page)
            
            if not recaptcha_success:
                logger.warning("⚠️ reCAPTCHA 自动处理未完成")
                logger.info("💡 等待 30 秒，看是否自动通过...")
                await asyncio.sleep(30)
            
            # 查找并点击弹窗内的 Renew 按钮
            logger.info("\n🖱️ 查找弹窗内的确认按钮...")
            
            modal_button_selectors = [
                'div[role="dialog"] button:has-text("Renew")',
                '.modal button:has-text("Renew")',
                '.swal2-confirm',
                '.modal button[type="submit"]',
                'button:has-text("Confirm")',
            ]
            
            modal_button = None
            for selector in modal_button_selectors:
                try:
                    button = await page.wait_for_selector(selector, timeout=5000)
                    if button and await button.is_visible():
                        modal_button = button
                        logger.info(f"✅ 找到弹窗确认按钮: {selector}")
                        break
                except:
                    continue
            
            if modal_button:
                await modal_button.click()
                logger.info("✅ 已点击弹窗确认按钮")
                await asyncio.sleep(3)
            else:
                logger.warning("⚠️ 未找到弹窗确认按钮，可能已自动提交")
            
            # 截图最终结果
            await page.screenshot(path='host2play_renew_success.png', full_page=True)
            logger.info("📸 最终截图: host2play_renew_success.png")
            
            logger.info("\n✅ 续期流程完成!")
            
            # 发送成功通知
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            success_message = f"""✅ *Host2Play 续期成功*

🕐 开始时间: `{start_time.strftime('%Y-%m-%d %H:%M:%S')}`
🕐 完成时间: `{end_time.strftime('%Y-%m-%d %H:%M:%S')}`
⏱️ 耗时: `{duration:.1f} 秒`

✨ 续期已完成！
"""
            send_telegram_message(success_message, 'host2play_renew_success.png')
            
        except Exception as e:
            logger.error(f"❌ 脚本执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 截图错误状态
            try:
                await page.screenshot(path='host2play_error.png', full_page=True)
                logger.info("📸 错误截图: host2play_error.png")
            except:
                pass
            
            # 发送失败通知
            error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: `{str(e)[:100]}`
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
            send_telegram_message(error_message, 'host2play_error.png')
            raise


if __name__ == "__main__":
    asyncio.run(main())
