"""
Host2Play 自动续期脚本 - Playwright + Camoufox + YOLO 版本
- 使用 Playwright + Camoufox 过 Cloudflare（参考 katabump）
- 使用 YOLO 模型自动识别 reCAPTCHA 图形验证（基于 Breaking-reCAPTCHAv2 项目改进）

主要改进（参考 https://github.com/aplesner/Breaking-reCAPTCHAv2）：
1. 改进的重试循环：使用双层循环，外层控制总尝试次数，内层持续寻找支持的验证码类型
2. 更好的图片变化检测：改进动态验证中的新图片检测逻辑，使用重试机制等待图片加载
3. 更健壮的错误处理：在每个关键步骤都检查验证状态，及时返回成功
4. 优化的延迟策略：使用更符合人类行为的随机延迟
5. 帧重新获取：处理可能的帧分离问题，每次操作前重新获取 frame 引用
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
    logging.info("✅ YOLO 模块导入成功")
except ImportError as e:
    YOLO_AVAILABLE = False
    logging.error(f"❌ YOLO 导入失败: {e}")
    logging.warning("⚠️ YOLO 未安装，将跳过图形验证")
except Exception as e:
    YOLO_AVAILABLE = False
    logging.error(f"❌ YOLO 导入异常: {e}")
    import traceback
    traceback.print_exc()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
MODEL_PATH = "model.onnx"  # 模型文件在脚本同一目录
MODEL_DOWNLOAD_URLS = [
    # 从你的 fork 仓库下载 reCAPTCHA 专用模型
    "https://media.githubusercontent.com/media/DannyLuna17/RecaptchaV2-IA-Solver/main/model.onnx",  # 推荐：直接从 LFS 存储
    "https://github.com/DannyLuna17/RecaptchaV2-IA-Solver/raw/main/model.onnx",  # 备选：raw API（可能返回 LFS 指针）
]
RENEW_URL = os.environ.get('RENEW_URL')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
VERBOSE = True


def download_yolo_model():
    """下载 YOLO 模型文件（如果不存在）"""
    # 如果模型文件已存在且大小正常，跳过下载
    if os.path.exists(MODEL_PATH):
        file_size = os.path.getsize(MODEL_PATH)
        if file_size > 1000000:  # 大于 1MB，认为是有效文件
            logger.info(f"✅ 模型文件已存在: {MODEL_PATH} ({file_size / (1024*1024):.2f} MB)")
            return True
        else:
            logger.warning(f"⚠️ 模型文件大小异常 ({file_size} bytes)，将重新下载")
            os.remove(MODEL_PATH)
    
    logger.info("📥 模型文件不存在，开始下载...")
    
    # 尝试多种下载方法
    for i, url in enumerate(MODEL_DOWNLOAD_URLS, 1):
        try:
            logger.info(f"🔄 尝试方法 {i}/{len(MODEL_DOWNLOAD_URLS)}: {url[:80]}...")
            
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            # 下载到临时文件
            temp_path = MODEL_PATH + ".tmp"
            with open(temp_path, 'wb') as f:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 每下载 10MB 显示一次进度
                        if downloaded % (10 * 1024 * 1024) == 0:
                            logger.info(f"   已下载: {downloaded / (1024*1024):.1f} MB")
            
            # 验证文件大小
            file_size = os.path.getsize(temp_path)
            if file_size < 1000000:
                logger.warning(f"⚠️ 下载的文件大小异常 ({file_size} bytes)，可能是 LFS 指针文件")
                os.remove(temp_path)
                continue
            
            # 重命名为正式文件
            os.rename(temp_path, MODEL_PATH)
            logger.info(f"✅ 模型下载成功！文件大小: {file_size / (1024*1024):.2f} MB")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ 方法 {i} 失败: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue
    
    logger.error("❌ 所有下载方法均失败！")
    return False


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
    """下载图片 - 增强版"""
    try:
        # 输出 URL 信息用于调试
        if VERBOSE:
            logger.info(f"  📥 下载图片 {name}: {url[:100]}{'...' if len(url) > 100 else ''}")
        
        # 检查是否是 Base64 编码的图片
        if url.startswith('data:image'):
            logger.info(f"  🔍 检测到 Base64 编码图片")
            import base64
            try:
                header, data = url.split(',', 1)
                image_data = base64.b64decode(data)
                with open(f'{name}.png', 'wb') as f:
                    f.write(image_data)
                logger.info(f"  ✅ Base64 图片解码成功: {len(image_data)} bytes")
            except Exception as e:
                logger.error(f"  ❌ Base64 解码失败: {e}")
                return False
        else:
            # 正常 URL，使用 requests 下载
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://www.google.com/recaptcha/',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            }
            
            response = requests.get(url, stream=True, timeout=10, headers=headers)
            response.raise_for_status()
            
            # 检查 Content-Type
            content_type = response.headers.get('Content-Type', '')
            if VERBOSE:
                logger.info(f"  📋 Content-Type: {content_type}")
            
            if 'image' not in content_type and content_type:
                logger.error(f"  ❌ URL 返回的不是图片: {content_type}")
                return False
            
            with open(f'{name}.png', 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
        
        # 验证图片文件
        file_size = os.path.getsize(f'{name}.png')
        if file_size < 1000:
            logger.error(f"  ❌ 图片文件过小: {name}.png ({file_size} bytes)")
            # 查看文件内容
            with open(f'{name}.png', 'rb') as f:
                content = f.read(500)
                logger.error(f"  📄 文件前 500 字节: {content[:500]}")
            return False
        
        # 尝试打开验证
        try:
            test_img = Image.open(f'{name}.png')
            test_img.verify()
            # 重新打开获取尺寸（verify 后需要重新打开）
            test_img = Image.open(f'{name}.png')
            if VERBOSE:
                logger.info(f"  ✅ 图片验证成功: {name}.png ({file_size} bytes, {test_img.size})")
        except Exception as e:
            logger.error(f"  ❌ 图片文件损坏: {name}.png - {e}")
            with open(f'{name}.png', 'rb') as f:
                content = f.read(100)
                logger.error(f"  📄 文件头: {content[:100]}")
            return False
        
        return True
    except Exception as e:
        if VERBOSE:
            logger.error(f"  ❌ 图片下载失败 {name}: {e}")
            import traceback
            traceback.print_exc()
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
    """解决 4x4 方格验证 - 使用角点算法（本地成功版本）"""
    try:
        if not os.path.exists("0.png"):
            return []
        
        image = Image.open("0.png")
        image = np.asarray(image)
        result = model.predict(image, task="detect", verbose=False)
        boxes = result[0].boxes.data
        
        target_index = []
        count = 0
        for num in result[0].boxes.cls:
            if num == target_num:
                target_index.append(count)
            count += 1
        
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


def normalize_captcha_image(path: str, expected_size: tuple[int, int]) -> bool:
    """将截图得到的验证码图片标准化到 YOLO 预期尺寸（3x3:300x300, 4x4:450x450）"""
    try:
        img = Image.open(path).convert('RGB')
        if img.size != expected_size:
            img = img.resize(expected_size, Image.BILINEAR)
            img.save(path)
        return True
    except Exception as e:
        logger.error(f"❌ 标准化图片失败 {path}: {e}")
        return False


async def screenshot_captcha_grid(challenge_frame: Frame, grid_size: int) -> bool:
    """截图整个验证码网格到 0.png。grid_size=3 或 4"""
    try:
        # rc-imageselect-target 包含表格网格
        grid = await challenge_frame.wait_for_selector('#rc-imageselect-target', timeout=10000)
        await grid.screenshot(path='0.png')
        expected = (300, 300) if grid_size == 3 else (450, 450)
        if not normalize_captcha_image('0.png', expected):
            return False
        try:
            fs = os.path.getsize('0.png')
            sz = Image.open('0.png').size
            logger.info(f"  ✅ 验证码网格截图成功: 0.png -> {expected}, 实际: {sz}, {fs} bytes")
        except Exception:
            logger.info(f"  ✅ 验证码网格截图成功: 0.png -> {expected}")
        return True
    except Exception as e:
        logger.error(f"  ❌ 验证码网格截图失败: {e}")
        return False


async def screenshot_captcha_tile(challenge_frame: Frame, index: int, grid_size: int) -> bool:
    """截图单个方格到 {index}.png。index 从 1 开始"""
    try:
        cell_selector = f'#rc-imageselect-target td:nth-child({index})'
        cell = await challenge_frame.wait_for_selector(cell_selector, timeout=10000)
        await cell.screenshot(path=f'{index}.png')
        expected = (100, 100) if grid_size == 3 else (112, 112)
        # 4x4 单格理论是 112.5，这里用 112 近似（只用于拼接/调试），主识别用 0.png
        normalize_captcha_image(f'{index}.png', expected)
        return True
    except Exception as e:
        logger.warning(f"    ⚠️ 方格截图失败 {index}: {e}")
        return False


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


async def solve_recaptcha_with_yolo(page: Page, max_attempts: int = 10) -> bool:
    """使用 YOLO 模型处理 reCAPTCHA 图形验证 - 基于 Breaking-reCAPTCHAv2 项目改进"""
    logger.info("🔍 检查 reCAPTCHA...")
    logger.info(f"📊 YOLO_AVAILABLE = {YOLO_AVAILABLE}")
    logger.info(f"📊 MODEL_PATH = {MODEL_PATH}")
    logger.info(f"📊 模型文件存在 = {os.path.exists(MODEL_PATH)}")
    
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
    
    # 检查并下载模型文件
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"⚠️ 模型文件不存在，尝试下载: {MODEL_PATH}")
        if not download_yolo_model():
            logger.error(f"❌ 模型文件下载失败")
            return False
    
    logger.info(f"✓ 加载 YOLO 模型: {MODEL_PATH}")
    logger.info(f"✓ 模型文件大小: {os.path.getsize(MODEL_PATH) / (1024*1024):.2f} MB")
    
    try:
        model = YOLO(MODEL_PATH, task="detect")
        logger.info(f"✅ YOLO 模型加载成功")
    except Exception as e:
        logger.error(f"❌ YOLO 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
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
        
        # 步骤 3: 持续尝试直到验证成功（参考 Breaking-reCAPTCHAv2）
        outer_attempt = 0
        while outer_attempt < max_attempts:
            outer_attempt += 1
            logger.info(f"\n=== 外层尝试 {outer_attempt}/{max_attempts} ===")
            
            try:
                # 内层循环: 寻找合适的验证码类型
                captcha_type = None
                answers = []
                img_urls = []
                target_num = 1000
                
                # 持续重载直到找到支持的类型
                reload_count = 0
                max_reload = 15
                
                while reload_count < max_reload:
                    reload_count += 1
                    
                    # 等待验证码加载
                    await asyncio.sleep(1.5)
                    
                    # 检查是否已通过验证
                    checkbox_frame_check = None
                    for frame in page.frames:
                        if "recaptcha" in frame.url and "anchor" in frame.url:
                            checkbox_frame_check = frame
                            break
                    
                    if checkbox_frame_check:
                        try:
                            checked = await checkbox_frame_check.query_selector('span[aria-checked="true"]', timeout=1000)
                            if checked:
                                logger.info("✓✓✓ reCAPTCHA 已自动通过！")
                                return True
                        except:
                            pass
                    
                    # 重新获取挑战框（可能已分离）
                    challenge_frame = None
                    for frame in page.frames:
                        if "recaptcha" in frame.url and "bframe" in frame.url:
                            challenge_frame = frame
                            break
                    
                    if not challenge_frame:
                        logger.info("✓✓✓ reCAPTCHA 验证成功（挑战框已消失）！")
                        return True
                    
                    # 获取目标类型
                    try:
                        target_element = await challenge_frame.wait_for_selector('#rc-imageselect strong', timeout=5000)
                        target_text = await target_element.text_content()
                        target_num = get_target_num_from_text(target_text)
                        
                        if VERBOSE:
                            logger.info(f"  [{reload_count}/{max_reload}] 目标: {target_text} (编号: {target_num})")
                    except Exception as e:
                        logger.warning(f"  获取目标类型失败: {e}")
                        reload_btn = await challenge_frame.query_selector('#recaptcha-reload-button')
                        if reload_btn:
                            await reload_btn.click()
                            await asyncio.sleep(1)
                        continue
                    
                    # 如果是不支持的类型，重新加载
                    if target_num == 1000:
                        if VERBOSE:
                            logger.info("  跳过不支持的类型，重新加载...")
                        reload_btn = await challenge_frame.query_selector('#recaptcha-reload-button')
                        if reload_btn:
                            random_delay(mu=0.3, sigma=0.1)
                            await reload_btn.click()
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
                    
                    # ⚠️ 重要：不要用 requests 下载 img_urls（会 410 Gone 过期）
                    # 改用 Playwright 在浏览器上下文中直接截图网格
                    if "squares" in title_text.lower():
                        logger.info("  检测到 4x4 方格验证")
                        if not await screenshot_captcha_grid(challenge_frame, grid_size=4):
                            continue
                        answers = square_solver(target_num, VERBOSE, model)
                        captcha_type = "squares"
                    else:
                        if "none" in title_text.lower():
                            logger.info("  检测到 3x3 动态验证")
                            captcha_type = "dynamic"
                        else:
                            logger.info("  检测到 3x3 选择验证")
                            captcha_type = "selection"

                        if not await screenshot_captcha_grid(challenge_frame, grid_size=3):
                            continue
                        answers = dynamic_and_selection_solver(target_num, VERBOSE, model)
                    
                    # 检查识别结果
                    if captcha_type == "squares":
                        if len(answers) >= 1 and len(answers) < 16:
                            logger.info(f"  ✓ 识别成功，答案: {answers}")
                            break
                        else:
                            logger.warning(f"  ✗ 4x4 识别结果异常: {len(answers)} 个")
                            reload_btn = await challenge_frame.query_selector('#recaptcha-reload-button')
                            if reload_btn:
                                await reload_btn.click()
                    else:
                        if len(answers) >= 1:
                            logger.info(f"  ✓ 识别成功，答案: {answers}")
                            break
                        else:
                            logger.warning(f"  ✗ 3x3 识别结果不足: {len(answers)} 个")
                            reload_btn = await challenge_frame.query_selector('#recaptcha-reload-button')
                            if reload_btn:
                                await reload_btn.click()
                    
                    # 等待重载
                    await challenge_frame.wait_for_selector('#rc-imageselect-target td', timeout=5000)
                
                # 如果重载次数过多，跳出
                if reload_count >= max_reload:
                    logger.warning(f"  重载次数过多 ({max_reload})，跳过本次尝试")
                    continue
                
                # 开始点击答案
                if captcha_type == "dynamic":
                    # 动态验证：点击并等待新图片
                    logger.info("  开始动态验证流程...")
                    
                    # 等待第一个方格可点击（确保 DOM 准备好）
                    await challenge_frame.wait_for_selector('#rc-imageselect-target td:first-child', state='visible', timeout=10000)
                    
                    for answer in answers:
                        # 使用 XPath 直接定位第 N 个方格，类似 Selenium
                        cell_selector = f'#rc-imageselect-target td:nth-child({answer})'
                        try:
                            # 等待元素可见并点击
                            await challenge_frame.wait_for_selector(cell_selector, state='visible', timeout=10000)
                            cell = await challenge_frame.query_selector(cell_selector)
                            await cell.click(force=True, timeout=5000)
                        except:
                            # 备选：使用 JavaScript 点击
                            cell = await challenge_frame.query_selector(cell_selector)
                            if cell:
                                await cell.evaluate('el => el.click()')
                        random_delay(mu=0.6, sigma=0.3)  # 和本地版本一样的延迟
                    
                    # 持续处理新图片
                    dynamic_rounds = 0
                    max_dynamic_rounds = 15
                    
                    while dynamic_rounds < max_dynamic_rounds:
                        dynamic_rounds += 1
                        
                        # 等待新图片加载
                        before_img_urls = img_urls
                        
                        # 检测新图片
                        is_new = False
                        retry_detect = 0
                        while retry_detect < 20 and not is_new:
                            retry_detect += 1
                            await asyncio.sleep(0.3)
                            
                            new_img_urls = []
                            img_elements = await challenge_frame.query_selector_all('#rc-imageselect-target img')
                            for img in img_elements:
                                url = await img.get_attribute('src')
                                if url:
                                    new_img_urls.append(url)
                            
                            # 检查是否有新图片
                            index_common = []
                            for answer in answers:
                                if answer <= len(new_img_urls) and answer <= len(before_img_urls):
                                    if new_img_urls[answer-1] == before_img_urls[answer-1]:
                                        index_common.append(answer)
                            
                            if len(index_common) < 1:
                                is_new = True
                                img_urls = new_img_urls
                        
                        if not is_new:
                            logger.info(f"    [轮次 {dynamic_rounds}] 没有新图片，结束动态验证")
                            break
                        
                        # ⚠️ 动态验证：不要下载新图片 URL（会过期 410）
                        # 直接对发生变化的方格截图，然后拼回 0.png
                        for answer in answers:
                            await screenshot_captcha_tile(challenge_frame, answer, grid_size=3)

                        # 更新主图片（0.png）
                        try:
                            for answer in answers:
                                main_img = Image.open('0.png').convert('RGB')
                                new_img = Image.open(f"{answer}.png").convert('RGB')
                                paste_new_img_on_main_img(np.asarray(main_img), np.asarray(new_img), answer)
                            normalize_captcha_image('0.png', (300, 300))
                        except Exception as e:
                            logger.warning(f"    更新图片失败: {e}")
                            # 兜底：直接重新截图整个网格
                            await screenshot_captcha_grid(challenge_frame, grid_size=3)
                        
                        # 重新识别
                        answers = dynamic_and_selection_solver(target_num, VERBOSE, model)
                        
                        if len(answers) >= 1:
                            logger.info(f"    [轮次 {dynamic_rounds}] 检测到 {len(answers)} 个新目标")
                            for answer in answers:
                                cell_selector = f'#rc-imageselect-target td:nth-child({answer})'
                                try:
                                    await challenge_frame.wait_for_selector(cell_selector, state='visible', timeout=10000)
                                    cell = await challenge_frame.query_selector(cell_selector)
                                    await cell.click(force=True, timeout=5000)
                                except:
                                    cell = await challenge_frame.query_selector(cell_selector)
                                    if cell:
                                        await cell.evaluate('el => el.click()')
                                random_delay(mu=0.6, sigma=0.3)
                        else:
                            logger.info(f"    [轮次 {dynamic_rounds}] 未识别到更多目标，结束")
                            break
                
                elif captcha_type == "selection" or captcha_type == "squares":
                    # 一次性选择：直接点击所有答案
                    logger.info(f"  开始 {captcha_type} 验证流程...")
                    # 等待第一个方格可点击
                    await challenge_frame.wait_for_selector('#rc-imageselect-target td:first-child', state='visible', timeout=10000)
                    
                    for answer in answers:
                        cell_selector = f'#rc-imageselect-target td:nth-child({answer})'
                        try:
                            await challenge_frame.wait_for_selector(cell_selector, state='visible', timeout=10000)
                            cell = await challenge_frame.query_selector(cell_selector)
                            await cell.click(force=True, timeout=5000)
                        except:
                            cell = await challenge_frame.query_selector(cell_selector)
                            if cell:
                                await cell.evaluate('el => el.click()')
                        random_delay(mu=0.6, sigma=0.3)
                
                # 点击验证按钮
                verify_btn = await challenge_frame.query_selector('#recaptcha-verify-button')
                if verify_btn:
                    random_delay(mu=2, sigma=0.2)
                    await verify_btn.click()
                
                # 等待验证结果
                await asyncio.sleep(4)
                
                # 检查是否通过
                checkbox_frame = None
                for frame in page.frames:
                    if "recaptcha" in frame.url and "anchor" in frame.url:
                        checkbox_frame = frame
                        break
                
                if checkbox_frame:
                    try:
                        checked = await checkbox_frame.query_selector('span[aria-checked="true"]', timeout=2000)
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
                
                logger.info("  验证未通过，进入下一轮尝试...")
                
            except Exception as e:
                logger.error(f"  本轮尝试失败: {e}")
                import traceback
                traceback.print_exc()
                
                if outer_attempt >= max_attempts:
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
        # 说明：0.png 是验证码网格主图，GitHub Actions 调试时需要作为 artifact 上传，不能无条件删除。
        keep_captcha_images = os.environ.get('KEEP_CAPTCHA_IMAGES', 'true').strip().lower() in ('1', 'true', 'yes')
        if not keep_captcha_images:
            # 仅在明确要求清理时才删除 0.png
            try:
                os.remove('0.png')
            except:
                pass
        # 默认只清理单格截图 1~16
        for i in range(1, 17):
            try:
                os.remove(f"{i}.png")
            except:
                pass


def check_yolo_status():
    """检查 YOLO 状态并输出详细诊断信息"""
    logger.info("\n" + "=" * 70)
    logger.info("🔍 YOLO 环境检查")
    logger.info("=" * 70)
    
    logger.info(f"1️⃣ YOLO_AVAILABLE = {YOLO_AVAILABLE}")
    
    if not YOLO_AVAILABLE:
        logger.error("❌ YOLO 模块不可用 - 图形验证将被跳过！")
        logger.error("   请检查 ultralytics 是否正确安装")
        return False
    
    logger.info(f"2️⃣ 模型路径: {MODEL_PATH}")
    logger.info(f"3️⃣ 当前工作目录: {os.getcwd()}")
    
    # 下载模型文件（如果不存在）
    logger.info("4️⃣ 检查并下载模型文件...")
    if not download_yolo_model():
        logger.error("❌ 模型文件下载失败")
        return False
    
    file_size = os.path.getsize(MODEL_PATH)
    logger.info(f"✅ 模型文件就绪，大小: {file_size / (1024*1024):.2f} MB")
    
    # 尝试加载模型
    try:
        logger.info("5️⃣ 尝试加载 YOLO 模型...")
        test_model = YOLO(MODEL_PATH, task="detect")
        logger.info("✅ YOLO 模型加载成功！")
        logger.info("=" * 70 + "\n")
        return True
    except Exception as e:
        logger.error(f"❌ YOLO 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        logger.info("=" * 70 + "\n")
        return False


async def main():
    """主函数"""
    # 验证环境变量
    if not RENEW_URL:
        logger.error("❌ 错误: RENEW_URL 环境变量未设置")
        return
    
    # 检查 YOLO 状态
    yolo_ready = check_yolo_status()
    if not yolo_ready:
        logger.warning("⚠️ YOLO 未就绪，脚本将继续但可能无法通过图形验证")
    
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
    logger.info("🚀 启动 Camoufox 浏览器...")
    logger.info(f"   Headless 模式: {is_ci}")
    
    async with AsyncCamoufox(
        headless=is_ci,
        os=["windows"],
        screen=Screen(max_width=1920, max_height=1080),
    ) as browser:
        logger.info("✅ Camoufox 浏览器启动成功")
        
        page = await browser.new_page()
        logger.info("✅ 新页面创建成功")
        
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
                # 关键：不要假成功。reCAPTCHA 未通过就直接失败退出，保留截图/0.png 供分析。
                logger.error("❌ reCAPTCHA 未通过，终止流程（避免假成功）")
                await page.screenshot(path='host2play_error_recaptcha.png', full_page=True)
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: reCAPTCHA 未通过
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
                send_telegram_message(error_message, 'host2play_error_recaptcha.png')
                return

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

            # 二次校验：页面必须出现成功信号，否则按失败处理（避免假成功）
            page_text = (await page.inner_text('body')) if await page.query_selector('body') else ''
            text_l = page_text.lower()
            if ('success' not in text_l) and ('renewed' not in text_l) and ('续期' not in page_text and '成功' not in page_text):
                logger.error("❌ 未检测到成功文案，判定为失败（避免假成功）")
                await page.screenshot(path='host2play_error_no_success_text.png', full_page=True)
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: 未检测到成功提示文案
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
                send_telegram_message(error_message, 'host2play_error_no_success_text.png')
                return

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
