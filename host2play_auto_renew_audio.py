"""
Host2Play 自动续期脚本 - Audio reCAPTCHA Solver 版本
- 使用 Playwright + Camoufox 过 Cloudflare
- 使用音频挑战 + 语音识别来解决 reCAPTCHA（参考 RecaptchaV2-Solver）
- 更适合在 GitHub Actions 等无头环境中运行

主要改进：
1. 使用音频挑战代替图像识别（避免图像检测失败）
2. 使用 Google Speech Recognition API 识别音频内容
3. 更稳定的 iframe 处理和错误恢复机制
"""
import asyncio
import logging
import random
import os
import io
import time
from typing import Optional, Dict
from datetime import datetime
import requests
import aiohttp
import speech_recognition as sr
from pydub import AudioSegment

from playwright.async_api import Page, Frame, TimeoutError as PlaywrightTimeoutError
from camoufox.async_api import AsyncCamoufox
from browserforge.fingerprints import Screen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
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
    """模拟人类点击行为"""
    target_x = x + random.uniform(-5, 5)
    target_y = y + random.uniform(-5, 5)
    
    await page.mouse.move(target_x, target_y, steps=random.randint(10, 25))
    await asyncio.sleep(random.uniform(0.1, 0.3))
    await page.mouse.down()
    await asyncio.sleep(random.uniform(0.05, 0.15))
    await page.mouse.up()


class AudioProcessor:
    """音频处理器 - 下载和识别音频"""
    
    def __init__(self, debug: bool = False):
        self.recognizer = sr.Recognizer()
        self.debug = debug
    
    async def download_audio(self, audio_url: str) -> bytes:
        """下载音频文件"""
        try:
            if self.debug:
                logger.info(f"  📥 下载音频: {audio_url[:100]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(audio_url) as response:
                    if response.status != 200:
                        raise Exception(f"音频下载失败，状态码: {response.status}")
                    audio_content = await response.read()
                    
                    if self.debug:
                        logger.info(f"  ✅ 音频下载成功: {len(audio_content)} bytes")
                    
                    return audio_content
        except Exception as e:
            logger.error(f"  ❌ 音频下载失败: {e}")
            raise
    
    def convert_to_wav(self, audio_content: bytes) -> io.BytesIO:
        """转换音频为 WAV 格式"""
        try:
            audio_bytes = io.BytesIO(audio_content)
            audio = AudioSegment.from_mp3(audio_bytes)
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            wav_bytes = io.BytesIO()
            audio.export(wav_bytes, format="wav", parameters=["-q:a", "0"])
            wav_bytes.seek(0)
            
            if self.debug:
                logger.info(f"  ✅ 音频转换成功")
            
            return wav_bytes
        except Exception as e:
            logger.error(f"  ❌ 音频转换失败: {e}")
            raise
    
    def recognize_audio(self, wav_bytes: io.BytesIO) -> str:
        """识别音频内容"""
        try:
            with sr.AudioFile(wav_bytes) as source:
                audio = self.recognizer.record(source)
                
                try:
                    text = str(self.recognizer.recognize_google(audio))
                    if self.debug:
                        logger.info(f"  🎤 识别结果（原始）: {text}")
                    
                    # 清理文本：只保留字母和数字
                    cleaned_text = ''.join(c.lower() for c in text if c.isalnum() or c.isspace())
                    if self.debug:
                        logger.info(f"  🎤 识别结果（清理）: {cleaned_text}")
                    
                    if not cleaned_text:
                        raise sr.UnknownValueError("识别结果为空")
                    
                    return cleaned_text.strip()
                    
                except sr.UnknownValueError:
                    raise Exception("无法理解音频内容")
                except sr.RequestError as e:
                    raise Exception(f"语音识别服务请求失败: {e}")
                    
        except Exception as e:
            logger.error(f"  ❌ 音频识别失败: {e}")
            raise
    
    async def process_audio(self, audio_url: str) -> str:
        """处理音频：下载、转换、识别"""
        audio_content = await self.download_audio(audio_url)
        
        # 在线程池中执行同步操作
        loop = asyncio.get_event_loop()
        wav_bytes = await loop.run_in_executor(None, self.convert_to_wav, audio_content)
        text = await loop.run_in_executor(None, self.recognize_audio, wav_bytes)
        
        return text


async def check_rate_limit(frame: Frame) -> bool:
    """检查是否被限流"""
    try:
        rate_limit = frame.locator(".rc-doscaptcha-header, .rc-doscaptcha-body, .rc-doscaptcha-header-text")
        rate_limit_text = await rate_limit.text_content(timeout=2000)
        if rate_limit_text and ("try again later" in rate_limit_text.lower() or 
                               "稍后再试" in rate_limit_text or
                               "unusual traffic" in rate_limit_text.lower()):
            return True
    except:
        pass
    return False


async def solve_recaptcha_audio(page: Page, max_attempts: int = 3) -> bool:
    """使用音频挑战解决 reCAPTCHA"""
    logger.info("🔍 开始处理 reCAPTCHA（音频方式）...")
    
    audio_processor = AudioProcessor(debug=VERBOSE)
    
    try:
        # 步骤 1: 查找并点击 reCAPTCHA checkbox
        # 增加随机延迟，避免被检测为机器人
        await asyncio.sleep(random.uniform(2.5, 4.0))
        
        checkbox_frame = None
        for frame in page.frames:
            if "recaptcha" in frame.url.lower() and "anchor" in frame.url.lower():
                checkbox_frame = frame
                break
        
        if not checkbox_frame:
            logger.warning("⚠️ 未找到 reCAPTCHA checkbox frame")
            return False
        
        logger.info("✓ 点击 reCAPTCHA checkbox...")
        try:
            checkbox = await checkbox_frame.wait_for_selector(
                '#recaptcha-anchor, .recaptcha-checkbox-border',
                timeout=10000
            )
            await asyncio.sleep(random.uniform(0.8, 1.5))
            await checkbox.click()
            logger.info("  ✅ Checkbox 已点击")
        except Exception as e:
            logger.error(f"  ❌ 点击 checkbox 失败: {e}")
            return False
        
        # 步骤 2: 等待挑战 iframe 出现（增加等待时间）
        await asyncio.sleep(random.uniform(4.0, 6.0))
        
        challenge_frame = None
        for frame in page.frames:
            if "recaptcha" in frame.url.lower() and "bframe" in frame.url.lower():
                challenge_frame = frame
                break
        
        if not challenge_frame:
            logger.info("✅ 无需挑战，checkbox 直接通过！")
            return True
        
        logger.info("✓ 检测到 reCAPTCHA 挑战，切换到音频模式...")
        
        # 步骤 3: 点击音频按钮
        for attempt in range(max_attempts):
            try:
                logger.info(f"\n=== 尝试 {attempt + 1}/{max_attempts} ===")
                
                # 第一次尝试前增加随机延迟
                if attempt == 0:
                    wait_time = random.uniform(1.5, 3.0)
                    logger.info(f"  ⏳ 等待 {wait_time:.1f} 秒后再操作...")
                    await asyncio.sleep(wait_time)
                
                # 检查是否被限流
                if await check_rate_limit(challenge_frame):
                    logger.error("❌ reCAPTCHA 已被限流，请稍后再试")
                    logger.warning("💡 建议：")
                    logger.warning("   1. 等待 15-30 分钟后重试")
                    logger.warning("   2. 使用不同的 IP 地址或代理")
                    logger.warning("   3. 避免短时间内多次尝试")
                    return False
                
                # 点击音频按钮
                try:
                    audio_button = await challenge_frame.wait_for_selector(
                        '#recaptcha-audio-button',
                        state='visible',
                        timeout=5000
                    )
                    # 人类化延迟
                    await asyncio.sleep(random.uniform(0.5, 1.2))
                    await audio_button.click()
                    logger.info("  ✅ 音频按钮已点击")
                    # 等待音频加载，增加时间
                    await asyncio.sleep(random.uniform(3.0, 5.0))
                except PlaywrightTimeoutError:
                    # 再次检查限流
                    if await check_rate_limit(challenge_frame):
                        logger.error("❌ reCAPTCHA 已被限流")
                        logger.warning("💡 建议：等待 15-30 分钟后重试")
                        return False
                    
                    logger.error("  ❌ 未找到音频按钮（可能已被限流）")
                    # 尝试截图看看当前状态
                    try:
                        await page.screenshot(path='host2play_audio_button_not_found.png', full_page=True)
                        logger.info("  📸 已保存截图: host2play_audio_button_not_found.png")
                    except:
                        pass
                    return False
                
                # 步骤 4: 获取音频下载链接
                try:
                    download_link = await challenge_frame.wait_for_selector(
                        '.rc-audiochallenge-tdownload-link',
                        state='visible',
                        timeout=15000  # 增加超时时间
                    )
                    audio_url = await download_link.get_attribute('href')
                    
                    if not audio_url:
                        logger.error("  ❌ 未获取到音频 URL")
                        continue
                    
                    logger.info(f"  ✅ 获取到音频 URL")
                    
                except PlaywrightTimeoutError:
                    logger.error("  ❌ 音频加载超时")
                    # 检查是否被限流
                    if await check_rate_limit(challenge_frame):
                        logger.error("❌ 音频加载失败：已被限流")
                        return False
                    continue
                
                # 步骤 5: 处理音频（下载、转换、识别）
                try:
                    logger.info("  🎤 开始处理音频...")
                    audio_text = await audio_processor.process_audio(audio_url)
                    logger.info(f"  ✅ 音频识别成功: {audio_text}")
                    
                except Exception as e:
                    logger.error(f"  ❌ 音频处理失败: {e}")
                    # 尝试重新加载音频挑战
                    try:
                        reload_button = await challenge_frame.query_selector('#recaptcha-reload-button')
                        if reload_button:
                            await reload_button.click()
                            await asyncio.sleep(2)
                    except:
                        pass
                    continue
                
                # 步骤 6: 输入识别结果
                try:
                    response_input = await challenge_frame.wait_for_selector(
                        '#audio-response',
                        state='visible',
                        timeout=5000
                    )
                    await response_input.fill(audio_text)
                    logger.info("  ✅ 已输入识别结果")
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"  ❌ 输入答案失败: {e}")
                    continue
                
                # 步骤 7: 点击验证按钮
                try:
                    verify_button = await challenge_frame.wait_for_selector(
                        '#recaptcha-verify-button',
                        state='visible',
                        timeout=5000
                    )
                    await verify_button.click()
                    logger.info("  ✅ 已点击验证按钮")
                    await asyncio.sleep(3)
                    
                except Exception as e:
                    logger.error(f"  ❌ 点击验证按钮失败: {e}")
                    continue
                
                # 步骤 8: 检查验证结果
                # 方法 1: 检查 checkbox 是否已选中
                checkbox_frame = None
                for frame in page.frames:
                    if "recaptcha" in frame.url.lower() and "anchor" in frame.url.lower():
                        checkbox_frame = frame
                        break
                
                if checkbox_frame:
                    try:
                        checked = await checkbox_frame.query_selector(
                            'span[aria-checked="true"]',
                            timeout=2000
                        )
                        if checked:
                            logger.info("✅✅✅ reCAPTCHA 验证成功！")
                            return True
                    except:
                        pass
                
                # 方法 2: 检查挑战框是否消失
                challenge_frame_check = None
                for frame in page.frames:
                    if "recaptcha" in frame.url.lower() and "bframe" in frame.url.lower():
                        challenge_frame_check = frame
                        break
                
                if not challenge_frame_check:
                    logger.info("✅✅✅ reCAPTCHA 验证成功（挑战框已消失）！")
                    return True
                
                # 方法 3: 检查是否有错误提示
                try:
                    error_msg = await challenge_frame.query_selector('.rc-audiochallenge-error-message')
                    if error_msg:
                        is_visible = await error_msg.is_visible()
                        if is_visible:
                            error_text = await error_msg.text_content()
                            logger.warning(f"  ⚠️ 验证失败: {error_text}")
                            
                            # 如果是"incorrect"，尝试重新获取音频
                            if "incorrect" in error_text.lower() or "multiple" in error_text.lower():
                                logger.info("  🔄 答案不正确，重新尝试...")
                                try:
                                    reload_button = await challenge_frame.query_selector('#recaptcha-reload-button')
                                    if reload_button:
                                        await reload_button.click()
                                        await asyncio.sleep(2)
                                except:
                                    pass
                                continue
                except:
                    pass
                
                logger.warning("  ⚠️ 验证未通过，继续尝试...")
                
            except Exception as e:
                logger.error(f"  ❌ 尝试过程出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        logger.warning(f"✗ 达到最大尝试次数 ({max_attempts})，验证失败")
        return False
        
    except Exception as e:
        logger.error(f"❌ reCAPTCHA 音频解决失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def find_and_click_turnstile(page: Page, retries: int = 20) -> bool:
    """查找并点击 Cloudflare Turnstile 验证框"""
    logger.info("🔍 寻找 Turnstile 验证框...")
    
    for attempt in range(retries):
        try:
            turnstile_frames = []
            
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
            
            for frame in turnstile_frames:
                try:
                    frame_element = await frame.frame_element()
                    is_visible = await frame_element.is_visible()
                    
                    if not is_visible:
                        continue
                    
                    box = await frame_element.bounding_box()
                    if not box:
                        continue
                    
                    click_x = box['x'] + box['width'] / 2
                    click_y = box['y'] + box['height'] / 2
                    
                    await human_click(page, click_x, click_y)
                    logger.info(f"✅ 已点击 Turnstile 验证框")
                    
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


async def main():
    """主函数"""
    if not RENEW_URL:
        logger.error("❌ 错误: RENEW_URL 环境变量未设置")
        return
    
    renew_url = RENEW_URL
    
    print("="*70)
    print("  🔐 Host2Play 自动续期脚本 (Audio Solver 版 v2.1)")
    print(f"  🌐 续期 URL: {renew_url[:50]}...")
    print("  🤖 模式: Playwright + Camoufox + Audio reCAPTCHA")
    print("  ⚡ 改进: 增强的限流检测和人类化行为")
    print("="*70)
    print()
    
    start_time = datetime.now()
    start_message = f"""🚀 *Host2Play 自动续期开始*

🕐 时间: `{start_time.strftime('%Y-%m-%d %H:%M:%S')}`
🤖 模式: Audio reCAPTCHA Solver

⏳ 正在处理中..."""
    send_telegram_message(start_message)
    
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    
    if is_ci:
        logger.info("🤖 检测到 CI 环境，使用 headless 模式")
    
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
            
            page_content = await page.content()
            
            if 'cloudflare' in page_content.lower() or 'turnstile' in page_content.lower():
                logger.info("⚠️ 检测到 Cloudflare 保护，尝试处理...")
                
                success = await find_and_click_turnstile(page)
                
                if success:
                    logger.info("✅ Turnstile 验证已完成")
                    await asyncio.sleep(3)
                else:
                    logger.warning("⚠️ Turnstile 自动处理失败，等待自动通过...")
                    await asyncio.sleep(10)
            else:
                logger.info("✅ 未检测到 Cloudflare 保护")
            
            await page.screenshot(path='host2play_01_after_load.png', full_page=True)
            logger.info("📸 截图保存: host2play_01_after_load.png")
            
            # Step 3: 查找并点击 Renew 按钮
            logger.info("\n[3/4] 🖱️ 查找并点击 'Renew' 按钮...")
            await asyncio.sleep(2)
            
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
            
            await renew_button.click()
            logger.info("✅ 已点击 Renew 按钮")
            await asyncio.sleep(3)
            
            await page.screenshot(path='host2play_02_after_button.png', full_page=True)
            logger.info("📸 截图保存: host2play_02_after_button.png")
            
            # Step 4: 处理 reCAPTCHA（音频方式）
            logger.info("\n[4/4] 🔐 处理 reCAPTCHA（音频方式）...")
            logger.info("💡 提示：使用音频验证避免图像识别问题")
            logger.info("⏰ 此过程可能需要 10-30 秒，请耐心等待...")
            
            recaptcha_success = await solve_recaptcha_audio(page)
            
            if not recaptcha_success:
                logger.error("❌ reCAPTCHA 未通过")
                await page.screenshot(path='host2play_error_recaptcha.png', full_page=True)
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: reCAPTCHA 音频验证未通过
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
                send_telegram_message(error_message, 'host2play_error_recaptcha.png')
                return
            
            # 查找并点击弹窗内的确认按钮
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
            
            await page.screenshot(path='host2play_renew_success.png', full_page=True)
            logger.info("📸 最终截图: host2play_renew_success.png")
            
            # 验证成功
            page_text = (await page.inner_text('body')) if await page.query_selector('body') else ''
            text_l = page_text.lower()
            
            if ('success' in text_l) or ('renewed' in text_l) or ('续期' in page_text and '成功' in page_text):
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                success_message = f"""✅ *Host2Play 续期成功*

🕐 开始时间: `{start_time.strftime('%Y-%m-%d %H:%M:%S')}`
🕐 结束时间: `{end_time.strftime('%Y-%m-%d %H:%M:%S')}`
⏱️ 耗时: `{duration:.1f}` 秒
🤖 方法: Audio reCAPTCHA Solver

✨ 续期操作已完成！"""
                send_telegram_message(success_message, 'host2play_renew_success.png')
                
                logger.info("\n" + "="*70)
                logger.info("  ✅✅✅ 续期成功！")
                logger.info(f"  ⏱️  耗时: {duration:.1f} 秒")
                logger.info("="*70)
            else:
                logger.error("❌ 未检测到成功文案")
                error_message = f"""⚠️ *Host2Play 续期状态未知*

❗ 未检测到明确的成功提示
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

请手动检查续期状态。"""
                send_telegram_message(error_message, 'host2play_renew_success.png')
                
        except Exception as e:
            logger.error(f"❌ 执行过程中出错: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                await page.screenshot(path='host2play_error.png', full_page=True)
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: {str(e)[:200]}
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
"""
                send_telegram_message(error_message, 'host2play_error.png')
            except:
                pass


if __name__ == "__main__":
    asyncio.run(main())
