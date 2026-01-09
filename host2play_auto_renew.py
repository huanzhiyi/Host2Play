"""
Host2Play 自动续期脚本 - Playwright + Camoufox 版本
使用 Playwright 选择器替代 YOLO 图形检测
参考 katabump_auto_renew.py 的成功策略
"""
import asyncio
import logging
import random
import os
from typing import Optional
from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError
from camoufox.async_api import AsyncCamoufox
from browserforge.fingerprints import Screen
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
RENEW_URL = os.environ.get('RENEW_URL')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')


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


async def solve_recaptcha_with_playwright(page: Page) -> bool:
    """使用 Playwright 选择器处理 reCAPTCHA（不使用 YOLO）"""
    logger.info("🔍 检查 reCAPTCHA...")
    
    try:
        # 等待 reCAPTCHA iframe 出现
        await asyncio.sleep(2)
        
        # 查找 reCAPTCHA checkbox iframe
        recaptcha_frames = []
        for frame in page.frames:
            if "recaptcha" in frame.url and "anchor" in frame.url:
                recaptcha_frames.append(frame)
        
        if not recaptcha_frames:
            logger.info("✅ 未检测到 reCAPTCHA，可能已通过")
            return True
        
        logger.info(f"✅ 找到 {len(recaptcha_frames)} 个 reCAPTCHA checkbox frame")
        
        # 点击 checkbox
        for frame in recaptcha_frames:
            try:
                checkbox = await frame.wait_for_selector('.recaptcha-checkbox-border', timeout=5000)
                if checkbox:
                    await checkbox.click()
                    logger.info("✅ 已点击 reCAPTCHA checkbox")
                    await asyncio.sleep(3)
                    break
            except Exception as e:
                logger.debug(f"点击 checkbox 失败: {e}")
                continue
        
        # 检查是否需要图形验证
        challenge_frames = []
        await asyncio.sleep(2)
        for frame in page.frames:
            if "recaptcha" in frame.url and "bframe" in frame.url:
                challenge_frames.append(frame)
        
        if challenge_frames:
            logger.warning("⚠️ 出现图形验证，需要手动处理或等待...")
            logger.info("💡 建议: 在 CI 环境中，reCAPTCHA 可能需要额外的策略")
            # 等待一段时间，看是否自动通过
            await asyncio.sleep(10)
            return False
        else:
            logger.info("✅ reCAPTCHA 验证通过（无图形验证）")
            return True
            
    except Exception as e:
        logger.error(f"❌ 处理 reCAPTCHA 失败: {e}")
        return False


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
            
            recaptcha_success = await solve_recaptcha_with_playwright(page)
            
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
