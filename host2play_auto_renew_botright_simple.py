"""
Host2Play 自动续期脚本 - Botright 简化版
- 使用 Botright（比 Camoufox 更强的反检测）
- 使用 playwright-recaptcha 音频识别
- 免费方案，无需付费服务

版本: v3.0-simple
更新: 2026-01-10
"""
import asyncio
import logging
import random
import os
from typing import Optional
from datetime import datetime
import requests

try:
    import botright
    from botright.playwright_mock import Page
    BOTRIGHT_AVAILABLE = True
except ImportError:
    BOTRIGHT_AVAILABLE = False
    print("⚠️ 警告: botright 未安装，请运行: pip install botright")

try:
    from playwright_recaptcha import recaptchav2
    PLAYWRIGHT_RECAPTCHA_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_RECAPTCHA_AVAILABLE = False
    print("⚠️ 警告: playwright-recaptcha 未安装，请运行: pip install playwright-recaptcha")

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


async def solve_recaptcha_with_audio(page: Page) -> bool:
    """使用 playwright-recaptcha 音频识别"""
    if not PLAYWRIGHT_RECAPTCHA_AVAILABLE:
        logger.error("❌ playwright-recaptcha 未安装")
        return False
    
    logger.info("🎤 使用音频识别方法解决 reCAPTCHA...")
    
    try:
        async with recaptchav2.AsyncSolver(page) as solver:
            token = await solver.solve_recaptcha(wait=True, wait_timeout=90)
            
            if token:
                logger.info(f"✅ reCAPTCHA 音频识别成功！")
                return True
            else:
                logger.warning("⚠️ 音频识别返回空结果")
                return False
                
    except Exception as e:
        error_msg = str(e).lower()
        
        if 'rate limit' in error_msg or 'try again later' in error_msg:
            logger.error("❌ 音频识别被限流")
            logger.warning("💡 建议：")
            logger.warning("   1. 等待 15-30 分钟后重试")
            logger.warning("   2. 降低运行频率（每3天或每周）")
            logger.warning("   3. 使用不同的 IP 地址或代理")
        elif 'timeout' in error_msg:
            logger.error("❌ 音频识别超时")
        else:
            logger.error(f"❌ 音频识别失败: {e}")
        
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
    
    if not BOTRIGHT_AVAILABLE:
        logger.error("❌ 错误: botright 未安装")
        logger.info("💡 请运行: pip install botright")
        return
    
    if not PLAYWRIGHT_RECAPTCHA_AVAILABLE:
        logger.error("❌ 错误: playwright-recaptcha 未安装")
        logger.info("💡 请运行: pip install playwright-recaptcha")
        return
    
    renew_url = RENEW_URL
    
    print("="*70)
    print("  🔐 Host2Play 自动续期脚本 (Botright 简化版)")
    print(f"  🌐 续期 URL: {renew_url[:50]}...")
    print("  🤖 模式: Botright + 音频识别（免费方案）")
    print("  ⚡ 反检测: Botright 增强")
    print("="*70)
    print()
    
    start_time = datetime.now()
    start_message = f"""🚀 *Host2Play 自动续期开始*

🕐 时间: `{start_time.strftime('%Y-%m-%d %H:%M:%S')}`
🤖 模式: Botright + 音频识别

⏳ 正在处理中..."""
    send_telegram_message(start_message)
    
    is_ci = os.environ.get('CI') == 'true' or os.environ.get('GITHUB_ACTIONS') == 'true'
    
    if is_ci:
        logger.info("🤖 检测到 CI 环境，使用 headless 模式")
    
    logger.info("🚀 启动 Botright 浏览器...")
    logger.info(f"   Headless 模式: {is_ci}")
    
    try:
        botright_client = await botright.Botright(headless=is_ci)
        browser = await botright_client.new_browser()
        page = await browser.new_page()
        
        logger.info("✅ Botright 浏览器启动成功")
        
        try:
            # Step 1: 访问续期页面
            logger.info("\n[1/4] 🌐 访问续期页面...")
            await page.goto(renew_url, wait_until='domcontentloaded')
            await asyncio.sleep(random.uniform(2, 4))
            
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
            await asyncio.sleep(random.uniform(1, 2))
            
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
            await asyncio.sleep(random.uniform(2, 4))
            
            await page.screenshot(path='host2play_02_after_button.png', full_page=True)
            logger.info("📸 截图保存: host2play_02_after_button.png")
            
            # Step 4: 处理 reCAPTCHA
            logger.info("\n[4/4] 🔐 检测并处理 reCAPTCHA...")
            logger.info("💡 提示：使用音频识别方式")
            logger.info("⏰ 此过程可能需要 30-60 秒，请耐心等待...")
            await asyncio.sleep(random.uniform(2, 3))
            
            recaptcha_success = await solve_recaptcha_with_audio(page)
            
            if not recaptcha_success:
                logger.error("❌ reCAPTCHA 解决失败")
                await page.screenshot(path='host2play_error_recaptcha.png', full_page=True)
                
                error_message = f"""❌ *Host2Play 续期失败*

❗ 错误: reCAPTCHA 验证失败
🕐 时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`

💡 建议:
• 等待 15-30 分钟后重试
• 降低运行频率（每3天或每周）
• 考虑使用代理
"""
                send_telegram_message(error_message, 'host2play_error_recaptcha.png')
                return
            
            # 等待提交完成
            logger.info("⏳ 等待提交完成...")
            await asyncio.sleep(3)
            
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
🤖 方法: Botright + 音频识别

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
        
        finally:
            await page.close()
            await browser.close()
            await botright_client.close()
            
    except Exception as e:
        logger.error(f"❌ 浏览器启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
