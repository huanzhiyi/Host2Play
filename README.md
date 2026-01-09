# 🤖 Host2Play 自动续期 - GitHub Actions

通过 GitHub Actions 自动续期 Host2Play 免费服务器，无需手动操作。

## ✨ 特性

- ✅ 自动定时续期（每天执行）
- ✅ 支持手动触发
- ✅ 自动保存成功截图
- ✅ 失败时保存日志便于调试
- ✅ 完全在云端运行，无需本地环境

## 🚀 快速配置（3 步）

### 步骤 1: 准备模型文件

将 reCAPTCHA 识别模型文件放入仓库**根目录**：

```bash
# 将 model.onnx (260+ MB) 放到仓库根目录

# 使用 Git LFS 管理大文件
git lfs install
git lfs track "model.onnx"
git add .gitattributes model.onnx
git commit -m "Add reCAPTCHA model"
git push
```

### 步骤 2: 配置 Secrets

进入仓库 Settings → Secrets and variables → Actions → New repository secret

**必需配置**：
- **名称**: `RENEW_URL`
- **值**: `https://host2play.gratis/server/renew?i=YOUR_SERVER_ID`

**可选配置（Telegram 通知）**：
- **名称**: `TELEGRAM_BOT_TOKEN`
- **值**: 你的 Telegram Bot Token
- **名称**: `TELEGRAM_CHAT_ID`
- **值**: 你的 Telegram Chat ID

> 💡 **如何获取续期链接？**
> 1. 登录 Host2Play
> 2. 进入服务器管理页面
> 3. 右键点击 "Renew" 按钮，复制链接地址

> 💡 **如何获取 Telegram 配置？**
> 1. 与 @BotFather 对话创建 Bot，获取 Token
> 2. 与你的 Bot 对话，发送任意消息
> 3. 访问 `https://api.telegram.org/bot<TOKEN>/getUpdates` 获取 Chat ID

### 步骤 3: 启用 Actions

1. 进入仓库 **Actions** 标签
2. 启用 workflows
3. 选择 **"Host2Play Auto Renew"**
4. 点击 **"Run workflow"** 手动触发测试

## 📅 执行计划

默认每天 UTC 02:00 (北京时间 10:00) 自动执行。

**修改执行时间：** 编辑 `.github/workflows/host2play_auto_renew.yml`

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # 每天 02:00 UTC
```

常用 cron 表达式：
- `0 */12 * * *` - 每 12 小时
- `0 0 * * 0` - 每周日
- `0 0 1 * *` - 每月 1 号

## 📥 查看结果

### 成功时
1. 进入 Actions → 选择运行记录
2. 在 **Artifacts** 下载截图
3. 查看 `host2play_renew_success.png`

### 失败时
1. 查看运行日志
2. 下载 Artifacts 中的日志文件
3. 参考故障排查指南

## 🔧 故障排查

### ❌ 模型文件未找到

```bash
# 确保模型文件已提交到根目录
ls -lh model.onnx
# 应显示 260+ MB
```

### ❌ Secret 未配置

检查：Settings → Secrets and variables → Actions → Repository secrets

确保 `RENEW_URL` 存在且格式正确。

### ❌ reCAPTCHA 验证失败

- GitHub Actions IP 可能被标记
- 多次尝试手动触发
- 检查模型文件是否完整

## 📖 详细文档

查看完整配置和高级用法：[HOST2PLAY_GITHUB_ACTIONS_SETUP.md](HOST2PLAY_GITHUB_ACTIONS_SETUP.md)

## 🔔 通知配置（可选）

### 邮件通知

GitHub 默认在 workflow 失败时发送邮件到你的注册邮箱。

### Telegram 通知

在 workflow 文件中添加 Telegram 通知步骤，配置：
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

详见：[通知配置](HOST2PLAY_GITHUB_ACTIONS_SETUP.md#-通知配置可选)

## 💡 本地测试

```bash
# 设置环境变量
export RENEW_URL="https://host2play.gratis/server/renew?i=YOUR_ID"
export HEADLESS="true"

# 运行脚本
python host2play_auto_renew.py
```

## 📊 使用配额

**GitHub Actions 免费配额：**
- 公共仓库：无限
- 私有仓库：每月 2000 分钟

本脚本每次约 2-5 分钟，每天一次约 150 分钟/月，远低于配额。

## 🔒 安全提示

- ✅ 使用 GitHub Secrets 保护敏感信息
- ✅ 建议使用私有仓库
- ✅ 不要在代码中硬编码续期链接
- ✅ 定期检查运行日志

## ❓ 常见问题

**Q: 多久续期一次？**  
A: Host2Play 通常 7-30 天需要续期。建议每天执行以确保不过期。

**Q: 可以续期多个服务器吗？**  
A: 可以。创建多个 workflow 文件或添加多个步骤，使用不同的 `RENEW_URL`。

**Q: 失败了怎么办？**  
A: 查看 Actions 日志，下载失败时的 Artifacts，根据错误信息排查。

## 📝 更新日志

- **v1.1.0** (2026-01-09)
  - 添加 GitHub Actions 支持
  - 支持环境变量配置
  - Headless 模式优化
  - 只保存成功截图
  - 等待页面完全加载后截图

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**开始使用 GitHub Actions 自动续期！** 🚀
