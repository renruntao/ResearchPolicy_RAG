# 智能政策问答系统实例

支持对国家自然科学基金相关政策文件进行智能化解读问答并提供回答的政策来源。

## 系统演示

https://github.com/user-attachments/assets/bdd0415b-0b88-4d8b-833b-e49e3f97e685

## 功能特点

- 📚 支持多个PDF文档上传和智能分析
- 💡 基于AI的智能问答功能
- 👀 实时PDF预览功能
- 📍 精确显示答案来源及页码
- 💭 支持聊天历史记录
- ⚙️ 可自定义文档分块大小和检索数量

## 环境要求

- Python 3.8+
- OpenAI API密钥

## 安装步骤

1. 克隆仓库
2. 安装依赖: pip install requirements.txt
3. 配置OpenAI API密钥
4. 运行程序

## 环境配置
1. 复制 `.env.example` 文件并重命名为 `.env`

2. 在 `.env` 文件中填入您的 API 配置：

   ```

   OPENAI_API_KEY=your_api_key_here

   OPENAI_API_BASE=your_api_base_url_here
