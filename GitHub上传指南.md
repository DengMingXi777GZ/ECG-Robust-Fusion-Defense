# GitHub 仓库创建与上传指南

## 一、创建 GitHub 仓库

### 步骤 1: 登录 GitHub
1. 打开 [https://github.com](https://github.com)
2. 登录您的账号

### 步骤 2: 创建新仓库
1. 点击右上角 **+** 号 → **New repository**
2. 填写仓库信息：
   - **Repository name**: `ECG-Adversarial-Defense` (建议)
   - **Description**: `Three-layer defense system against adversarial attacks on ECG deep learning models`
   - **Public** / **Private**: 选择 Public (或 Private)
   - **Initialize this repository with**: 
     - ☑️ Add a README
     - ☑️ Add .gitignore (选择 Python)
     - ☑️ Choose a license (选择 MIT)
3. 点击 **Create repository**

---

## 二、本地项目准备

### 步骤 1: 检查当前目录
```bash
cd E:\Code\Master
ls  # 确认项目文件在此目录
```

### 步骤 2: 初始化 Git 仓库
```bash
# 在项目根目录执行
git init
```

### 步骤 3: 配置 Git (如未配置)
```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 三、准备上传文件

### 已创建的文件
以下文件已创建，可直接使用：

| 文件 | 作用 |
|------|------|
| `.gitignore` | 忽略不需要上传的文件 (模型权重、数据等) |
| `README_GitHub.md` | GitHub 仓库主页说明 |

### 需要重命名 README
```bash
# 将 README_GitHub.md 重命名为 README.md
mv README_GitHub.md README.md
```

---

## 四、上传步骤

### 方法一: 命令行上传 (推荐)

#### 步骤 1: 添加远程仓库
```bash
# 替换 YOUR_USERNAME 为您的 GitHub 用户名
git remote add origin https://github.com/YOUR_USERNAME/ECG-Adversarial-Defense.git
```

#### 步骤 2: 添加文件到暂存区
```bash
# 添加所有文件
git add .

# 或逐个添加关键文件
git add README.md
git add .gitignore
git add requirements.txt
git add attacks/
git add defense/
git add models/
git add features/
git add data/
git add evaluation/
git add visualization/
git add *.py
git add *.md
```

#### 步骤 3: 提交更改
```bash
git commit -m "Initial commit: ECG Adversarial Defense System

- Layer 1: Attack algorithms (FGSM, PGD, SAP)
- Layer 2: Defense training (AT, NSR)
- Layer 3: Feature fusion with handcrafted features
- Adversarial detector with AUC-ROC 0.92
- Comprehensive evaluation and visualization tools"
```

#### 步骤 4: 推送到 GitHub
```bash
# 如果远程仓库已有内容，先拉取
git pull origin main --allow-unrelated-histories

# 推送
git push -u origin main

# 如果默认分支是 master
git push -u origin master
```

---

### 方法二: GitHub Desktop (图形界面)

1. 下载安装 [GitHub Desktop](https://desktop.github.com/)
2. 登录 GitHub 账号
3. File → Add local repository
4. 选择项目文件夹 `E:\Code\Master`
5. 填写提交信息，点击 **Commit to main**
6. 点击 **Publish repository**

---

### 方法三: VSCode 上传

1. 打开 VSCode
2. 打开项目文件夹
3. 点击左侧 **源代码管理** 图标
4. 点击 **+** 号暂存所有更改
5. 填写提交信息，点击 **提交**
6. 点击 **发布到 GitHub**

---

## 五、大文件处理 (可选)

如果模型权重文件 (>100MB) 需要上传，使用 Git LFS:

```bash
# 安装 Git LFS
git lfs install

# 追踪大文件
git lfs track "*.pth"
git lfs track "data/*.npy"

# 提交 .gitattributes
git add .gitattributes
git commit -m "Add Git LFS for large files"
```

---

## 六、验证上传

### 检查 GitHub 仓库
1. 打开 `https://github.com/YOUR_USERNAME/ECG-Adversarial-Defense`
2. 确认文件都已上传
3. 检查 README 是否正常显示

### 本地测试克隆
```bash
# 在另一个文件夹测试
cd /tmp
git clone https://github.com/YOUR_USERNAME/ECG-Adversarial-Defense.git
cd ECG-Adversarial-Defense
ls
```

---

## 七、后续更新

### 日常更新流程
```bash
# 查看更改
git status

# 添加更改的文件
git add filename

# 提交
git commit -m "Update description"

# 推送
git push
```

### 添加标签 (版本发布)
```bash
# 创建标签
git tag -a v1.0 -m "Version 1.0: Complete three-layer defense system"

# 推送标签
git push origin v1.0
```

---

## 八、常见问题

### Q1: 提示 "fatal: not a git repository"
```bash
# 确保在项目根目录执行
cd E:\Code\Master
git init
```

### Q2: 提示 "Permission denied"
```bash
# 使用 SSH 或检查权限
# 生成 SSH 密钥
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"

# 复制公钥到 GitHub Settings -> SSH and GPG keys
cat ~/.ssh/id_rsa.pub
```

### Q3: 提示 "failed to push some refs"
```bash
# 先拉取远程更改
git pull origin main --rebase

# 再推送
git push
```

### Q4: 如何忽略已追踪的文件
```bash
# 停止追踪模型权重
git rm --cached checkpoints/*.pth
git commit -m "Stop tracking model weights"
```

---

## 九、仓库美化建议

### 添加徽章 (Badges)
在 README.md 顶部添加：
```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
```

### 添加项目结构图
```markdown
```
📦 ECG-Adversarial-Defense
├── 📁 attacks/          # 攻击算法
├── 📁 defense/          # 防御训练
├── 📁 models/           # 模型定义
├── 📁 features/         # 特征工程
├── 📁 data/             # 数据加载
├── 📁 evaluation/       # 评估分析
└── 📁 visualization/    # 可视化
```
```

---

## 十、完整命令速查

```bash
# 1. 初始化
git init

# 2. 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/ECG-Adversarial-Defense.git

# 3. 添加文件
git add .

# 4. 提交
git commit -m "Initial commit"

# 5. 推送
git push -u origin main

# 6. 后续更新
git add .
git commit -m "Update"
git push
```

---

**完成!** 🎉

您的项目现在应该在 GitHub 上可访问：
`https://github.com/YOUR_USERNAME/ECG-Adversarial-Defense`
