![[Pasted image 20260207093703.png]]


## Question 1 

### 平台信息
**App/平台名称：** Kivicube（弥知科技 WebAR 平台）  
**官网：** https://www.kivicube.com

---

### Step-by-Step 操作流程

#### Step 1：素材准备与上传（PC 端）
1. 访问 Kivicube 官网，使用微信扫码登录创作者中心
2. 下载开源机器人模型（推荐 Sketchfab CC 授权模型，如 "Mech Robot" 或 "Robot Boy"）
3. 格式检查：确保模型为 `.glb` 格式（若不为该格式，使用 Blender 导出时勾选 "+Y Up" 坐标系）
4. 在 Kivicube 控制台点击 **"创建场景"** → 选择 **"SLAM 空间定位"** 模板
5. 上传模型文件，系统自动完成 Draco 压缩（减小 70% 体积，适配移动端带宽）
![[Pasted image 20260207100030.png]]
![[Pasted image 20260207100016.png]]
#### Step 2：AR 场景配置（Web 端编辑器）
1. **初始位姿设置**：将机器人 Scale 设为 0.4（适应室内桌面/地面尺度）
2. **交互配置**：添加 "点击模型触发放大动画"（可选，用于展示 3D 细节）
3. **开启环境适配**：勾选 **"平面检测"**（Horizontal Plane）与 **"光照估计"**（Lighting Estimation）
4. 点击 **"发布"** → 生成体验二维码与短链接
![[Pasted image 20260207100144.png]]
![[Pasted image 20260207100957.png]]
#### Step 3：移动端部署与调试（手机端）
1. **环境扫描**：打开手机浏览器（iOS Safari 13+ 或 Android Chrome 81+），扫码进入
2. **权限授予**：允许摄像头与陀螺仪权限（关键：必须点击"允许"而非"仅在使用期间"）
3. **平面检测阶段**：缓慢平移手机扫描地面，直至屏幕出现 **蓝色网格平面**（Kivicube 的平面检测可视化）
4. **放置机器人**：点击屏幕锁定位置，机器人模型从屏幕顶部"降落"至平面

---

### 必需传感器（Hardware Sensors）
Kivicube WebAR 基于 WebXR Device API 与传感器融合，必需以下传感器协同：
1. **RGB 摄像头（主摄）**
2. **IMU（惯性测量单元）**
   - **陀螺仪（Gyroscope）**：测量角速度（rad/s），补偿快速旋转时的运动模糊
   - **加速度计（Accelerometer）**：测量线性加速度，提供重力方向向量（g≈9.8m/s²），区分上下方向
1. **环境光传感器（Ambient Light Sensor）**（iOS 设备必需）
   - **功能**：采集环境色温与光照强度（lux），实现虚拟机器人表面 PBR 材质与现实光照匹配
---
### 核心算法（Core Algorithms）

1. **视觉惯性里程计（Visual-Inertial Odometry, VIO）**
   - 基于 MSCKF（Multi-State Constraint Kalman Filter）框架，紧耦合 IMU 预积分与视觉特征点观测
2. **平面检测算法（Plane Detection）**
   - 基于 RANSAC 的地面平面拟合：从稀疏点云中随机采样三点计算平面方程，统计内点（inliers）数量
1. **光照估计（Light Estimation）**
   - 基于神经网络的 HDR 环境光重建：从单帧图像估计主光源方向
