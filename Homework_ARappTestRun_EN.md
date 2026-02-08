![[Pasted image 20260207093703.png]]


## Question 1 

### Platform Information
**App/Platform Name:** Kivicube (Mizhi Technology WebAR Platform)  
**Official Website:** https://www.kivicube.com

---

### Step-by-Step Operation Procedure

#### Step 1: Asset Preparation and Upload (PC)
1. Visit the Kivicube official website and log in to the Creator Center via WeChat QR code scan
2. Download an open-source robot model (recommended: Sketchfab CC-licensed models such as "Mech Robot" or "Robot Boy")
3. Format check: Ensure the model is in `.glb` format (if not, use Blender to export with "+Y Up" coordinate system)
4. In the Kivicube console, click **"Create Scene"** → Select **"SLAM Spatial Positioning"** template
5. Upload the model file; the system will automatically apply Draco compression (reduces file size by 70%, suitable for mobile bandwidth)
![[Pasted image 20260207100030.png]]
![[Pasted image 20260207100016.png]]

#### Step 2: AR Scene Configuration (Web Editor)
1. **Initial Pose Settings**: Set the robot Scale to 0.4 (adapted for indoor desktop/floor scale)
2. **Interaction Configuration**: Add "Tap model to trigger zoom animation" (optional, for showcasing 3D details)
3. **Enable Environment Adaptation**: Check **"Plane Detection"** (Horizontal Plane) and **"Lighting Estimation"**
4. Click **"Publish"** → Generate experience QR code and short link
![[Pasted image 20260207100144.png]]
![[Pasted image 20260207100957.png]]

#### Step 3: Mobile Deployment and Debugging (Phone)
1. **Environment Scanning**: Open mobile browser (iOS Safari 13+ or Android Chrome 81+), scan the QR code to enter
2. **Permission Granting**: Allow camera and gyroscope permissions (critical: must click "Allow" rather than "Only While Using")
3. **Plane Detection Phase**: Slowly pan the phone to scan the ground until a **blue grid plane** appears on the screen (Kivicube's plane detection visualization)
4. **Place the Robot**: Tap the screen to lock the position; the robot model "drops" from the top of the screen onto the plane

---

### Required Hardware Sensors
Kivicube WebAR is based on the WebXR Device API and sensor fusion, requiring the following sensors to work together:

1. **RGB Camera (Main Camera)**

2. **IMU (Inertial Measurement Unit)**
   - **Gyroscope**: Measures angular velocity (rad/s), compensating for motion blur during rapid rotation
   - **Accelerometer**: Measures linear acceleration, providing gravity direction vector (g≈9.8m/s²), distinguishing up/down directions

3. **Ambient Light Sensor** (required for iOS devices)
   - **Function**: Captures ambient color temperature and light intensity (lux), enabling PBR material matching of the virtual robot surface with real-world lighting

---

### Core Algorithms

1. **Visual-Inertial Odometry (VIO)**
   - Based on the MSCKF (Multi-State Constraint Kalman Filter) framework, tightly coupled IMU pre-integration with visual feature point observations

2. **Plane Detection Algorithm**
   - RANSAC-based ground plane fitting: Randomly sample three points from the sparse point cloud to calculate the plane equation, counting the number of inliers

3. **Light Estimation**
   - Neural network-based HDR environment light reconstruction: Estimate main light source direction from a single frame
