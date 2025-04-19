import dxcam
import ctypes
import win32api
import win32gui
import ctypes
def get_real_window_rect(hwnd):
    # 获取系统DPI缩放
    user32 = ctypes.windll.user32
    dpi = user32.GetDpiForWindow(hwnd)
    scale = dpi / 96.0
async def processing_pipeline(self):
    while True:
        frame = await self.capturer.get_frame_async()
        tasks = [
            self.detect_balls(frame),
            self.calculate_paths(frame),
            self.update_ui(frame)
        ]
        await asyncio.gather(*tasks)
def auto_calibrate(self):
    # 采集多帧样本
    samples = [capturer.get_frame() for _ in range(30)]
    
    # 自动计算颜色阈值
    hsv_values = np.concatenate([frame.hsv for frame in samples])
    self.lower_thresh = np.percentile(hsv_values, 5, axis=0)
    self.upper_thresh = np.percentile(hsv_values, 95, axis=0)
    
    # 自动调整环形半径
    positions = [detect_center(frame) for frame in samples]
    self.center = np.median(positions, axis=0)
class GPUPool:
    def __init__(self):
        self.streams = [cuda.Stream() for _ in range(4)]
        self.current = 0
        
    def get_stream(self):
        stream = self.streams[self.current % 4]
        self.current +=1
        return stream 
    # 获取窗口真实坐标
    rect = win32gui.GetWindowRect(hwnd)
    return (
        int(rect[0] * scale),
        int(rect[1] * scale),
        int(rect[2] * scale),
        int(rect[3] * scale)
    )

def find_target_window(title_part):
    def callback(hwnd, ctx):
        if title_part.lower() in win32gui.GetWindowText(hwnd).lower():
            ctx.append(hwnd)
        return True
    
    matched = []
    win32gui.EnumWindows(callback, matched)
    return matched[0] if matched else None

def get_real_resolution():
    # 获取物理分辨率
    hDC = ctypes.windll.user32.GetDC(0)
    width = ctypes.windll.gdi32.GetDeviceCaps(hDC, 8)  # HORZRES
    height = ctypes.windll.gdi32.GetDeviceCaps(hDC, 10) # VERTRES
    ctypes.windll.user32.ReleaseDC(0, hDC)
    
    # 获取缩放比例
    scale = win32api.GetDpiForSystem() / 96
    return (int(width*scale), int(height*scale))

class DebugOverlay:
    def __init__(self):
        self.window = cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
        self.overlay = None
        
    def update_display(self, frame, balls, paths):
        # 在CPU端绘制调试信息
        cpu_frame = frame.download()
        for (x,y), _ in balls:
            cv2.circle(cpu_frame, (x,y), 5, (0,255,0), -1)
        
        # 绘制路径规划曲线
        if len(paths) >=2:
            cv2.polylines(cpu_frame, [np.array(paths)], False, (255,0,0), 2)
            
        cv2.imshow("Debug", cpu_frame)
        cv2.waitKey(1)

class FallbackSystem:
    def __init__(self):
        self.normal_mode = True
        self.fallback_counter = 0
        
    def handle_exception(self, e):
        if isinstance(e, GraphicsTimeoutError):
            self._switch_to_software_rendering()
        elif isinstance(e, CUDAMemoryError):
            self._release_gpu_resources()
            self._enable_cpu_fallback()
            
    def _enable_cpu_fallback(self):
        # 切换到CPU处理流程
        self.color_detector = CPUColorDetector()
        self.mask_generator = CPUMaskGenerator()
        self.normal_mode = False
        
class CPUColorDetector:  # 备用CPU处理类
    def detect(self, frame):
        # 使用传统图像处理方法
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, ...)

class VisionPipeline:
    def __init__(self):
        self.hsv_converter = cv2.cuda.cvtColor
        self.thresholder = cv2.cuda.inRange
        self.morphology = cv2.cuda.createMorphologyFilter(
            cv2.MORPH_CLOSE, CV_8UC1, np.ones((3,3), np.uint8))
        
    def process_frame(self, gpu_frame):
        # GPU上完成全部处理
        gpu_hsv = self.hsv_converter(gpu_frame, cv2.COLOR_BGR2HSV)
        gpu_mask = self.thresholder(gpu_hsv, lowerb, upperb)
        gpu_processed = self.morphology.apply(gpu_mask)
        return cv2.cuda.findContours(gpu_processed, ...)
    
    __global__ void fast_color_convert(half* src, half* dst) {
    // 利用Tensor Core加速计算
    asm("vcvt.pack.sat.d8.f16.f16 %0, %1, %2;" ...)
}
    
async def processing_pipeline(self):
    while True:
        frame = await self.capturer.get_frame_async()
        tasks = [
            self.detect_balls(frame),
            self.calculate_paths(frame),
            self.update_ui(frame)
        ]
        await asyncio.gather(*tasks)

def auto_calibrate(self):
    # 采集多帧样本
    samples = [capturer.get_frame() for _ in range(30)]
    
    # 自动计算颜色阈值
    hsv_values = np.concatenate([frame.hsv for frame in samples])
    self.lower_thresh = np.percentile(hsv_values, 5, axis=0)
    self.upper_thresh = np.percentile(hsv_values, 95, axis=0)
    
    # 自动调整环形半径
    positions = [detect_center(frame) for frame in samples]
    self.center = np.median(positions, axis=0)

class GPUPool:
    def __init__(self):
        self.streams = [cuda.Stream() for _ in range(4)]
        self.current = 0
        
    def get_stream(self):
        stream = self.streams[self.current % 4]
        self.current +=1
        return stream

class AdaptiveSampler:
    def __init__(self):
        self.prev_frame = None
        self.sample_rate = 60  # 默认全速
        
    def check_motion(self, current_frame):
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return False
            
        # 使用GPU差分计算
        diff = cv2.cuda.absdiff(current_frame, self.prev_frame)
        changed_pixels = cv2.cuda.sum(diff)[0] / 255
        motion_level = changed_pixels / (diff.cols * diff.rows)
        
        # 动态调整采样率
        if motion_level > 0.01:
            self.sample_rate = 60
        else:
            self.sample_rate = max(10, self.sample_rate//2)
            
        self.prev_frame = current_frame.clone()
        return motion_level > 0.01
    
class ScreenCapturer:
    def __init__(self):
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.camera.start(target_fps=60, video_mode=True)
        
    def get_frame(self):
        # 获取带NVIDIA CUDA共享资源的帧
        frame = self.camera.get_latest_frame()
        return cv2.cuda_GpuMat(frame)  # 直接传输到GPU内存
    
    def detect_color_balls(img):
    # HSV颜色空间阈值处理
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # 形态学优化+轮廓检测
    kernel = np.ones((3,3),np.uint8)
    processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 亚像素级中心点定位
    contours = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = [((cX, cY), cv2.contourArea(c)) for c in contours if area > min_size]
    return polar_sort(balls)  # 极坐标排序

class RadarScanner:
    def __init__(self, center):
        self.center = center
        self.ring_step = 20  # 像素半径步进
        
    def generate_rings(self):
        max_r = max_distance(self.center)
        return [RingLayer(r) for r in range(0, max_r, self.ring_step)]
    
    def smooth_swipe(start, end):
        distance = calc_distance(start, end)
        duration = max(0.3, distance * 0.002)  # 速度动态适配
        pyautogui.moveTo(start)
        pyautogui.dragTo(end, duration=duration, button='left')

def polar_sort(balls, center):
    # 转换为极坐标系
    polar_coords = []
    for (x,y), _ in balls:
        dx, dy = x - center[0], y - center[1]
        r = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        polar_coords.append( (r, theta, (x,y)) )
    
    # 按半径分组后角度排序
    sorted_balls = sorted(polar_coords, key=lambda x: (x[0], x[1]))
    return [item[2] for item in sorted_balls]

def is_aligned(points, threshold=2.0):
    if len(points) < 2: return True
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # 水平线检测
    if max(y_coords) - min(y_coords) < threshold:
        return True
    # 垂直线检测
    elif max(x_coords) - min(x_coords) < threshold:
        return True
    return False

def plan_path(current_layout, target_line):
    # 构建球体移动关系图
    graph = {}
    for ball in current_layout:
        neighbors = find_swipeable_balls(ball, current_layout)
        graph[ball] = neighbors
    
    # 使用A*算法寻找最优移动序列
    return a_star_search(graph, start=current_layout, goal=target_line)

class MotionPredictor:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4,2)
        # 初始化状态转移矩阵...
        
    def update(self, new_pos):
        self.kalman.predict()
        self.kalman.correct(new_pos)
        return self.kalman.statePost
    

class PathRLEnv(gym.Env):
    def __init__(self, screen_size=(1920,1080)):
        self.observation_space = spaces.Box(low=0, high=255, shape=(1080,1920,3))
        self.action_space = spaces.Discrete(8)  # 8方向滑动
        
    def step(self, action):
        # 执行滑动操作
        perform_swipe(action)
        
        # 获取奖励信号
        reward = self._calculate_reward()
        
        # 获取新状态
        next_state = get_screen()
        
        return next_state, reward, done, info
        
    def _calculate_reward(self):
        # 多维度奖励函数
        time_penalty = -0.1  # 时间惩罚项
        alignment_bonus = 10.0 if is_aligned() else 0
        path_efficiency = 1.0 / (movement_distance + 1e-5)
        return time_penalty + alignment_bonus + path_efficiency
    

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU()
        )
        
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softmax2d()
        )
        
        self.value_stream = nn.Linear(64*9*16, 512)
        self.action_stream = nn.Linear(64*9*16, 512)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # 空间注意力机制
        attn_weights = self.spatial_attn(features)
        context = torch.sum(features * attn_weights, dim=(2,3))
        
        # Dueling DQN架构
        value = self.value_stream(context)
        advantage = self.action_stream(context)
        q_values = value + (advantage - advantage.mean())
        return q_values


class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros(capacity)
        self.pos = 0
        
    def add(self, experience, td_error):
        priority = (abs(td_error) + 1e-5) ** 0.6
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.pos] = experience
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size):
        probs = self.priorities / self.priorities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        return [self.memory[idx] for idx in indices]
    

class HybridTrainer:
    def __init__(self):
        self.online_net = DQN(input_shape, num_actions)
        self.target_net = DQN(input_shape, num_actions)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-5)
        
    def update_model(self):
        # 双网络更新策略
        if self.step_counter % UPDATE_TARGET_EVERY == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            
        # 优先经验采样
        batch = self.buffer.sample(BATCH_SIZE)
        
        # 计算TD误差
        current_q = self.online_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].detach()
        expected_q = rewards + (GAMMA * next_q * (1 - dones))
        
        # 优先权更新
        td_errors = torch.abs(expected_q - current_q.squeeze())
        self.buffer.update_priorities(indices, td_errors)
        
        # 梯度裁剪优化
        loss = F.smooth_l1_loss(current_q, expected_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10)
        self.optimizer.step()


class TransferAdapter:
    def __init__(self, base_model):
        # 固定特征提取层
        for param in base_model.feature_extractor.parameters():
            param.requires_grad = False
            
        # 动态调整输出层
        self.new_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, num_new_actions)
        )
        
    def forward(self, x):
        features = self.base_model.feature_extractor(x)
        return self.new_head(features)
    

class PolicyDistiller:
    def __init__(self, teacher, student):
        self.teacher = teacher
        self.student = student
        self.temperature = 3.0
        
    def distill(self, states):
        # 教师模型生成软标签
        with torch.no_grad():
            teacher_probs = F.softmax(self.teacher(states)/self.temperature, dim=1)
            
        # 学生模型学习
        student_logits = self.student(states)
        loss = KL_divergence(F.log_softmax(student_logits/self.temperature, dim=1),
                             teacher_probs)
        return loss

Transition = namedtuple('Transition', 
    ('state', 'action', 'next_state', 'reward', 'done'))

def preprocess(state):
    # GPU加速预处理
    gpu_img = cv2.cuda_GpuMat(state)
    resized = cv2.cuda.resize(gpu_img, (256,144))
    normalized = cv2.cuda.normalize(resized, 0, 1, cv2.NORM_MINMAX)
    return normalized.download()

while training:
    state = env.reset()
    episode_reward = 0
    
    while True:
        # ε-greedy策略
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = model.predict(state)
            
        next_state, reward, done = env.step(action)
        buffer.add((state, action, next_state, reward, done))
        
        # 异步训练
        if len(buffer) > BATCH_SIZE:
            trainer.update_model()
            
        state = next_state
        episode_reward += reward
        
        if done: break

class EvaluationMetrics:
    def __init__(self):
        self.success_rate = deque(maxlen=100)
        self.time_cost = deque(maxlen=100)
        self.path_efficiency = []
        
    def log_episode(self, success, time, path_length):
        self.success_rate.append(success)
        self.time_cost.append(time)
        self.path_efficiency.append(optimal_length/path_length)

class SpatioTemporalAttention(nn.Module):
    def __init__(self):
        self.spatial_attn = SpatialGate()
        self.temporal_attn = TemporalGate()
        
    def forward(self, x):
        # x shape: (batch, seq_len, C, H, W)
        spatial_weights = self.spatial_attn(x)  # 空间重要区域
        temporal_weights = self.temporal_attn(x) # 关键时间步
        return x * spatial_weights * temporal_weights
    
class EvolutionaryStrategy:
    def __init__(self, population_size=50):
        self.population = [DQN() for _ in range(population_size)]
        
    def evolve(self, env, generations=100):
        for gen in range(generations):
            fitness = []
            for model in self.population:
                reward = evaluate_model(model, env)
                fitness.append(reward)
            
            # 选择前20%精英
            elite_indices = np.argsort(fitness)[-int(0.2*len(fitness)):]
            elites = [self.population[i] for i in elite_indices]
            
            # 生成新一代
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent = random.choice(elites)
                child = mutate(parent)
                new_population.append(child)
            
            self.population = new_population

    class MetaLearner:
    def __init__(self, inner_lr=1e-3, meta_lr=1e-4):
        self.model = DQN()
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=meta_lr)
        
    def meta_update(self, tasks):
        for task in tasks:
            # 内循环适应
            cloned_model = copy.deepcopy(self.model)
            fast_weights = cloned_model.parameters()
            
            for _ in range(5):  # 少量梯度更新
                loss = compute_loss(task, cloned_model)
                grads = torch.autograd.grad(loss, fast_weights)
                fast_weights = [w - inner_lr*g for w,g in zip(fast_weights, grads)]
                
            # 外循环元优化
            meta_loss = compute_loss(task, cloned_model)
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

    class CanaryRelease:
    def __init__(self):
        self.new_model = None
        self.traffic_ratio = 0.0
        
    def update_traffic(self, success_rate):
        # 成功率超过阈值时逐步增加流量
        if success_rate > 0.95:
            self.traffic_ratio = min(1.0, self.traffic_ratio + 0.1)
        else:
            self.traffic_ratio = max(0.0, self.traffic_ratio - 0.2)
            
    def get_action(self, state):
        if random.random() < self.traffic_ratio:
            return self.new_model.predict(state)
        else:
            return self.old_model.predict(state)
        
class MonitoringDashboard:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1)
        
    def update(self, metrics):
        # 绘制实时曲线
        self.ax1.clear()
        self.ax1.plot(metrics.success_rate, label='成功率')
        self.ax2.clear()
        self.ax2.plot(metrics.path_efficiency, label='路径效率')
        plt.pause(0.01)
