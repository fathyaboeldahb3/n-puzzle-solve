# n_puzzle_final_with_comments.py
# نسخة نهائية من مشروع N-Puzzle مع تعليقات تفصيلية بالعربي
# حفظ الملف ثم شغّله: python n_puzzle_final_with_comments.py
# ملاحظة: يلزم تثبيت pygame: pip install pygame

import pygame
import sys
import random
import time
import heapq

# ----------------------------
# تهيئة pygame والإعدادات العامة
# ----------------------------
pygame.init()
# أحجام النافذة — اخترت حجم مناسب للشكل اللي بعتها
WIDTH, HEIGHT = 980, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("N-Puzzle Solver")
# خطوط للعرض
FONT = pygame.font.SysFont("Arial", 22)
TITLE_FONT = pygame.font.SysFont("Arial", 36, bold=True)
clock = pygame.time.Clock()

# ----------------------------
# ألوان مستخدمة (ثابتة)
# ----------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)    # لون الزر المحدد (بنفسجي)
DARK_BLACK = (20, 20, 20) # لون الأزرار غير المحددة
BUTTON_LIGHT = (240, 240, 255)  # لون البلاطات داخل اللوح

# ----------------------------
# تعاريف سلوك الشفل بحسب الحجم والصعوبة
# - هذه الخريطة تُستخدم لتحديد عدد الحركات العشوائية (shuffle)
# ----------------------------
SHUFFLE_MAP = {
    3: {"Easy": 10, "Medium": 20, "Hard": 50},
    4: {"Easy": 30, "Medium": 80, "Hard": 120},
    5: {"Easy": 80, "Medium": 160, "Hard": 300},
}

# حدود للحماية (لمنع استهلاك وقت غير محدود)
MAX_STEPS = 500000

# ----------------------------
# دالة مساعدة لحساب القيم الهيوريستية
# ----------------------------
def heuristic_val(state, goal, size, method="manhattan"):
    """
    ترجع قيمة الـ heuristic للحالة state بالنسبة للـ goal.
    method: "manhattan" أو "misplaced"
    """
    if method == "manhattan":
        dist = 0
        pos = {v:i for i,v in enumerate(goal)}
        for i, val in enumerate(state):
            if val == 0: 
                continue
            goal_idx = pos[val]
            x1, y1 = divmod(i, size)
            x2, y2 = divmod(goal_idx, size)
            dist += abs(x1 - x2) + abs(y1 - y2)
        return dist
    elif method == "misplaced":
        return sum(1 for i, v in enumerate(state) if v != 0 and v != goal[i])
    return 0

# ----------------------------
# دوال حالات البازل: جيران، شفل، قابلية الحل
# ----------------------------
def get_neighbors(state, size):
    """ترجع قائمة الحالات الناتجة من تحريك البلاطة الفارغة (0) بمقدار خطوة"""
    moves = []
    idx = state.index(0)
    row, col = divmod(idx, size)
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < size and 0 <= c < size:
            new_idx = r * size + c
            new_state = list(state)
            new_state[idx], new_state[new_idx] = new_state[new_idx], new_state[idx]
            moves.append(tuple(new_state))
    return moves

def generate_by_moves(goal, size, moves):
    """
    نبدأ من الحالة الهدف ثم نعمل 'moves' حركات عشوائية صالحة بحيث تكون النتيجة قابلة للحل.
    هذه طريقة مضمونة لإنشاء حالة قابلة للحل (لأننا بدئنا من goal).
    """
    cur = tuple(goal)
    prev = None
    for _ in range(moves):
        neighs = get_neighbors(cur, size)
        # منع التراجع الفوري للخلف لعمل شفل أفضل
        choices = [n for n in neighs if n != prev]
        if not choices:
            choices = neighs
        nxt = random.choice(choices)
        prev = cur
        cur = nxt
    return cur

def is_solvable(state, size, goal):
    """
    تحقق قابلية الحل عن طريق حساب inversions.
    طريقة معيارية: 
    - لو الحجم فردي: inversions يجب أن تكون زوجية.
    - لو الحجم زوجي: تعتمد على موقع الصفر (صف من القاع).
    (مكتوبة علشان لو استخدمت توليد عشوائي مختلف)
    """
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    if size % 2 == 1:
        return inv % 2 == 0
    else:
        row = state.index(0) // size
        # صف من القاع = size - row
        return ((inv + (size - row)) % 2) == 0

# ----------------------------
# الخوارزميات: DFS (stack)، Best-First (Greedy)، A*
# - كلها ترجع (path, nodes, reason)
# - path: قائمة الحالات (من البداية إلى الهدف) أو None لو مش لقيت
# - nodes: عدد العقد الموسعة
# - reason: "solved" أو "time_limit" أو "max_steps" أو "exhausted"
# ----------------------------
def dfs(start, goal, size, time_limit=None, max_steps=MAX_STEPS):
    """
    DFS عبارة عن stack (عمق أولاً) — نستخدم هنا نهج غير متكرر مع visited لمنع الدوران.
    ملاحظة: هذا DFS قد يستغرق وقت طويل لذا هناك وقت/حد للعقد.
    """
    start_time = time.time()
    stack = [(start, [start])]
    visited = set()
    nodes = 0
    while stack:
        # فحص حدود الوقت
        if time_limit and (time.time() - start_time) > time_limit:
            return None, nodes, "time_limit"
        state, path = stack.pop()
        nodes += 1
        if nodes > max_steps:
            return None, nodes, "max_steps"
        if state == goal:
            return path, nodes, "solved"
        if state in visited:
            continue
        visited.add(state)
        # نضيف الجيران (لا نتحقق من عمق هنا لأن المثال السابق طلب تعطيل DFS للحجم الأكبر)
        for n in get_neighbors(state, size):
            if n not in visited:
                stack.append((n, path + [n]))
    return None, nodes, "exhausted"

def best_first(start, goal, size, heuristic="manhattan", time_limit=None, max_steps=MAX_STEPS):
    """
    Best-First (Greedy) يعتمد على أقل قيمة هيوريستيك فقط (g not considered).
    """
    start_time = time.time()
    pq = [(heuristic_val(start, goal, size, heuristic), [start])]
    visited = set()
    nodes = 0
    while pq:
        if time_limit and (time.time() - start_time) > time_limit:
            return None, nodes, "time_limit"
        _, path = heapq.heappop(pq)
        state = path[-1]
        nodes += 1
        if nodes > max_steps:
            return None, nodes, "max_steps"
        if state == goal:
            return path, nodes, "solved"
        if state in visited:
            continue
        visited.add(state)
        for n in get_neighbors(state, size):
            if n not in visited:
                h = heuristic_val(n, goal, size, heuristic)
                heapq.heappush(pq, (h, path + [n]))
    return None, nodes, "exhausted"

def a_star(start, goal, size, heuristic="manhattan", time_limit=None, max_steps=MAX_STEPS):
    """
    A* يستخدم f = g + h ويخزن أفضل g لكل حالة.
    """
    start_time = time.time()
    pq = [(heuristic_val(start, goal, size, heuristic), 0, [start])]  # (f, g, path)
    best_g = {start: 0}
    nodes = 0
    while pq:
        if time_limit and (time.time() - start_time) > time_limit:
            return None, nodes, "time_limit"
        f, g, path = heapq.heappop(pq)
        state = path[-1]
        nodes += 1
        if nodes > max_steps:
            return None, nodes, "max_steps"
        if state == goal:
            return path, nodes, "solved"
        # إذا كان لدينا g أفضل لهذه الحالة نكمل
        if best_g.get(state, float('inf')) < g:
            continue
        for n in get_neighbors(state, size):
            new_g = g + 1
            if new_g < best_g.get(n, float('inf')):
                best_g[n] = new_g
                h = heuristic_val(n, goal, size, heuristic)
                heapq.heappush(pq, (new_g + h, new_g, path + [n]))
    return None, nodes, "exhausted"

# ----------------------------
# رسم اللوحة (Board) — تستخدم في الأنيميشن وعرض الحالة
# ----------------------------
def draw_board(state, size, top_left, board_size):
    """
    ترسم لوح البازل في المكان top_left وبحجم board_size × board_size.
    كل بلاطة لها هامش بسيط (padding).
    """
    tile_size = board_size // size
    x0, y0 = top_left
    for i, val in enumerate(state):
        r, c = divmod(i, size)
        rect = pygame.Rect(x0 + c*tile_size + 6, y0 + r*tile_size + 6, tile_size - 12, tile_size - 12)
        if val == 0:
            # المربع الفارغ أسود
            pygame.draw.rect(screen, BLACK, rect, border_radius=10)
        else:
            # بلاطة فاتحة مع رقم
            pygame.draw.rect(screen, BUTTON_LIGHT, rect, border_radius=10)
            txt = FONT.render(str(val), True, BLACK)
            screen.blit(txt, txt.get_rect(center=rect.center))

# ----------------------------
# شاشة النتائج (بعد ما تشوف الأنيميشن و تضغط Next)
# ----------------------------
def result_screen(algo_name, steps, nodes, elapsed, reason=None):
    """
    تعرض معلومات عن نتيجة الحل:
    - هل تم حل البازل؟
    - عدد الخطوات
    - عدد العقد الموسعة
    - الوقت المستغرق
    """
    running = True
    while running:
        screen.fill(WHITE)
        title = TITLE_FONT.render("Puzzle Finished", True, PURPLE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))

        lines = [
            f"Algorithm: {algo_name}",
            f"Solved: {'Yes' if steps is not None else 'No'}",
            f"Steps: {len(steps)-1 if steps is not None else 'N/A'}",
            f"Nodes expanded: {nodes}",
            f"Time taken: {round(elapsed, 4)} s"
        ]
        if reason and reason != "solved":
            lines.append(f"Reason: {reason}")

        for i, ln in enumerate(lines):
            surf = FONT.render(ln, True, BLACK)
            screen.blit(surf, (WIDTH//2 - 200, 150 + i * 36))

        info_surf = FONT.render("Press ESC to return to menu", True, BLACK)
        screen.blit(info_surf, (WIDTH//2 - info_surf.get_width()//2, HEIGHT - 80))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return

        pygame.display.flip()
        clock.tick(30)

# ----------------------------
# دالة الأنيميشن: تعرض الخطوات بوتيرة 200ms مع معالجة للأحداث (حتى يمكن إغلاق النافذة)
# بعد الانتهاء تظهر زر Next للانتقال لعرض النتائج
# ----------------------------
def animate_solution_and_show_next(start_state, path, size, nodes, elapsed, algo_name, reason):
    """
    path: قائمة الحالات من البداية إلى الهدف (شاملة البداية والهدف)
    start_state: الحالة الابتدائية (path[0] عادة)
    """
    # إعداد المكان المرسوم للوحة
    board_size = min(520, 520)
    margin_left = (WIDTH - board_size) // 2
    margin_top = 80

    # إذا لم يكن هناك path (لم يُحل)، نعرض رسالة قصيرة بدل الأنيميشن
    if not path:
        # عرض رسالة بسيطة ثم زر Next
        while True:
            screen.fill(WHITE)
            txt = TITLE_FONT.render("No solution found", True, RED)
            screen.blit(txt, (WIDTH//2 - txt.get_width()//2, 60))
            # زر Next
            next_rect = pygame.Rect(WIDTH//2 - 80, HEIGHT - 120, 160, 50)
            pygame.draw.rect(screen, PURPLE, next_rect, border_radius=12)
            screen.blit(FONT.render("Next", True, WHITE), (next_rect.centerx - 32, next_rect.centery - 12))
            pygame.display.flip()
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if next_rect.collidepoint(ev.pos):
                        result_screen(algo_name, None, nodes, elapsed, reason)
                        return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    return
            clock.tick(30)

    # الآن لو فيه path — نعرض كل حالة بدوران غير محبوس
    # سنقوم بتوقيت كل خطوة بحيث تكون كل خطوة مرئية لمدة ~200ms
    step_index = 0
    last_time = time.time()
    step_delay = 0.200  # 200ms ثابت كما طلبت
    anim_done = False

    while True:
        now = time.time()
        # إذا الوقت مر للتقدم خطوة
        if not anim_done and now - last_time >= step_delay:
            step_index += 1
            last_time = now
            # لو وصلنا لنهاية المسار (آخر حالة في path) نوقف الأنيميشن
            if step_index >= len(path):
                anim_done = True
                step_index = len(path) - 1

        # رسم الإطار الحالي
        screen.fill(WHITE)
        # عنوان
        title = TITLE_FONT.render("Solving animation", True, PURPLE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
        # حالة
        draw_board(path[step_index], size, (margin_left, margin_top), board_size)
        info = FONT.render(f"Step {step_index}/{len(path)-1}", True, BLACK)
        screen.blit(info, (20, 20))

        # زر Next يظهر فقط بعد انتهاء الأنيميشن
        next_rect = pygame.Rect(WIDTH//2 - 80, HEIGHT - 120, 160, 50)
        if anim_done:
            pygame.draw.rect(screen, PURPLE, next_rect, border_radius=12)
            screen.blit(FONT.render("Next", True, WHITE), (next_rect.centerx - 32, next_rect.centery - 12))
        else:
            # لو مش خلص نرسم زر معطل بلون رمادي
            pygame.draw.rect(screen, (200,200,200), next_rect, border_radius=12)
            screen.blit(FONT.render("Next", True, (120,120,120)), (next_rect.centerx - 32, next_rect.centery - 12))

        # معالجة الأحداث أثناء الأنيميشن (مهم: يفتح يغلق النافذة ويفصل الرجوع)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if anim_done and next_rect.collidepoint(ev.pos):
                    # ننتقل لشاشة النتائج مع بيانات الحل
                    result_screen(algo_name, path, nodes, elapsed, reason)
                    return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return
                # مفاتيح مفيدة: سهم يمين ويسار للتنقّل يدوياً أثناء الأنيميشن
                if ev.key == pygame.K_RIGHT:
                    if step_index < len(path)-1:
                        step_index += 1
                if ev.key == pygame.K_LEFT:
                    if step_index > 0:
                        step_index -= 1

        pygame.display.flip()
        clock.tick(60)

# ----------------------------
# شاشة الإعدادات / القائمة الرئيسية (مطابقة للصورة)
# - أعمدة: أحجام (يسار)، خوارزميات (منتصف)، هيوريستيك (يمين وسط)، صعوبة (يمين)
# - أزرار Start و Info أسفل منتصف الصفحة بالبنفسجي
# ----------------------------
class MenuButton:
    """
    كلاس بسيط للأزرار في القائمة: يدعم الرسم وحالة 'selected' والتعامل مع النقر.
    key: قيمة تمثل ماذا يعني هذا الزر (مثلاً 3 أو 'DFS' أو 'Medium')
    callback: دالة تستقبل المفتاح وتطبّق التغيير
    """
    def __init__(self, text, x, y, w, h, key, callback):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.key = key
        self.callback = callback
        self.selected = False

    def draw(self, surf):
        # الخلفية بنفسجي لو مختار و اسود غامق لو لا
        bg = PURPLE if self.selected else DARK_BLACK
        pygame.draw.rect(surf, bg, self.rect, border_radius=12)
        pygame.draw.rect(surf, WHITE, self.rect, 2, border_radius=12)
        txt = FONT.render(self.text, True, WHITE)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and self.rect.collidepoint(ev.pos):
            # عند النقر نُنادي الـ callback مع المفتاح
            self.callback(self.key)

def choose_settings():
    """
    شاشة اختيار الإعدادات — تعرض جميع الأزرار وتُحدّث اختيارات المستخدم.
    بعد الضغط Start ينشئ حالة البازل ويشغّل الخوارزمية ثم الأنيميشن ثم النتائج.
    """
    # القيم الافتراضية
    chosen_size = 3
    chosen_algo = "DFS"
    chosen_heur = "manhattan"
    chosen_diff = "Medium"

    # دوال تحديث بسيطة تُغلق DFS إذا اخترت حجم أكبر من 3
    def set_size(k):
        nonlocal chosen_size, chosen_algo
        chosen_size = k
        if chosen_size > 3 and chosen_algo == "DFS":
            # نجبر الانتقال إلى A* لأن DFS غير مسموح هنا
            set_algo("A*")

    def set_algo(a):
        nonlocal chosen_algo
        chosen_algo = a

    def set_heur(h):
        nonlocal chosen_heur
        chosen_heur = h

    def set_diff(d):
        nonlocal chosen_diff
        chosen_diff = d

    # مواقع الأزرار متطابقة تقريبًا مع الصورة
    buttons = []

    # أحجام (عمود يسار)
    buttons.append(MenuButton("3x3", 120, 160, 140, 56, 3, lambda k: set_size(k)))
    buttons.append(MenuButton("4x4", 120, 240, 140, 56, 4, lambda k: set_size(k)))
    buttons.append(MenuButton("5x5", 120, 320, 140, 56, 5, lambda k: set_size(k)))

    # خوارزميات (عمود منتصف)
    buttons.append(MenuButton("DFS", 360, 160, 180, 56, "DFS", lambda k: set_algo(k)))
    buttons.append(MenuButton("Best-First", 360, 240, 180, 56, "Best-First", lambda k: set_algo(k)))
    buttons.append(MenuButton("A*", 360, 320, 180, 56, "A*", lambda k: set_algo(k)))

    # هيوريستيك (عمود يمين وسط) — باينة دايمًا لكن تتعمل فقط مع Best-First / A*
    buttons.append(MenuButton("Manhattan", 620, 160, 160, 48, "manhattan", lambda k: set_heur(k)))
    buttons.append(MenuButton("Misplaced", 620, 230, 160, 48, "misplaced", lambda k: set_heur(k)))

    # صعوبات (عمود أقصى اليمين)
    buttons.append(MenuButton("Easy", 820, 160, 120, 48, "Easy", lambda k: set_diff(k)))
    buttons.append(MenuButton("Medium", 820, 240, 120, 48, "Medium", lambda k: set_diff(k)))
    buttons.append(MenuButton("Hard", 820, 320, 120, 48, "Hard", lambda k: set_diff(k)))

    # أزرار Start و Info (منتصف أسفل)
    start_btn = MenuButton("Start", WIDTH//2 - 100, 460, 200, 64, "start", lambda k: None)
    info_btn = MenuButton("Info", WIDTH//2 - 100, 540, 200, 52, "info", lambda k: None)
    # نجعل Start و Info بنفسجي كما في التصميم
    start_btn.selected = True
    info_btn.selected = True

    # ربط فعلي لزر Start و Info مُباشر بعد تعريف الدوال
    def start_game():
        nonlocal chosen_size, chosen_algo, chosen_heur, chosen_diff
        # منع DFS أكبر من 3
        if chosen_algo == "DFS" and chosen_size > 3:
            # تعرض تحذير بسيط في القائمة بدل التشغيل — لكن هنعكس لآخر لحظة
            return
        shuffle = SHUFFLE_MAP[chosen_size][chosen_diff]
        # إنشاء بداية البازل من خلال توليد حركات من goal
        goal = tuple(list(range(1, chosen_size*chosen_size)) + [0])
        start_state = generate_by_moves(goal, chosen_size, shuffle)
        # تأكد من القابلية (احتياطي)
        if not is_solvable(start_state, chosen_size, goal):
            start_state = generate_by_moves(goal, chosen_size, shuffle)

        # تنفيذ الخوارزمية المختارة
        t1 = time.time()
        if chosen_algo == "DFS":
            path, nodes, reason = dfs(start_state, goal, chosen_size, time_limit=20, max_steps=MAX_STEPS)
        elif chosen_algo == "Best-First":
            path, nodes, reason = best_first(start_state, goal, chosen_size, heuristic=chosen_heur, time_limit=60, max_steps=MAX_STEPS)
        else:  # A*
            path, nodes, reason = a_star(start_state, goal, chosen_size, heuristic=chosen_heur, time_limit=120, max_steps=MAX_STEPS)
        elapsed = time.time() - t1

        # نعرض الأنيميشن مع تفعيل زر Next بعد الانتهاء
        animate_solution_and_show_next(start_state, path, chosen_size, nodes, elapsed, chosen_algo, reason)

    # الآن نربط callback لزر Start و Info
    start_btn.callback = lambda k: start_game()
    info_btn.callback = lambda k: info_screen()

    # حلقة العرض الرئيسية للقائمة
    running = True
    while running:
        screen.fill(WHITE)
        # عنوان
        title = TITLE_FONT.render("N-Puzzle Solver", True, PURPLE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))

        # تحديث حالة اختيار كل زر (selected) بناءً على المتغيرات
        for b in buttons:
            if b.key == 3 and chosen_size == 3:
                b.selected = True
            elif b.key == 4 and chosen_size == 4:
                b.selected = True
            elif b.key == 5 and chosen_size == 5:
                b.selected = True
            elif b.key == "DFS" and chosen_algo == "DFS":
                b.selected = True
            elif b.key == "Best-First" and chosen_algo == "Best-First":
                b.selected = True
            elif b.key == "A*" and chosen_algo == "A*":
                b.selected = True
            elif b.key == "manhattan" and chosen_heur == "manhattan":
                b.selected = True
            elif b.key == "misplaced" and chosen_heur == "misplaced":
                b.selected = True
            elif b.key == "Easy" and chosen_diff == "Easy":
                b.selected = True
            elif b.key == "Medium" and chosen_diff == "Medium":
                b.selected = True
            elif b.key == "Hard" and chosen_diff == "Hard":
                b.selected = True
            else:
                # لكن لا نغيّر حالة أزرار Start/Info هنا؛ هم ثابتون بنفسجي
                if b.key not in ("start", "info"):
                    b.selected = False

        # رسم الأزرار
        for b in buttons:
            b.draw(screen)
        start_btn.draw(screen)
        info_btn.draw(screen)

        # عرض تحذير إذا DFS والـ size > 3 (تنبيه للمستخدم)
        if chosen_algo == "DFS" and chosen_size > 3:
            warn = FONT.render("DFS not allowed for size > 3 (will not start).", True, (180,0,0))
            screen.blit(warn, (120, 410))

        # شريط الملخص في الأسفل (يتحدّث أوتوماتيك)
        shuffle_val = SHUFFLE_MAP[chosen_size][chosen_diff]
        summary = f"Size: {chosen_size}x{chosen_size} | Algo: {chosen_algo} | Heuristic: {chosen_heur} | Difficulty: {chosen_diff} | Shuffle: {shuffle_val}"
        screen.blit(FONT.render(summary, True, BLACK), (40, HEIGHT - 40))

        # التعامل مع الأحداث (نمر على كل زر وندير حدث النقر)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # نمر على أزرار القائمة للاستجابة للنقر
                for b in buttons:
                    b.handle_event(ev)
                start_btn.handle_event(ev)
                info_btn.handle_event(ev)
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return

        pygame.display.flip()
        clock.tick(30)

# ----------------------------
# شاشة المعلومات البسيطة
# ----------------------------
def info_screen():
    running = True
    while running:
        screen.fill(WHITE)
        lines = [
            "N-Puzzle Solver",
            "Implemented with Python + Pygame",
            "Algorithms: DFS (limited), Best-First (Greedy), A* (optimal if heuristic admissible)",
            "Heuristics: Manhattan, Misplaced",
            "Choose settings then press Start. Press ESC to return."
        ]
        for i, ln in enumerate(lines):
            screen.blit(FONT.render(ln, True, BLACK), (WIDTH//2 - 350, 150 + i*36))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return
        pygame.display.flip()
        clock.tick(30)

# ----------------------------
# القائمة الرئيسية البسيطة (منها تروح للإعدادات أو الخروج)
# ----------------------------
def main_menu():
    # ثلاثة أزرار: Settings، Info، Quit
    def goto_settings(_): choose_settings()
    def goto_info(_): info_screen()
    start_btn = MenuButton("Settings", WIDTH//2 - 120, 220, 240, 70, "settings", goto_settings)
    info_btn = MenuButton("Info", WIDTH//2 - 120, 320, 240, 70, "info", goto_info)
    quit_btn = MenuButton("Quit", WIDTH//2 - 120, 420, 240, 70, "quit", lambda k: sys.exit())

    # نجعل زر Settings بنفسجي بشكل افتراضي كما في الصورة
    start_btn.selected = True

    while True:
        screen.fill(WHITE)
        title = TITLE_FONT.render("N-Puzzle Solver", True, PURPLE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 40))

        start_btn.draw(screen)
        info_btn.draw(screen)
        quit_btn.draw(screen)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                start_btn.handle_event(ev)
                info_btn.handle_event(ev)
                quit_btn.handle_event(ev)

        pygame.display.flip()
        clock.tick(30)

# ----------------------------
# نقطة البداية لتشغيل البرنامج
# ----------------------------
if __name__ == "__main__":
    main_menu()
