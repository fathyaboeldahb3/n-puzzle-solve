# n_puzzle_final.py
import pygame
import sys
import random
import time
import heapq

# ----------------------------
# إعداد Pygame والاعدادات العامة
# ----------------------------
pygame.init()
WIDTH, HEIGHT = 980, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("N-Puzzle Solver")
FONT = pygame.font.SysFont("Arial", 22)
TITLE_FONT = pygame.font.SysFont("Arial", 36, bold=True)
clock = pygame.time.Clock()

# ألوان
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
DARK_BLACK = (20, 20, 20)
GRAY = (40, 40, 40)
LIGHT_BG = (245, 245, 245)
BUTTON_LIGHT = (240, 240, 255)

# ----------------------------
# خريطة الشفل حسب الحجم والصعوبة
# ----------------------------
SHUFFLE_MAP = {
    3: {"Easy": 10, "Medium": 20, "Hard": 50},
    4: {"Easy": 30, "Medium": 80, "Hard": 120},
    5: {"Easy": 80, "Medium": 160, "Hard": 300},
}

# ----------------------------
# زر Button مع حالة محددة
# ----------------------------
class Button:
    def __init__(self, text, x, y, w, h, callback=None, key=None):
        self.text = text
        self.rect = pygame.Rect(x, y, w, h)
        self.callback = callback
        self.key = key  # مفتاح لمطابقة الاختيار
        self.selected = False

    def draw(self, surface):
        # لون الخلفية: بنفسجي لو محدد، اسود غامق لو مش محدد
        bg = PURPLE if self.selected else DARK_BLACK
        # نص أبيض عند المحدد، أبيض عند غير المحدد
        pygame.draw.rect(surface, bg, self.rect, border_radius=12)
        # outline أبيض خفيف
        pygame.draw.rect(surface, WHITE, self.rect, 2, border_radius=12)
        txt = FONT.render(self.text, True, WHITE)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                if self.callback:
                    self.callback(self.key)

# ----------------------------
# دوال بازل و توليد حالات
# ----------------------------
def get_neighbors(state, size):
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

def manhattan(state, goal, size):
    dist = 0
    pos = {val:i for i,val in enumerate(goal)}
    for i, val in enumerate(state):
        if val == 0: continue
        goal_idx = pos[val]
        x1, y1 = divmod(i, size)
        x2, y2 = divmod(goal_idx, size)
        dist += abs(x1 - x2) + abs(y1 - y2)
    return dist

def generate_by_moves(goal, size, moves):
    cur = tuple(goal)
    prev = None
    for _ in range(moves):
        neighs = get_neighbors(cur, size)
        choices = [n for n in neighs if n != prev]
        if not choices:
            choices = neighs
        nxt = random.choice(choices)
        prev = cur
        cur = nxt
    return cur

def is_solvable(state, size, goal):
    # نتحقق من القابلية للحل بواسطة inversions
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
        # صف من القاع
        return ((inv + (size - row)) % 2) == 0

# ----------------------------
# كلاس NPuzzle (حالة، خوارزميات، هيوريستكس)
# ----------------------------
class NPuzzle:
    def __init__(self, size, shuffle_moves):
        self.size = size
        self.goal = tuple(list(range(1, size*size)) + [0])
        # نضمن ان الشفل ينتج حالة قابلة للحل
        start = generate_by_moves(self.goal, size, shuffle_moves)
        if not is_solvable(start, size, self.goal):
            # إن لم تكن قابلة للحل — نعاود
            start = generate_by_moves(self.goal, size, shuffle_moves)
        self.start = start
        self.steps = []
        self.nodes = 0
        self.time_taken = 0.0
        self.solved = False

    # DFS محدود العمق (يعتمد على max_depth)
    def dfs(self, max_depth=50, time_limit=20):
        start_time = time.time()
        visited = set()
        found_path = None
        nodes = 0
        sys.setrecursionlimit(10000)

        def dfs_rec(state, depth, path):
            nonlocal found_path, nodes
            if found_path is not None:
                return
            if time.time() - start_time > time_limit:
                return
            if state == self.goal:
                found_path = list(path)
                return
            if depth >= max_depth:
                return
            visited.add(state)
            for n in get_neighbors(state, self.size):
                if n not in visited:
                    nodes += 1
                    dfs_rec(n, depth+1, path + [n])

        dfs_rec(self.start, 0, [])
        self.time_taken = time.time() - start_time
        self.nodes = nodes
        if found_path is not None:
            self.steps = found_path
            self.solved = True
        else:
            self.steps = []
            self.solved = False

    # Best-First (Greedy)
    def best_first(self, heuristic="manhattan", time_limit=60, max_nodes=200000):
        start_time = time.time()
        pq = []
        visited = set()
        start_h = manhattan(self.start, self.goal, self.size) if heuristic=="manhattan" else sum(1 for i,v in enumerate(self.start) if v!=0 and v!=self.goal[i])
        heapq.heappush(pq, (start_h, self.start, []))
        nodes = 0
        found = False
        while pq:
            if time.time() - start_time > time_limit:
                break
            h, state, path = heapq.heappop(pq)
            if state == self.goal:
                self.steps = path
                found = True
                break
            if state in visited:
                continue
            visited.add(state)
            for n in get_neighbors(state, self.size):
                if n not in visited:
                    nodes += 1
                    hv = manhattan(n, self.goal, self.size) if heuristic=="manhattan" else sum(1 for i,v in enumerate(n) if v!=0 and v!=self.goal[i])
                    heapq.heappush(pq, (hv, n, path + [n]))
            if nodes > max_nodes:
                break
        self.time_taken = time.time() - start_time
        self.nodes = nodes
        self.solved = found

    # A* search
    def a_star(self, heuristic="manhattan", time_limit=120, max_nodes=500000):
        start_time = time.time()
        open_heap = []
        start_h = manhattan(self.start, self.goal, self.size) if heuristic=="manhattan" else sum(1 for i,v in enumerate(self.start) if v!=0 and v!=self.goal[i])
        heapq.heappush(open_heap, (start_h, 0, self.start, []))  # f, g, state, path
        best_g = {self.start: 0}
        nodes = 0
        found = False
        while open_heap:
            if time.time() - start_time > time_limit:
                break
            f, g, state, path = heapq.heappop(open_heap)
            if state == self.goal:
                self.steps = path
                found = True
                break
            if best_g.get(state, float('inf')) < g:
                continue
            for n in get_neighbors(state, self.size):
                new_g = g + 1
                if new_g < best_g.get(n, float('inf')):
                    best_g[n] = new_g
                    nodes += 1
                    h = manhattan(n, self.goal, self.size) if heuristic=="manhattan" else sum(1 for i,v in enumerate(n) if v!=0 and v!=self.goal[i])
                    heapq.heappush(open_heap, (new_g + h, new_g, n, path + [n]))
            if nodes > max_nodes:
                break
        self.time_taken = time.time() - start_time
        self.nodes = nodes
        self.solved = found

# ----------------------------
# رسم البازل والأنيميشن
# ----------------------------
def draw_board(state, size, top_left, board_size):
    tile_size = board_size // size
    x0, y0 = top_left
    for i, val in enumerate(state):
        r, c = divmod(i, size)
        rect = pygame.Rect(x0 + c*tile_size + 4, y0 + r*tile_size + 4, tile_size - 8, tile_size - 8)
        if val == 0:
            pygame.draw.rect(screen, BLACK, rect, border_radius=8)
        else:
            pygame.draw.rect(screen, BUTTON_LIGHT, rect, border_radius=8)
            txt = FONT.render(str(val), True, BLACK)
            screen.blit(txt, txt.get_rect(center=rect.center))

# ----------------------------
# شاشة النتائج (بعد الحل)
# ----------------------------
def show_results(puzzle):
    running = True
    while running:
        screen.fill(WHITE)
        title = TITLE_FONT.render("Algorithm finished!", True, PURPLE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))

        texts = [
            f"Solved: {'Yes' if puzzle.solved else 'No'}",
            f"Steps: {len(puzzle.steps)}",
            f"Nodes expanded: {puzzle.nodes}",
            f"Time taken: {round(puzzle.time_taken, 4)} sec",
            "Press SPACE to see animation (if solved) | ESC to return"
        ]
        for i,t in enumerate(texts):
            surf = FONT.render(t, True, BLACK)
            screen.blit(surf, (60, 120 + i*40))

        # أزرار بسيطة
        back_rect = pygame.Rect(WIDTH - 260, HEIGHT - 110, 200, 50)
        pygame.draw.rect(screen, PURPLE, back_rect, border_radius=12)
        screen.blit(FONT.render("Back to Menu", True, WHITE), back_rect.center)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return
                if ev.key == pygame.K_SPACE and puzzle.solved and puzzle.steps:
                    animate_solution(puzzle)
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if back_rect.collidepoint(ev.pos):
                    return

        pygame.display.flip()
        clock.tick(30)

def animate_solution(puzzle):
    states = [puzzle.start] + puzzle.steps
    size = puzzle.size
    board_size = min(520, 520)
    tile_size = board_size // size
    margin_left = (WIDTH - board_size)//2
    margin_top = 80

    idx = 0
    running = True
    autoplay = False
    autoplay_speed = 6  # frames per second when autoplay
    last_auto = 0

    while running:
        screen.fill(WHITE)
        # رسم اللوحة
        draw_board(states[idx], size, (margin_left, margin_top), board_size)
        info = FONT.render(f"Step {idx}/{len(states)-1}", True, BLACK)
        screen.blit(info, (20, 20))

        # أزرار Prev / Next / Auto / Back
        prev_rect = pygame.Rect(WIDTH//2 - 260, HEIGHT - 100, 140, 48)
        next_rect = pygame.Rect(WIDTH//2 + 120, HEIGHT - 100, 140, 48)
        auto_rect = pygame.Rect(WIDTH//2 - 60, HEIGHT - 100, 140, 48)
        back_rect = pygame.Rect(WIDTH - 220, HEIGHT - 100, 160, 48)

        pygame.draw.rect(screen, DARK_BLACK, prev_rect, border_radius=10)
        pygame.draw.rect(screen, DARK_BLACK, next_rect, border_radius=10)
        pygame.draw.rect(screen, PURPLE if autoplay else DARK_BLACK, auto_rect, border_radius=10)
        pygame.draw.rect(screen, PURPLE, back_rect, border_radius=10)

        screen.blit(FONT.render("Prev", True, WHITE), prev_rect.center)
        screen.blit(FONT.render("Next", True, WHITE), next_rect.center)
        screen.blit(FONT.render("Auto", True, WHITE), auto_rect.center)
        screen.blit(FONT.render("Back", True, WHITE), back_rect.center)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    return
                if ev.key == pygame.K_RIGHT:
                    idx = min(idx + 1, len(states)-1)
                if ev.key == pygame.K_LEFT:
                    idx = max(idx - 1, 0)
                if ev.key == pygame.K_SPACE:
                    autoplay = not autoplay
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                if prev_rect.collidepoint(ev.pos):
                    idx = max(idx - 1, 0)
                if next_rect.collidepoint(ev.pos):
                    idx = min(idx + 1, len(states)-1)
                if auto_rect.collidepoint(ev.pos):
                    autoplay = not autoplay
                if back_rect.collidepoint(ev.pos):
                    return

        if autoplay:
            now = time.time()
            if now - last_auto > 1.0/autoplay_speed:
                idx = min(idx + 1, len(states)-1)
                last_auto = now
                if idx == len(states)-1:
                    autoplay = False

        pygame.display.flip()
        clock.tick(60)

# ----------------------------
# شاشة الإعدادات / القائمة بتصميم مطابق للصورة
# ----------------------------
def choose_settings():
    # القيم الافتراضية
    chosen_size = 3
    chosen_algo = "DFS"
    chosen_diff = "Medium"
    chosen_heur = "manhattan"

    # دوال للتحديث
    def set_size(k):
        nonlocal chosen_size, chosen_algo
        chosen_size = k
        # لو الحجم أكبر من 3: ماينفعش DFS
        if chosen_size > 3 and chosen_algo == "DFS":
            # إجبار التحويل إلى A*
            set_algo("A*")

    def set_algo(a):
        nonlocal chosen_algo
        chosen_algo = a

    def set_diff(d):
        nonlocal chosen_diff
        chosen_diff = d

    def set_heur(h):
        nonlocal chosen_heur
        chosen_heur = h

    # أزرار (نحط أماكن مطابقة تقريبًا للصورة)
    # أعمدة: left sizes, middle algos, right difficulties
    buttons = []

    # أحجام
    buttons.append(Button("3x3", 110, 160, 140, 56, callback=lambda k: set_size(k), key=3))
    buttons.append(Button("4x4", 110, 240, 140, 56, callback=lambda k: set_size(k), key=4))
    buttons.append(Button("5x5", 110, 320, 140, 56, callback=lambda k: set_size(k), key=5))

    # خوارزميات
    buttons.append(Button("DFS", 340, 160, 160, 56, callback=lambda k: set_algo(k), key="DFS"))
    buttons.append(Button("Best-First", 340, 240, 160, 56, callback=lambda k: set_algo(k), key="Best-First"))
    buttons.append(Button("A*", 340, 320, 160, 56, callback=lambda k: set_algo(k), key="A*"))

    # هيوريستيك
    buttons.append(Button("Manhattan", 560, 160, 160, 48, callback=lambda k: set_heur(k), key="manhattan"))
    buttons.append(Button("Misplaced", 560, 230, 160, 48, callback=lambda k: set_heur(k), key="misplaced"))

    # صعوبات (يمين أعلى)
    buttons.append(Button("Easy", 760, 160, 120, 48, callback=lambda k: set_diff(k), key="Easy"))
    buttons.append(Button("Medium", 760, 230, 120, 48, callback=lambda k: set_diff(k), key="Medium"))
    buttons.append(Button("Hard", 760, 300, 120, 48, callback=lambda k: set_diff(k), key="Hard"))

    # Start و Info (مركزيين تحت)
    start_btn = Button("Start", WIDTH//2 - 100, 460, 200, 64, callback=lambda k: None, key="start")
    info_btn = Button("Info", WIDTH//2 - 100, 540, 200, 52, callback=lambda k: None, key="info")

    # دالة لتحديث حالة الأزرار المحددة
    def update_button_states():
        for b in buttons:
            if b.key == 3 and chosen_size == 3: b.selected = True
            elif b.key == 4 and chosen_size == 4: b.selected = True
            elif b.key == 5 and chosen_size == 5: b.selected = True
            elif b.key == "DFS" and chosen_algo == "DFS": b.selected = True
            elif b.key == "Best-First" and chosen_algo == "Best-First": b.selected = True
            elif b.key == "A*" and chosen_algo == "A*": b.selected = True
            elif b.key == "manhattan" and chosen_heur == "manhattan": b.selected = True
            elif b.key == "misplaced" and chosen_heur == "misplaced": b.selected = True
            elif b.key == "Easy" and chosen_diff == "Easy": b.selected = True
            elif b.key == "Medium" and chosen_diff == "Medium": b.selected = True
            elif b.key == "Hard" and chosen_diff == "Hard": b.selected = True
            else:
                b.selected = False
        # Start و Info أزرار بنفسجية ثابتة (كما في الصورة)
        start_btn.selected = True
        info_btn.selected = True

    # دالة بدء اللعبة فعليًا
    def start_game():
        nonlocal chosen_size, chosen_algo, chosen_diff, chosen_heur
        # لو DFS و size > 3 نقفل
        if chosen_algo == "DFS" and chosen_size > 3:
            # نعرض رسالة صغيرة ثم نعود
            return
        shuffle = SHUFFLE_MAP[chosen_size][chosen_diff]
        puzz = NPuzzle(chosen_size, shuffle)
        # نحدد تنفيذ الخوارزمية
        if chosen_algo == "DFS":
            max_depth = 50 if chosen_size == 3 else 30
            puzz.dfs(max_depth=max_depth, time_limit=30)
        elif chosen_algo == "Best-First":
            puzz.best_first(heuristic=chosen_heur)
        else:
            puzz.a_star(heuristic=chosen_heur)
        show_results(puzz)

    # إعادة ربط الأزرار Start / Info callback
    start_btn.callback = lambda k: start_game()
    info_btn.callback = lambda k: info_screen()

    # حلقة الواجهة
    running = True
    while running:
        screen.fill(WHITE)
        title = TITLE_FONT.render("N-Puzzle Solver", True, PURPLE)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 30))

        # تحديث حالات الأزرار
        update_button_states()

        # رسم الأزرار
        for b in buttons:
            b.draw(screen)
        start_btn.draw(screen)
        info_btn.draw(screen)

        # رسالة تحذير لو DFS مش متاحة
        if chosen_size > 3 and chosen_algo == "DFS":
            warn = FONT.render("DFS not allowed for size > 3 (auto-switched to A*)", True, (180,0,0))
            screen.blit(warn, (120, 400))

        # عرض الـ Summary في الأسفل
        shuffle_val = SHUFFLE_MAP[chosen_size][chosen_diff]
        summary = f"Size: {chosen_size}x{chosen_size} | Algo: {chosen_algo} | Difficulty: {chosen_diff} | Shuffle: {shuffle_val}"
        summary_surf = FONT.render(summary, True, BLACK)
        screen.blit(summary_surf, (60, HEIGHT - 40))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                # تحقق ضغط على أزرار القائمة
                for b in buttons:
                    b.handle_event(ev)
                start_btn.handle_event(ev)
                info_btn.handle_event(ev)

        pygame.display.flip()
        clock.tick(30)

# ----------------------------
# شاشة المعلومة البسيطة
# ----------------------------
def info_screen():
    running = True
    while running:
        screen.fill(WHITE)
        lines = [
            "N-Puzzle Solver",
            "Developed in Python with Pygame",
            "Algorithms: DFS (limited), Best-First (Greedy), A*",
            "Choose Size, Algorithm, Heuristic and Difficulty then Start.",
            "Press ESC to return."
        ]
        for i,ln in enumerate(lines):
            screen.blit(FONT.render(ln, True, BLACK), (WIDTH//2 - 300, 150 + i*40))
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                return
        pygame.display.flip()
        clock.tick(30)

# ----------------------------
# القائمة الرئيسية البسيطة (تنقلك للاعدادات او للخروج)
# ----------------------------
def main_menu():
    start_btn = Button("Settings", WIDTH//2 - 120, 220, 240, 70, callback=lambda k: choose_settings(), key="settings")
    info_btn = Button("Info", WIDTH//2 - 120, 320, 240, 70, callback=lambda k: info_screen(), key="info")
    quit_btn = Button("Quit", WIDTH//2 - 120, 420, 240, 70, callback=lambda k: sys.exit(), key="quit")
    # نجعل الزرار الأوسط بنفسجي مثل التصميم
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
# نقطة البداية
# ----------------------------
if __name__ == "__main__":
    main_menu()
