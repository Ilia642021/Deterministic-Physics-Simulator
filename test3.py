import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider
import time
import pickle

# ---------------- НАСТРОЙКИ ----------------
FPS = 100
SIM_TIME = 20  # секунд
DT = 1.0 / FPS
TOTAL_FRAMES = int(SIM_TIME * FPS) + 1  # +1 для начального кадра

# Шарики: [позиция, скорость, радиус, цвет]
balls = [
    [np.array([-4.0, -2.0], dtype=float), np.array([2.5, 2.2], dtype=float), 0.25, 'red'],
    [np.array([0.0, 0.0], dtype=float), np.array([1.5, -2.5], dtype=float), 0.3, 'blue'],
    [np.array([2.5, 2.0], dtype=float), np.array([-2.2, -1.8], dtype=float), 0.2, 'green']
]

# Стены
walls = [
    (np.array([-5, -3], dtype=float), np.array([5, -3], dtype=float)),
    (np.array([-5, 3], dtype=float), np.array([5, 3], dtype=float)),
    (np.array([-5, -3], dtype=float), np.array([-5, 3], dtype=float)),
    (np.array([5, -3], dtype=float), np.array([5, 3], dtype=float)),
    (np.array([0, -1], dtype=float), np.array([2, 2], dtype=float))  # наклонная стена
]


# ---------------- ФИЗИКА ----------------
def reflect_vector(v, n):
    """Отражение вектора относительно нормали"""
    n = n / np.linalg.norm(n)
    return v - 2 * np.dot(v, n) * n


def collide_circle_time(p1, v1, r1, p2, v2, r2):
    """Время столкновения двух движущихся кругов"""
    dp = p1 - p2
    dv = v1 - v2
    a = np.dot(dv, dv)
    b = 2 * np.dot(dp, dv)
    c = np.dot(dp, dp) - (r1 + r2) ** 2

    if a == 0:
        return None

    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    t_candidates = [t for t in (t1, t2) if 1e-8 < t <= 1.0]
    return min(t_candidates) if t_candidates else None


def collide_line_time_with_normal(p, v, r, A, B):
    """Время столкновения круга с отрезком и нормаль в точке столкновения"""
    AB = B - A
    AB_len = np.linalg.norm(AB)
    if AB_len < 1e-8:
        return None

    AB_unit = AB / AB_len
    n1 = np.array([AB[1], -AB[0]]) / AB_len
    n2 = -n1

    dist_to_line = np.dot(p - A, n1)

    if dist_to_line >= 0:
        n = n1
        effective_dist = dist_to_line
    else:
        n = n2
        effective_dist = -dist_to_line

    vn = np.dot(v, n)

    if vn >= -1e-10:
        t_line = None
    else:
        t_line = (effective_dist - r) / (-vn)

    t_A = collide_circle_time(p, v, r, A, np.zeros(2), 1e-8)
    t_B = collide_circle_time(p, v, r, B, np.zeros(2), 1e-8)

    candidates = []

    if t_line is not None and t_line > 1e-8:
        contact_point = p + v * t_line
        AP = contact_point - A
        proj = np.dot(AP, AB_unit)
        if 0 <= proj <= AB_len:
            candidates.append((t_line, n, 'line'))

    if t_A is not None and t_A > 1e-8:
        collision_point = p + v * t_A
        normal_A = collision_point - A
        normal_A_len = np.linalg.norm(normal_A)
        if normal_A_len > 1e-8:
            normal_A /= normal_A_len
            candidates.append((t_A, normal_A, 'corner'))

    if t_B is not None and t_B > 1e-8:
        collision_point = p + v * t_B
        normal_B = collision_point - B
        normal_B_len = np.linalg.norm(normal_B)
        if normal_B_len > 1e-8:
            normal_B /= normal_B_len
            candidates.append((t_B, normal_B, 'corner'))

    return min(candidates, key=lambda x: x[0]) if candidates else None


def resolve_circle_collision(pos1, vel1, r1, pos2, vel2, r2):
    """Разрешение столкновения двух кругов"""
    collision_vector = pos1 - pos2
    distance = np.linalg.norm(collision_vector)

    if distance == 0:
        return vel1, vel2

    normal = collision_vector / distance
    rel_vel = vel1 - vel2
    vel_along_normal = np.dot(rel_vel, normal)

    if vel_along_normal > 0:
        return vel1, vel2

    impulse = 2 * vel_along_normal / (r1 + r2)
    new_vel1 = vel1 - impulse * normal * r2
    new_vel2 = vel2 + impulse * normal * r1

    return new_vel1, new_vel2


def update_physics(balls, walls, dt):
    """Обновление физики за временной шаг с сохранением информации о столкновениях"""
    new_balls = [b.copy() for b in balls]
    collision_events = []

    remaining_time = dt

    for iteration in range(10):
        if remaining_time <= 1e-8:
            break

        earliest_t = remaining_time
        collision_info = None

        # Столкновения со стенами
        for i, (pos, vel, r, color) in enumerate(new_balls):
            for A, B in walls:
                result = collide_line_time_with_normal(pos, vel, r, A, B)
                if result is not None and result[0] < earliest_t:
                    earliest_t = result[0]
                    collision_info = ('wall', i, result[1], result[2], A, B)

        # Столкновения между шарами
        for i in range(len(new_balls)):
            for j in range(i + 1, len(new_balls)):
                pos1, vel1, r1, color1 = new_balls[i]
                pos2, vel2, r2, color2 = new_balls[j]
                t_col = collide_circle_time(pos1, vel1, r1, pos2, vel2, r2)
                if t_col is not None and t_col < earliest_t:
                    earliest_t = t_col
                    collision_info = ('circle', i, j)

        if collision_info is None:
            for i in range(len(new_balls)):
                pos, vel, r, color = new_balls[i]
                new_balls[i][0] = pos + vel * remaining_time
            break

        # Двигаем все шары до момента столкновения
        for i in range(len(new_balls)):
            pos, vel, r, color = new_balls[i]
            new_balls[i][0] = pos + vel * earliest_t

        # Обрабатываем столкновение и сохраняем информацию
        if collision_info[0] == 'wall':
            i, normal, col_type, A, B = collision_info[1], collision_info[2], collision_info[3], collision_info[4], \
            collision_info[5]
            collision_point = new_balls[i][0] - normal * new_balls[i][2]
            collision_events.append({
                'type': 'wall',
                'ball_index': i,
                'point': collision_point.copy(),
                'normal': normal.copy(),
                'wall_type': col_type,
                'wall_points': (A.copy(), B.copy())
            })
            new_balls[i][1] = reflect_vector(new_balls[i][1], normal)
        else:
            i, j = collision_info[1], collision_info[2]
            pos1, vel1, r1, color1 = new_balls[i]
            pos2, vel2, r2, color2 = new_balls[j]
            collision_point = (pos1 + pos2) / 2
            collision_events.append({
                'type': 'circle',
                'ball_indices': (i, j),
                'point': collision_point.copy()
            })
            new_vel1, new_vel2 = resolve_circle_collision(pos1, vel1, r1, pos2, vel2, r2)
            new_balls[i][1] = new_vel1
            new_balls[j][1] = new_vel2

        remaining_time -= earliest_t

    return new_balls, collision_events


# ---------------- ПРЕДВЫЧИСЛЕНИЕ КАДРОВ ----------------
frames = []
collision_history = []  # история столкновений для каждого кадра
balls_state = [b.copy() for b in balls]

print("Вычисление физики...")
t0 = time.time()

# Кадр 0 - начальное состояние (без движения)
frames.append([(b[0].copy(), b[1].copy(), b[2], b[3]) for b in balls_state])
collision_history.append([])  # в начальном кадре нет столкновений

# Вычисляем остальные кадры
for frame in range(1, TOTAL_FRAMES):
    balls_state, collisions = update_physics(balls_state, walls, DT)
    frames.append([(b[0].copy(), b[1].copy(), b[2], b[3]) for b in balls_state])
    collision_history.append(collisions)

elapsed = (time.time() - t0) * 1000
print(f"Просчитано {TOTAL_FRAMES} кадров за {elapsed:.1f} мс")

# ---------------- ВИЗУАЛИЗАЦИЯ ----------------
fig, ax = plt.subplots(figsize=(12, 9))
plt.subplots_adjust(bottom=0.2, top=0.95)
ax.set_xlim(-6, 6)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_title("Расширенная визуализация физики столкновений", fontsize=14, pad=20)
ax.grid(True, alpha=0.3)

# Отрисовка стен
for (p1, p2) in walls:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', lw=2)

# Шары
circles = [Circle(b[0], b[2], color=b[3], ec='black', lw=1.5, alpha=0.8) for b in balls]
for c in circles:
    ax.add_patch(c)

# Векторы скорости
velocity_lines = [ax.plot([], [], 'red', lw=2, alpha=0.7)[0] for _ in balls]
velocity_arrows = [ax.plot([], [], 'red', marker='>', markersize=6)[0] for _ in balls]

# Траектории за КАДР (ломанная линия с учетом столкновений)
trajectory_lines = [ax.plot([], [], color=balls[i][3], alpha=0.6, lw=2, linestyle='-')[0] for i in range(len(balls))]

# Полная траектория (все позиции с начала)
full_trajectory_lines = [ax.plot([], [], color=balls[i][3], alpha=0.3, lw=1, linestyle=':')[0] for i in
                         range(len(balls))]

# Визуализация столкновений
collision_points = ax.plot([], [], 'ro', markersize=8, alpha=0, label='Столкновения')[0]
collision_normals = [ax.plot([], [], 'purple', lw=2, alpha=0)[0] for _ in range(5)]  # для нормалей
circle_collision_marks = ax.plot([], [], 'yo', markersize=10, alpha=0, markeredgecolor='orange', markeredgewidth=2)[0]

# Информационный текст
info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)

# Ползунок и кнопки
ax_slider = plt.axes([0.15, 0.12, 0.7, 0.03])
slider = Slider(ax_slider, 'Кадр', 1, TOTAL_FRAMES, valinit=1, valfmt='%d')

# Кнопки управления
button_y = 0.04
button_width = 0.08
ax_prev = plt.axes([0.15, button_y, button_width, 0.04])
ax_play = plt.axes([0.25, button_y, button_width, 0.04])
ax_next = plt.axes([0.35, button_y, button_width, 0.04])
ax_rewind = plt.axes([0.45, button_y, button_width, 0.04])
ax_toggle_traj = plt.axes([0.55, button_y, button_width, 0.04])
ax_toggle_coll = plt.axes([0.65, button_y, button_width, 0.04])
ax_toggle_full_traj = plt.axes([0.75, button_y, button_width, 0.04])

b_prev = Button(ax_prev, '◀◀')
b_play = Button(ax_play, '▶')
b_next = Button(ax_next, '▶▶')
b_rewind = Button(ax_rewind, '⟲')
b_toggle_traj = Button(ax_toggle_traj, 'Траект.кадр')
b_toggle_coll = Button(ax_toggle_coll, 'Столкновения')
b_toggle_full_traj = Button(ax_toggle_full_traj, 'Вся траект.')

# Переменные для управления визуализацией
playing = False
show_trajectories = True
show_collisions = True
show_full_trajectories = False
last_play_time = 0


def calculate_frame_trajectory_with_collisions(frame_idx):
    """Вычисляет траекторию за кадр с учетом всех столкновений (ломанная линия)"""
    if frame_idx <= 1:  # Кадр 0 или 1 - начальное положение
        return [np.array([frames[0][i][0]]) for i in range(len(balls))]

    trajectories = []
    for i in range(len(balls)):
        # Начальная позиция в предыдущем кадре
        start_pos = frames[frame_idx - 1][i][0]
        # Конечная позиция в текущем кадре
        end_pos = frames[frame_idx][i][0]

        # Получаем все столкновения для этого шара в этом кадре
        collisions_in_frame = []
        if frame_idx < len(collision_history):
            for collision in collision_history[frame_idx]:
                if (collision['type'] == 'wall' and collision['ball_index'] == i) or \
                        (collision['type'] == 'circle' and i in collision['ball_indices']):
                    collisions_in_frame.append(collision)

        # Сортируем столкновения по времени (они уже должны быть в правильном порядке)
        if collisions_in_frame:
            # Создаем ломанную траекторию через точки столкновений
            trajectory_points = [start_pos]
            current_pos = start_pos.copy()
            remaining_vector = end_pos - start_pos
            total_distance = np.linalg.norm(remaining_vector)

            if total_distance > 0:
                direction = remaining_vector / total_distance

                for collision in collisions_in_frame:
                    # Позиция столкновения
                    collision_pos = collision['point']
                    # Вектор от текущей позиции до столкновения
                    to_collision = collision_pos - current_pos
                    dist_to_collision = np.linalg.norm(to_collision)

                    if dist_to_collision > 0 and dist_to_collision < total_distance:
                        trajectory_points.append(collision_pos)
                        current_pos = collision_pos.copy()

            # Добавляем конечную точку
            trajectory_points.append(end_pos)
            trajectories.append(np.array(trajectory_points))
        else:
            # Если столкновений не было - прямая линия
            trajectories.append(np.array([start_pos, end_pos]))

    return trajectories


def draw_frame(frame_num):
    """Отрисовка кадра с дополнительной информацией"""
    # frame_num от 1 до TOTAL_FRAMES, но индексы от 0 до TOTAL_FRAMES-1
    i = frame_num - 1

    frame = frames[i]

    # Обновляем позиции шаров
    for j, (pos, vel, r, color) in enumerate(frame):
        circles[j].center = pos

        # Векторы скорости
        vel_scale = 0.3
        velocity_lines[j].set_data([pos[0], pos[0] + vel[0] * vel_scale],
                                   [pos[1], pos[1] + vel[1] * vel_scale])
        velocity_arrows[j].set_data([pos[0] + vel[0] * vel_scale],
                                    [pos[1] + vel[1] * vel_scale])

    # Траектории за КАДР (ломанная линия)
    if show_trajectories and frame_num > 1:  # Для кадра 1 траектории еще нет
        trajectories = calculate_frame_trajectory_with_collisions(i)
        for j, trajectory in enumerate(trajectories):
            if len(trajectory) > 1:
                trajectory_lines[j].set_data(trajectory[:, 0], trajectory[:, 1])
            else:
                trajectory_lines[j].set_data([], [])
    else:
        for line in trajectory_lines:
            line.set_data([], [])

    # Полная траектория (все позиции с начала до текущего кадра)
    if show_full_trajectories and frame_num > 1:
        for j in range(len(balls)):
            all_positions = np.array([frames[k][j][0] for k in range(frame_num)])
            full_trajectory_lines[j].set_data(all_positions[:, 0], all_positions[:, 1])
    else:
        for line in full_trajectory_lines:
            line.set_data([], [])

    # Визуализация столкновений
    if show_collisions and i < len(collision_history):
        collisions = collision_history[i]
        wall_collision_points = []
        circle_collision_points = []
        normal_lines = []

        for collision in collisions:
            if collision['type'] == 'wall':
                wall_collision_points.append(collision['point'])
                # Линия нормали
                normal_start = collision['point']
                normal_end = normal_start + collision['normal'] * 0.5
                normal_lines.append((normal_start, normal_end))
            else:  # circle collision
                circle_collision_points.append(collision['point'])

        # Отображаем точки столкновений
        if wall_collision_points:
            points = np.array(wall_collision_points)
            collision_points.set_data(points[:, 0], points[:, 1])
            collision_points.set_alpha(0.8)
        else:
            collision_points.set_alpha(0)

        if circle_collision_points:
            points = np.array(circle_collision_points)
            circle_collision_marks.set_data(points[:, 0], points[:, 1])
            circle_collision_marks.set_alpha(0.8)
        else:
            circle_collision_marks.set_alpha(0)

        # Отображаем нормали
        for k, line in enumerate(collision_normals):
            if k < len(normal_lines):
                start, end = normal_lines[k]
                line.set_data([start[0], end[0]], [start[1], end[1]])
                line.set_alpha(0.7)
            else:
                line.set_alpha(0)
    else:
        collision_points.set_alpha(0)
        circle_collision_marks.set_alpha(0)
        for line in collision_normals:
            line.set_alpha(0)

    # Информационный текст
    collisions_count = len(collision_history[i]) if i < len(collision_history) else 0
    info_text.set_text(
        f"Кадр: {frame_num}/{TOTAL_FRAMES}\n"
        f"Время: {(frame_num - 1) / FPS:.2f}с\n"
        f"Столкновений: {collisions_count}\n"
        f"Шар 1: v={np.linalg.norm(frame[0][1]):.2f}\n"
        f"Шар 2: v={np.linalg.norm(frame[1][1]):.2f}\n"
        f"Шар 3: v={np.linalg.norm(frame[2][1]):.2f}"
    )

    fig.canvas.draw_idle()


def update(val):
    draw_frame(int(slider.val))


def prev_frame(event):
    slider.set_val(max(1, int(slider.val) - 1))


def next_frame(event):
    slider.set_val(min(TOTAL_FRAMES, int(slider.val) + 1))


def rewind(event):
    slider.set_val(1)


def toggle_trajectories(event):
    global show_trajectories
    show_trajectories = not show_trajectories
    b_toggle_traj.label.set_text('Траект.кадр ✓' if show_trajectories else 'Траект.кадр')
    draw_frame(int(slider.val))


def toggle_collisions(event):
    global show_collisions
    show_collisions = not show_collisions
    b_toggle_coll.label.set_text('Столкновения ✓' if show_collisions else 'Столкновения')
    draw_frame(int(slider.val))


def toggle_full_trajectories(event):
    global show_full_trajectories
    show_full_trajectories = not show_full_trajectories
    b_toggle_full_traj.label.set_text('Вся траект. ✓' if show_full_trajectories else 'Вся траект.')
    draw_frame(int(slider.val))


def play_pause(event):
    global playing, last_play_time
    playing = not playing
    b_play.label.set_text('❚❚' if playing else '▶')
    last_play_time = time.time()


# Подключаем обработчики
slider.on_changed(update)
b_prev.on_clicked(prev_frame)
b_next.on_clicked(next_frame)
b_play.on_clicked(play_pause)
b_rewind.on_clicked(rewind)
b_toggle_traj.on_clicked(toggle_trajectories)
b_toggle_coll.on_clicked(toggle_collisions)
b_toggle_full_traj.on_clicked(toggle_full_trajectories)


# ---------------- АНИМАЦИЯ ----------------
def animate():
    global playing, last_play_time
    frame_interval = 1.0 / FPS

    while plt.fignum_exists(fig.number):
        current_time = time.time()

        if playing and current_time - last_play_time >= frame_interval:
            current_frame = int(slider.val)
            next_frame = current_frame + 1
            if next_frame > TOTAL_FRAMES:
                next_frame = 1  # Зацикливаем анимацию
            slider.set_val(next_frame)
            last_play_time = current_time

        plt.pause(0.01)


# Легенда
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Шар 1'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Шар 2'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Шар 3'),
    plt.Line2D([0], [0], color='red', lw=2, label='Вектор скорости'),
    plt.Line2D([0], [0], color='red', lw=2, linestyle='-', label='Траект. за кадр'),
    plt.Line2D([0], [0], color='red', lw=1, linestyle=':', label='Вся траектория'),
    plt.Line2D([0], [0], marker='o', color='red', markersize=8, label='Столкн. со стеной'),
    plt.Line2D([0], [0], marker='o', color='yellow', markersize=8, label='Столкн. шаров')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

# Запуск
print("Запуск визуализации...")
draw_frame(1)
plt.show(block=False)
animate()