import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# ---------- ФИЗИКА ----------

def collide_circle_time(p, v, r, c_pos, c_r):
    dp = p - c_pos
    a = np.dot(v, v)
    b = 2 * np.dot(dp, v)
    c = np.dot(dp, dp) - (r + c_r) ** 2
    if a == 0:
        return None
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None
    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    t_candidates = [t for t in (t1, t2) if t > 1e-8]
    return min(t_candidates) if t_candidates else None


def collide_line_time(p, v, r, A, B):
    # Вектор отрезка и его длина
    AB = B - A
    AB_len = np.linalg.norm(AB)
    if AB_len < 1e-8:
        return None

    # Нормаль к отрезку (единичный вектор)
    n = np.array([AB[1], -AB[0]]) / AB_len

    # Расстояние от центра круга до линии
    dist_to_line = np.dot(p - A, n)

    # Компонента скорости, перпендикулярная линии
    vn = np.dot(v, n)

    # Время столкновения с бесконечной линией
    if abs(vn) < 1e-10:
        t_line = None
    else:
        t_line = (r - dist_to_line) / vn if vn > 0 else (-r - dist_to_line) / vn

    # Проверяем столкновение с торцами отрезка (кругами в точках A и B)
    t_A = collide_circle_time(p, v, r, A, 0)  # радиус 0 для точечного столкновения
    t_B = collide_circle_time(p, v, r, B, 0)

    # Собираем все возможные времена столкновения
    candidates = []

    # Столкновение с бесконечной линией
    if t_line is not None and t_line > 1e-8:
        # Проверяем, попадает ли точка контакта в пределы отрезка
        contact_point = p + v * t_line - n * (r if dist_to_line > 0 else -r)
        proj = np.dot(contact_point - A, AB) / (AB_len ** 2)
        if 0 <= proj <= 1:
            candidates.append((t_line, n))

    # Столкновение с точкой A
    if t_A is not None and t_A > 1e-8:
        normal_A = (p + v * t_A - A)
        normal_A_len = np.linalg.norm(normal_A)
        if normal_A_len > 1e-8:
            normal_A /= normal_A_len
            candidates.append((t_A, normal_A))

    # Столкновение с точкой B
    if t_B is not None and t_B > 1e-8:
        normal_B = (p + v * t_B - B)
        normal_B_len = np.linalg.norm(normal_B)
        if normal_B_len > 1e-8:
            normal_B /= normal_B_len
            candidates.append((t_B, normal_B))

    # Возвращаем самое раннее столкновение
    if candidates:
        t_min, normal_min = min(candidates, key=lambda x: x[0])
        return t_min
    return None


def reflect_velocity(v, n):
    n = n / np.linalg.norm(n)
    return v - 2 * np.dot(v, n) * n


def simulate_path(p, v, r, circles, lines, dt):
    path = [p.copy()]
    impacts = []
    t_remain = dt

    while t_remain > 1e-6:
        earliest_t = t_remain
        normal = None
        hit_point = None
        collision_type = None

        # столкновение с кругом
        for c_pos, c_r in circles:
            t_col = collide_circle_time(p, v, r, c_pos, c_r)
            if t_col is not None and t_col < earliest_t:
                earliest_t = t_col
                hit_point = p + v * t_col
                normal = (hit_point - c_pos) / np.linalg.norm(hit_point - c_pos)
                collision_type = "circle"

        # столкновение с линией (используем исправленную функцию)
        for A, B in lines:
            result = collide_line_time_with_normal(p, v, r, A, B)
            if result is not None and result[0] < earliest_t:
                earliest_t = result[0]
                normal = result[1]
                collision_type = "line"

        # если не было столкновений
        if collision_type is None:
            p = p + v * t_remain
            path.append(p.copy())
            break

        # иначе двигаем до столкновения
        p = p + v * earliest_t
        path.append(p.copy())
        impacts.append(p.copy())
        v = reflect_velocity(v, normal)
        t_remain -= earliest_t

    return np.array(path), impacts, v


# Вспомогательная функция, которая возвращает и время, и нормаль
def collide_line_time_with_normal(p, v, r, A, B):
    # Вектор отрезка и его длина
    AB = B - A
    AB_len = np.linalg.norm(AB)
    if AB_len < 1e-8:
        return None

    # Единичный вектор вдоль отрезка
    AB_unit = AB / AB_len

    # Две возможные нормали (обе стороны отрезка)
    n1 = np.array([AB[1], -AB[0]]) / AB_len  # нормаль 1
    n2 = -n1  # нормаль 2 (противоположная сторона)

    # Определяем, с какой стороны от отрезка находится круг
    dist_to_line = np.dot(p - A, n1)

    # Выбираем нормаль, направленную ОТ отрезка (в сторону, где должен быть круг)
    if dist_to_line >= 0:
        n = n1  # круг с одной стороны
        effective_dist = dist_to_line
    else:
        n = n2  # круг с другой стороны
        effective_dist = -dist_to_line

    # Компонента скорости по направлению к отрезку
    vn = np.dot(v, n)

    # Если скорость направлена от отрезка - столкновения не будет
    if vn >= 0:
        t_line = None
    else:
        # Время столкновения: (расстояние - радиус) / скорость сближения
        t_line = (effective_dist - r) / (-vn)

    # Проверяем столкновение с торцами отрезка
    t_A = collide_circle_time(p, v, r, A, 1e-8)  # маленький радиус для стабильности
    t_B = collide_circle_time(p, v, r, B, 1e-8)

    candidates = []

    # Столкновение с линией
    if t_line is not None and t_line > 1e-8:
        # Проверяем, попадает ли точка контакта в пределы отрезка
        contact_point = p + v * t_line
        # Проекция на отрезок
        AP = contact_point - A
        proj = np.dot(AP, AB_unit)
        if 0 <= proj <= AB_len:
            candidates.append((t_line, n))

    # Столкновение с точкой A
    if t_A is not None and t_A > 1e-8:
        # Нормаль от точки A к кругу
        collision_point = p + v * t_A
        normal_A = collision_point - A
        normal_A_len = np.linalg.norm(normal_A)
        if normal_A_len > 1e-8:
            normal_A /= normal_A_len
            candidates.append((t_A, normal_A))

    # Столкновение с точкой B
    if t_B is not None and t_B > 1e-8:
        # Нормаль от точки B к кругу
        collision_point = p + v * t_B
        normal_B = collision_point - B
        normal_B_len = np.linalg.norm(normal_B)
        if normal_B_len > 1e-8:
            normal_B /= normal_B_len
            candidates.append((t_B, normal_B))

    return min(candidates, key=lambda x: x[0]) if candidates else None


# ---------- СЦЕНА ----------

ball_r = 0.25
ball_start = np.array([-4.0, -1.5])
ball_vel = np.array([3.0, 1.1])

circles = [
    (np.array([-3.0, -0.4]), 0.6),
    (np.array([-2.1, -1.5]), 0.6),
    (np.array([-3.5, -2.7]), 0.6)
]

lines = [
    (np.array([-5.0, -3.0]), np.array([5.0, -3.0])),  # нижняя
    (np.array([-5.0, 3.0]), np.array([5.0, 3.0])),  # верхняя
    (np.array([-5.0, -3.0]), np.array([-5.0, 3.0])),  # левая
    (np.array([5.0, -3.0]), np.array([5.0, 3.0])),  # правая
    (np.array([-3.9, 2.16]), np.array([-2.0, 2.5]))
]

# ---------- ВИЗУАЛИЗАЦИЯ ----------

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
ax.set_xlim(-6, 6)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.set_title("Множественные столкновения (круги + линии)")

# препятствия
for c_pos, c_r in circles:
    circ = plt.Circle(c_pos, c_r, color='gray', alpha=0.4)
    ax.add_patch(circ)

for A, B in lines:
    ax.plot([A[0], B[0]], [A[1], B[1]], color='black', lw=2)

# шар и путь
ball = plt.Circle(ball_start, ball_r, color='dodgerblue')
ax.add_patch(ball)
path_line, = ax.plot([], [], 'r--', lw=1)
impact_marks, = ax.plot([], [], 'rx', markersize=5)

# ползунок времени
ax_dt = plt.axes([0.2, 0.1, 0.6, 0.03])
dt_slider = Slider(ax_dt, 'Δt', 0.0, 20.0, valinit=1.0, valstep=0.05)


def update(val):
    dt = dt_slider.val
    path, impacts, v_end = simulate_path(ball_start, ball_vel, ball_r, circles, lines, dt)
    path_line.set_data(path[:, 0], path[:, 1])
    if len(impacts) > 0:
        imps = np.array(impacts)
        impact_marks.set_data(imps[:, 0], imps[:, 1])
    else:
        impact_marks.set_data([], [])
    ball.center = path[-1]
    fig.canvas.draw_idle()


dt_slider.on_changed(update)
update(1.0)
plt.show()
