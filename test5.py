import json
import matplotlib.pyplot as plt
import numpy as np

# Загружаем JSON (предполагаем, файл в текущей папке)
json_file = 'simulation_result.json'
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("✅ JSON загружен")
except FileNotFoundError:
    print(f"❌ Файл {json_file} не найден. Убедитесь, что он в папке.")
    exit(1)

# Извлекаем implementation
try:
    implementation_code = data['visualization_schema']['custom_renderers']['custom_arrowhead']['implementation']
    print("✅ Implementation извлечена")
except KeyError:
    print("❌ custom_renderers или custom_arrowhead не найдены в JSON")
    exit(1)

# Компилируем код (как в UniversalRenderer)
local_namespace = {}
global_namespace = {
    'np': np,
    'plt': plt,
    'matplotlib': plt,
    'Circle': plt.Circle,
    'Polygon': plt.matplotlib.patches.Polygon,
    'Line2D': plt.Line2D
}

try:
    exec(implementation_code, global_namespace, local_namespace)
    print("✅ Код скомпилирован")
except Exception as e:
    print(f"❌ Ошибка компиляции: {e}")
    exit(1)

# Ищем функцию
if 'draw_arrowhead' in local_namespace:
    draw_arrowhead_func = local_namespace['draw_arrowhead']
    print("✅ Функция draw_arrowhead найдена")
else:
    print("❌ Функция draw_arrowhead не найдена")
    exit(1)

# Извлекаем параметры из frames[0].objects.velocity_arrows[0]
try:
    frame = data['frames'][0]
    velocity_arrow = frame['objects']['velocity_arrows'][0]
    test_position = velocity_arrow['position']  # list, e.g. [-4.0, -2.0]
    test_velocity = velocity_arrow['velocity']  # list, e.g. [2.5, 2.2]
    test_scale = velocity_arrow['scale']  # float, e.g. 0.4
    test_color = velocity_arrow['color']  # str, e.g. 'darkred'
    print(data['visualization_schema']['object_types']['velocity_arrow']['drawing_instructions']['steps'][1]['args'][
              'head_size'])  # -> 0.5 * @scale
    test_head_size = 0.5 * test_scale
    print(
        f"✅ Параметры извлечены: pos={test_position}, vel={test_velocity}, scale={test_scale}, color={test_color}, head_size={test_head_size}")
except KeyError:
    print("❌ velocity_arrows не найдены в frames[0]")
    exit(1)

# Создаём тестовую фигуру
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-6, 6)  # Как в config.world_bounds
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True)
ax.set_title("Тест наконечника из JSON")

# Рисуем линию стрелки (shaft) для сравнения
shaft_end = [test_position[0] + test_velocity[0] * test_scale, test_position[1] + test_velocity[1] * test_scale]
ax.plot([test_position[0], shaft_end[0]], [test_position[1], shaft_end[1]], color=test_color, linewidth=2, zorder=5)
print("✅ Линия стрелки нарисована")

# Вызываем функцию для наконечника
try:
    result = draw_arrowhead_func(ax, test_position, test_velocity, test_scale, test_color, test_head_size)
    print(f"Функция вернула: {result}")
    if result:
        print("✅ Наконечник должен быть виден на графике")
    else:
        print("⚠️ Функция вернула None (velocity слишком мал)")
except Exception as e:
    print(f"❌ Ошибка при вызове: {e}")
    import traceback

    traceback.print_exc()

# Показываем график
plt.show()
