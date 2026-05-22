# main.py
import os

import numpy as np

from physics_engine import DeterministicPhysicsEngine
from visualizer import analyze_simulation, visualize_simulation
import matplotlib.pyplot as plt


def main():
    # Создание демонстрационной симуляции
    print("Создание симуляции...")
    # Шары: [позиция, скорость, радиус, цвет]
    balls = [
        [np.array([-4.0, -2.0], dtype=np.float64), np.array([2.5, 2.2], dtype=np.float64), 0.25, (1, 0, 0, 0.5)],
        [np.array([0.0, 0.0], dtype=np.float64), np.array([1.5, -2.5], dtype=np.float64), 0.3, (0, 0, 1, 0.6)],
        [np.array([2.5, 2.0], dtype=np.float64), np.array([-2.2, -1.8], dtype=np.float64), 0.2, (0, 0.5, 0, 0.7)]
    ]

    # Стены
    walls = [
        (np.array([-5, -3], dtype=np.float64), np.array([5, -3], dtype=np.float64)),
        (np.array([-5, 3], dtype=np.float64), np.array([5, 3], dtype=np.float64)),
        (np.array([-5, -3], dtype=np.float64), np.array([-5, 3], dtype=np.float64)),
        (np.array([5, -3], dtype=np.float64), np.array([5, 3], dtype=np.float64)),
        (np.array([0, -1], dtype=np.float64), np.array([2, 2], dtype=np.float64))  # наклонная стена
    ]

    # Создание и запуск симуляции
    physics_engine = DeterministicPhysicsEngine(fps=60, sim_time=5, seed=42)  # Уменьшил для теста
    physics_engine.set_objects(balls, walls)
    result = physics_engine.simulate()
    filename = 'simulation_result.json'
    # Сохранение результатов в JSON
    physics_engine.save_result(filename)

    json_size = os.path.getsize(filename) / 1024
    print(f"JSON размер: {json_size:.2f} KB")

    print(f"Создано кадров: {len(result['frames'])}")
    print(f"Первый кадр содержит {len(result['frames'][0]['objects']['circles'])} шаров")


    # Анализ симуляции
    analyze_simulation(filename)
    print("\n" + "=" * 50)

    # Запуск визуализации
    visualize_simulation(filename)

    # Ожидание закрытия окна
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nВизуализация завершена")

    except FileNotFoundError:
        print(f"❌ Файл {filename} не найден!")
        print("Доступные JSON файлы:")
        for f in os.listdir('.'):
            if f.endswith('.json'):
                print(f"  - {f}")


if __name__ == "__main__":
    main()
