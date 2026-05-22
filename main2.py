# main.py

import sys
import os

import numpy as np

from physics_engine import DeterministicPhysicsEngine
from visualizer import analyze_simulation, visualize_simulation
import matplotlib.pyplot as plt


def main():
    # Создание сложной демонстрационной симуляции
    print("Создание сложной симуляции...")

    # Больше шаров с разными свойствами
    balls = [
        # Быстрые маленькие шары
        [np.array([-4.5, -2.5], dtype=np.float64), np.array([3.0, 2.8], dtype=np.float64), 0.15, (1, 0, 0, 0.7)],
        [np.array([4.0, 2.0], dtype=np.float64), np.array([-2.8, -2.5], dtype=np.float64), 0.12, (0, 1, 0, 0.7)],
        [np.array([0.5, -2.0], dtype=np.float64), np.array([1.5, 3.2], dtype=np.float64), 0.18, (0, 0, 1, 0.7)],

        # Медленные большие шары
        [np.array([-3.0, 0.5], dtype=np.float64), np.array([0.8, 1.2], dtype=np.float64), 0.4, (1, 0.5, 0, 0.6)],
        [np.array([2.5, -1.0], dtype=np.float64), np.array([-1.2, 0.8], dtype=np.float64), 0.35, (0.5, 0, 0.5, 0.6)],

        # Средние шары со сложными траекториями
        [np.array([-1.4, 2.0], dtype=np.float64), np.array([2.2, -1.5], dtype=np.float64), 0.25, (1, 1, 0, 0.8)],
        [np.array([1.5, 1.5], dtype=np.float64), np.array([-1.8, 2.2], dtype=np.float64), 0.22, (0, 1, 1, 0.8)],
        [np.array([-2.0, -1.0], dtype=np.float64), np.array([1.2, -2.8], dtype=np.float64), 0.28, (1, 0, 1, 0.8)],

        # Дополнительные шары для большего хаоса
        [np.array([3.0, 0.5], dtype=np.float64), np.array([-2.5, 1.5], dtype=np.float64), 0.2, (0.8, 0.2, 0.2, 0.9)],
        [np.array([-3.5, 1.5], dtype=np.float64), np.array([1.8, -1.2], dtype=np.float64), 0.16, (0.2, 0.8, 0.2, 0.9)],
    ]

    # Сложная система стен - лабиринт с препятствиями
    walls = [
        # Внешние стены
        (np.array([-6, -4], dtype=np.float64), np.array([6, -4], dtype=np.float64)),
        (np.array([-6, 4], dtype=np.float64), np.array([6, 4], dtype=np.float64)),
        (np.array([-6, -4], dtype=np.float64), np.array([-6, 4], dtype=np.float64)),
        (np.array([6, -4], dtype=np.float64), np.array([6, 4], dtype=np.float64)),

        # Центральные препятствия
        (np.array([-3, -2], dtype=np.float64), np.array([-1, 0], dtype=np.float64)),
        (np.array([1, -1], dtype=np.float64), np.array([3, 1], dtype=np.float64)),
        (np.array([-2, 2], dtype=np.float64), np.array([0, 3], dtype=np.float64)),
        (np.array([2, -3], dtype=np.float64), np.array([4, -1], dtype=np.float64)),

        # Вертикальные перегородки
        (np.array([-4, -3], dtype=np.float64), np.array([-4, -1], dtype=np.float64)),
        (np.array([-2, 1], dtype=np.float64), np.array([-2, 3], dtype=np.float64)),
        (np.array([0, -3], dtype=np.float64), np.array([0, -1], dtype=np.float64)),
        (np.array([2, 1], dtype=np.float64), np.array([2, 3], dtype=np.float64)),
        (np.array([4, -3], dtype=np.float64), np.array([4, -1], dtype=np.float64)),

        # Горизонтальные перегородки
        (np.array([-5, -2], dtype=np.float64), np.array([-3, -2], dtype=np.float64)),
        (np.array([-1, -2], dtype=np.float64), np.array([1, -2], dtype=np.float64)),
        (np.array([3, -2], dtype=np.float64), np.array([5, -2], dtype=np.float64)),
        (np.array([-5, 0], dtype=np.float64), np.array([-3, 0], dtype=np.float64)),
        (np.array([-1, 0], dtype=np.float64), np.array([1, 0], dtype=np.float64)),
        (np.array([3, 0], dtype=np.float64), np.array([5, 0], dtype=np.float64)),
        (np.array([-5, 2], dtype=np.float64), np.array([-3, 2], dtype=np.float64)),
        (np.array([-1, 2], dtype=np.float64), np.array([1, 2], dtype=np.float64)),
        (np.array([3, 2], dtype=np.float64), np.array([5, 2], dtype=np.float64)),

        # Диагональные препятствия для усложнения траекторий
        (np.array([-3.5, -1], dtype=np.float64), np.array([-2.5, 0], dtype=np.float64)),
        (np.array([2.5, -0.5], dtype=np.float64), np.array([3.5, 0.5], dtype=np.float64)),
        (np.array([-1, -0.5], dtype=np.float64), np.array([0, 0.5], dtype=np.float64)),
        (np.array([1.5, 2.5], dtype=np.float64), np.array([2.5, 3.5], dtype=np.float64)),
    ]

    # Создание и запуск симуляции с увеличенным временем
    physics_engine = DeterministicPhysicsEngine(fps=30, sim_time=600, seed=42)
    physics_engine.set_objects(balls, walls)
    result = physics_engine.simulate()

    # Сохранение результатов в JSON
    physics_engine.save_result('complex_simulation_result.json')

    json_size = os.path.getsize('complex_simulation_result.json') / 1024
    print(f"JSON размер: {json_size:.2f} KB")

    print(f"Создано кадров: {len(result['frames'])}")
    print(f"Первый кадр содержит {len(result['frames'][0]['objects']['circles'])} шаров")
    print(f"Количество стен: {len(walls)}")

    # Определение файла для визуализации
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Используем новый файл по умолчанию
        filename = 'complex_simulation_result.json'

    try:
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