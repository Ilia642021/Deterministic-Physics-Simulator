import numpy as np
from physics_engine import DeterministicPhysicsEngine
import hashlib
import json
import sys

def get_file_hash(filename):
    with open(filename, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def run_simulation(seed, filename):
    # Сложный сценарий
    width, height = 10, 10
    walls = [
        (np.array([-5, -5]), np.array([5, -5])),
        (np.array([-5, 5]), np.array([5, 5])),
        (np.array([-5, -5]), np.array([-5, 5])),
        (np.array([5, -5]), np.array([5, 5])),
        (np.array([-2, -2]), np.array([2, 2])),
    ]
    
    balls = []
    for i in range(10):
        pos = np.array([-4 + i*0.8, 4 - i*0.2])
        vel = np.array([1.5, -2.5])
        balls.append([pos, vel, 0.2, (1, 0, 0, 1)])
        
    engine = DeterministicPhysicsEngine(fps=60, sim_time=5, seed=seed)
    engine.set_objects(balls, walls)
    engine.simulate()
    engine.save_result(filename)
    return engine.compute_state_hash()

def verify():
    print("Проверка детерминизма...")
    
    # Запуск 1
    hash1 = run_simulation(42, 'verify_1.json')
    # Запуск 2
    hash2 = run_simulation(42, 'verify_2.json')
    
    print(f"Хеш 1: {hash1}")
    print(f"Хеш 2: {hash2}")
    
    if hash1 == hash2:
        print("✅ Хеши состояния (последний кадр) совпадают!")
    else:
        print("❌ ОШИБКА: Хеши состояния различаются!")
        # Найдем первый различающийся кадр
        with open('verify_1.json', 'r') as f1, open('verify_2.json', 'r') as f2:
            j1 = json.load(f1)
            j2 = json.load(f2)
            for i, (f1_data, f2_data) in enumerate(zip(j1['frames'], j2['frames'])):
                if f1_data != f2_data:
                    print(f"Первое различие в кадре {i}")
                    # Можно вывести детали...
                    break
        sys.exit(1)
        
    file_hash1 = get_file_hash('verify_1.json')
    file_hash2 = get_file_hash('verify_2.json')
    
    print(f"Хеш файла 1: {file_hash1}")
    print(f"Хеш файла 2: {file_hash2}")
    
    if file_hash1 == file_hash2:
        print("✅ Файлы JSON идентичны!")
    else:
        print("⚠️ Файлы JSON различаются байт-в-байт (вероятно, из-за метаданных).")
        # Сравниваем только кадры
        with open('verify_1.json', 'r') as f1, open('verify_2.json', 'r') as f2:
            j1 = json.load(f1)
            j2 = json.load(f2)
            if j1['frames'] == j2['frames']:
                print("✅ Данные всех кадров идентичны!")
            else:
                print("❌ ОШИБКА: Данные кадров различаются!")
                for i, (f1_data, f2_data) in enumerate(zip(j1['frames'], j2['frames'])):
                    if f1_data != f2_data:
                        print(f"Первое различие в кадре {i}")
                        break
                sys.exit(1)

if __name__ == "__main__":
    verify()
