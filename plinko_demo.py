сдimport numpy as np
from physics_engine import DeterministicPhysicsEngine
from visualizer import visualize_simulation, analyze_simulation
import matplotlib.pyplot as plt
import os

def create_plinko_board(width=10, height=15, rows=10, cols=9):
    pegs = []
    x_spacing = width / (cols + 1)
    y_spacing = height / (rows + 1)
    
    for r in range(rows):
        offset = (x_spacing / 2) if r % 2 == 1 else 0
        for c in range(cols):
            x = -width/2 + (c + 1) * x_spacing + offset
            y = height/2 - (r + 1) * y_spacing
            pegs.append([np.array([x, y], dtype=np.float64), 0.15, (0.5, 0.5, 0.5, 1.0)])
            
    # Внешние стены
    walls = [
        (np.array([-width/2, -height/2], dtype=np.float64), np.array([width/2, -height/2], dtype=np.float64)),
        (np.array([-width/2, height/2], dtype=np.float64), np.array([width/2, height/2], dtype=np.float64)),
        (np.array([-width/2, -height/2], dtype=np.float64), np.array([-width/2, height/2], dtype=np.float64)),
        (np.array([width/2, -height/2], dtype=np.float64), np.array([width/2, height/2], dtype=np.float64)),
    ]
    
    return walls, pegs

def main():
    print("Создание симуляции Плинко...")
    width, height = 12, 18
    walls, pegs = create_plinko_board(width=width, height=height, rows=12, cols=10)
    
    # Создаем 50 шаров сверху
    balls = []
    for i in range(50):
        # Случайное (но детерминированное через seed) положение сверху
        x = np.random.RandomState(i).uniform(-width/4, width/4)
        y = height/2 - 0.5 - (i * 0.1)
        pos = np.array([x, y], dtype=np.float64)
        vel = np.array([0, -2.0], dtype=np.float64)
        radius = 0.12
        color = (np.random.RandomState(i).rand(), np.random.RandomState(i+100).rand(), np.random.RandomState(i+200).rand(), 0.8)
        balls.append([pos, vel, radius, color])
        
    engine = DeterministicPhysicsEngine(fps=30, sim_time=15, seed=42)
    engine.set_objects(balls, walls, pegs)
    
    print(f"Запуск симуляции: {len(balls)} шаров, {len(pegs)} пегов, {len(walls)} стен")
    engine.simulate()
    
    filename = 'plinko_result.json'
    engine.save_result(filename)
    
    print("\nЗапуск визуализации...")
    visualize_simulation(filename)
    plt.show()

if __name__ == "__main__":
    main()
