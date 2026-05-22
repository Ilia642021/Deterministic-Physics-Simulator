import unittest
import numpy as np
from physics_engine import DeterministicPhysicsEngine

class TestPhysicsEngine(unittest.TestCase):
    def setUp(self):
        self.engine = DeterministicPhysicsEngine(fps=60, sim_time=1.0, seed=42)

    def test_determinism(self):
        """Проверка детерминированности: одинаковый seed должен давать одинаковый хеш."""
        balls = [[np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.2, (1, 0, 0)]]
        walls = [(np.array([-2, -2]), np.array([2, -2])), (np.array([-2, 2]), np.array([2, 2]))]
        
        self.engine.set_objects(balls, walls)
        res1 = self.engine.simulate()
        hash1 = res1['stats']['state_hash']
        
        # Сбрасываем и запускаем снова
        engine2 = DeterministicPhysicsEngine(fps=60, sim_time=1.0, seed=42)
        engine2.set_objects(balls, walls)
        res2 = engine2.simulate()
        hash2 = res2['stats']['state_hash']
        
        self.assertEqual(hash1, hash2, "Хеши состояний при одинаковом seed должны совпадать")

    def test_wall_collision_at_step_end(self):
        """
        Тест граничного условия: удар об стену точно в конце временного шага.
        Шарик должен отразиться, а не пройти насквозь.
        """
        fps = 60
        dt = 1.0 / fps
        # Помещаем шарик так, чтобы при скорости 1.0 он ударился о стену ровно через dt
        # Радиус 0.1, стена на x=1.0. Начальная позиция x = 1.0 - 0.1 - (1.0 * dt)
        radius = 0.1
        wall_x = 1.0
        velocity = np.array([10.0, 0.0]) # Быстрый шар
        start_x = wall_x - radius - (velocity[0] * dt)
        
        balls = [[np.array([start_x, 0.0]), velocity.copy(), radius, (1, 0, 0)]]
        walls = [(np.array([wall_x, -5.0]), np.array([wall_x, 5.0]))]
        
        self.engine.fps = fps
        self.engine.dt = dt
        self.engine.total_frames = 3 # 0, 1, 2
        self.engine.set_objects(balls, walls)
        
        # Симулируем 2 кадра
        # Кадр 0: начальное положение
        # Кадр 1: должен быть ровно в момент удара или сразу после отражения
        result = self.engine.simulate()
        
        # Проверяем положение во втором кадре (индекс 1)
        # В первом кадре (t=dt) он должен быть в точке удара и сменить скорость.
        # Во втором кадре (t=2*dt) он должен двигаться в обратном направлении.
        frame1 = result['frames'][1]
        ball1_pos = frame1['objects']['circles'][0]['position']
        ball1_vel = frame1['objects']['circles'][0]['velocity']
        
        self.assertLessEqual(ball1_pos[0], wall_x - radius + 1e-7, "Шарик прошел сквозь стену!")
        self.assertLess(ball1_vel[0], 0, "Скорость не изменила направление после удара в конце шага")

    def test_ball_ball_collision(self):
        """Тест столкновения двух шаров."""
        balls = [
            [np.array([-1.0, 0.0]), np.array([1.0, 0.0]), 0.5, (1, 0, 0)],
            [np.array([1.0, 0.0]), np.array([-1.0, 0.0]), 0.5, (0, 1, 0)]
        ]
        walls = [] # Нет стен
        
        self.engine.sim_time = 2.0
        self.engine.set_objects(balls, walls)
        result = self.engine.simulate()
        
        # В конце симуляции шары должны разлететься
        final_frame = result['frames'][-1]
        b1_pos = final_frame['objects']['circles'][0]['position']
        b2_pos = final_frame['objects']['circles'][1]['position']
        
        self.assertLess(b1_pos[0], b2_pos[0], "Шары прошли друг сквозь друга!")

    def test_corner_collision(self):
        """Тест столкновения с углом (краем стены)."""
        radius = 0.1
        # Стена от (1, 1) до (2, 1). Шарик летит в точку (1, 1)
        balls = [[np.array([0.0, 0.0]), np.array([1.0, 1.0]), radius, (1, 0, 0)]]
        walls = [(np.array([1.0, 1.0]), np.array([2.0, 1.0]))]
        
        self.engine.sim_time = 2.0
        self.engine.set_objects(balls, walls)
        result = self.engine.simulate()
        
        # Должно быть хотя бы одно столкновение
        self.assertGreater(result['stats']['collision_count'], 0, "Столкновение с углом не зафиксировано")

    def test_ball_stuck_in_wall(self):
        """Тест: шарик уже немного зашел за стену. Он должен отразиться."""
        radius = 0.1
        wall_x = 1.0
        # Шарик на x = 0.95, радиус 0.1. Он зашел на 0.05 в стену (стена на x=1.0).
        # Центр шарика должен быть на 0.9, чтобы не касаться. 
        # Если центр на 0.95, он уже "внутри" на 0.05.
        balls = [[np.array([0.95, 0.0]), np.array([1.0, 0.0]), radius, (1, 0, 0)]]
        walls = [(np.array([wall_x, -5.0]), np.array([wall_x, 5.0]))]
        
        self.engine.sim_time = 0.1
        self.engine.set_objects(balls, walls)
        result = self.engine.simulate()
        
        # Скорость должна стать отрицательной
        final_frame = result['frames'][-1]
        ball_vel = final_frame['objects']['circles'][0]['velocity']
        self.assertLess(ball_vel[0], 0, "Шарик, застрявший в стене, не отразился")
        
        # Позиция не должна бесконечно расти
        ball_pos = final_frame['objects']['circles'][0]['position']
        self.assertLessEqual(ball_pos[0], wall_x - radius + 1e-7)

    def test_multiple_collisions_per_tick(self):
        """Тест нескольких столкновений за один тик."""
        # Узкий коридор. Шарик мечется между стен.
        radius = 0.1
        walls = [
            (np.array([0.0, -1.0]), np.array([0.0, 1.0])),
            (np.array([0.3, -1.0]), np.array([0.3, 1.0]))
        ]
        # Коридор шириной 0.3. Шарик 0.2. Свободного места 0.1.
        # Скорость 10.0. За тик dt=0.016 он пролетит 0.16, что больше чем 0.1.
        balls = [[np.array([0.15, 0.0]), np.array([10.0, 0.0]), radius, (1, 0, 0)]]
        
        self.engine.fps = 60
        self.engine.sim_time = 0.1
        self.engine.set_objects(balls, walls)
        result = self.engine.simulate()
        
        self.assertGreater(result['stats']['collision_count'], 2, "Несколько столкновений за тик не обработаны")

if __name__ == '__main__':
    unittest.main()
