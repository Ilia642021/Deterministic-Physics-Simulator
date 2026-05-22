import os
import numpy as np
import hashlib
import time
import json
from typing import List, Dict, Any


class NumpyEncoder(json.JSONEncoder):
    """Кастомный энкодер для преобразования numpy типов в стандартные Python типы"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class VisualizationTemplate:
    """Шаблон для создания универсальной схемы визуализации"""

    @staticmethod
    def create_empty_schema():
        """Создает пустой шаблон схемы визуализации"""
        return {
            'metadata': {
                'version': '1.0',
                'simulation_type': 'custom',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'schema_version': '1.0'
            },
            'config': {},
            'stats': {},
            'visualization_schema': {
                'display_modes': {},
                'object_types': {},
                'custom_renderers': {},
                'layers': {
                    'background': 0,
                    'trajectories': 5,
                    'objects': 10,
                    'overlay': 20
                }
            },
            'frames': [],
            'static_objects': {
                'walls': [],
                'info_panels': []
            }
        }

    @staticmethod
    def create_standard_physics_schema():
        """Создает стандартную схему для физических симуляций с шарами"""
        schema = VisualizationTemplate.create_empty_schema()

        # Добавляем стандартные режимы отображения
        schema['visualization_schema']['display_modes'] = {
            'velocity_vectors': {
                'name': 'Векторы скорости',
                'default': True,
                'description': 'Показывает векторы скорости объектов'
            },
            'trajectory_per_frame': {
                'name': 'Траектория за кадр',
                'default': True,
                'description': 'Путь объекта между кадрами'
            },
            'full_trajectory': {
                'name': 'Полная траектория',
                'default': False,
                'description': 'Вся история движения'
            },
            'collision_points': {
                'name': 'Точки столкновений',
                'default': True,
                'description': 'Места столкновений объектов'
            },
            'collision_normals': {
                'name': 'Нормали столкновений',
                'default': False,
                'description': 'Направления отскоков при столкновениях'
            }
        }

        # Добавляем стандартные типы объектов
        schema['visualization_schema']['object_types'].update({
            'circle': {
                'render_method': 'primitive_circle',
                'parameters': ['position', 'radius', 'color', 'edgecolor', 'linewidth'],
                'drawing_instructions': {
                    'library_method': 'matplotlib.patches.Circle',
                    'constructor_args': {
                        'xy': '@position',
                        'radius': '@radius',
                        'facecolor': '@color',  # Теперь @color — RGBA-кортеж
                        'edgecolor': '@edgecolor',
                        'linewidth': '@linewidth'
                    },
                    'zorder': 10
                }
            },
            'wall': {
                'render_method': 'primitive_line',
                'parameters': ['start_pos', 'end_pos', 'color', 'linewidth', 'style'],
                'drawing_instructions': {
                    'library_method': 'matplotlib.lines.Line2D',
                    'constructor_args': {
                        'xdata': ['@start_pos[0]', '@end_pos[0]'],
                        'ydata': ['@start_pos[1]', '@end_pos[1]'],
                        'color': '@color',
                        'linewidth': '@linewidth',
                        'linestyle': '@style'
                    },
                    'zorder': 8
                }
            },
            'velocity_arrow': {
                'render_method': 'composite_arrow',
                'parameters': ['position', 'velocity', 'scale', 'color', 'linewidth'],
                'tags': ['velocity_vectors'],
                'drawing_instructions': {
                    'steps': [
                        {
                            'type': 'arrow_shaft',
                            'method': 'matplotlib.lines.Line2D',
                            'args': {
                                'xdata': [
                                    '@position[0]',
                                    '@position[0] + @velocity[0] * @scale'
                                ],
                                'ydata': [
                                    '@position[1]',
                                    '@position[1] + @velocity[1] * @scale'
                                ],
                                'color': '@color',
                                'linewidth': '@linewidth'
                            }
                        },
                        {
                            'type': 'arrow_head',
                            'method': 'custom_arrowhead',
                            'args': {
                                'position': '@position',
                                'velocity': '@velocity',
                                'scale': '@scale',
                                'color': '@color',
                                'head_size': '0.3 * @scale'
                            }
                        }
                    ],
                    'zorder': 9
                }
            },
            'hit_point': {
                'render_method': 'primitive_marker',
                'parameters': ['position', 'size', 'color', 'edgecolor', 'marker_style'],
                'tags': ['collision_points'],
                'drawing_instructions': {
                    'library_method': 'matplotlib.lines.Line2D',
                    'constructor_args': {
                        'xdata': ['@position[0]'],
                        'ydata': ['@position[1]'],
                        'marker': '@marker_style',
                        'markersize': '@size',
                        'markerfacecolor': '@color',
                        'markeredgecolor': '@edgecolor',
                        'linestyle': 'None'
                    },
                    'zorder': 11
                }
            },
            'trajectory_line': {
                'render_method': 'primitive_polyline',
                'parameters': ['points', 'color', 'linewidth', 'linestyle', 'alpha'],
                'tags': ['trajectory_per_frame', 'full_trajectory'],
                'drawing_instructions': {
                    'library_method': 'matplotlib.lines.Line2D',
                    'constructor_args': {
                        'xdata': '@points[:, 0]',
                        'ydata': '@points[:, 1]',
                        'color': '@color',
                        'linewidth': '@linewidth',
                        'linestyle': '@linestyle',
                        'alpha': '@alpha'
                    },
                    'zorder': 5
                }
            },
            'collision_normal': {
                'render_method': 'primitive_line',
                'parameters': ['start_pos', 'end_pos', 'color', 'linewidth'],
                'tags': ['collision_normals'],
                'drawing_instructions': {
                    'library_method': 'matplotlib.lines.Line2D',
                    'constructor_args': {
                        'xdata': ['@start_pos[0]', '@end_pos[0]'],
                        'ydata': ['@start_pos[1]', '@end_pos[1]'],
                        'color': '@color',
                        'linewidth': '@linewidth',
                        'linestyle': '--'
                    },
                    'zorder': 7
                }
            }
        })

        # Добавляем кастомные рендереры
        schema['visualization_schema']['custom_renderers'] = {
            'custom_arrowhead': {
                'description': 'Рисует треугольный наконечник стрелки',
                'implementation':
                    '''
                    def draw_arrowhead(ax, position, velocity, scale, color, head_size):
                        try:
                            import numpy as np
                            from matplotlib.patches import Polygon
                    
                            position = np.array(position, dtype=float)
                            velocity = np.array(velocity, dtype=float)
                            head_size = float(head_size)
                    
                            if np.linalg.norm(velocity) < 1e-10:
                                return None
                    
                            tip_pos = position + velocity * scale
                            direction = velocity / np.linalg.norm(velocity)
                            perpendicular = np.array([-direction[1], direction[0]])
                    
                            points = [
                                tip_pos,
                                tip_pos - direction * head_size + perpendicular * head_size * 0.6,
                                tip_pos - direction * head_size - perpendicular * head_size * 0.6
                            ]
                    
                            arrowhead = Polygon(points, color=color, linewidth=0.1, zorder=10)
                            ax.add_patch(arrowhead)
                            return arrowhead
                        except Exception as e:
                            print(f"Error in draw_arrowhead: {e}")
                            import traceback
                            traceback.print_exc()
                            return None
                    '''
            }
        }

        return schema


class DeterministicPhysicsEngine:
    def __init__(self, fps: int = 60, sim_time: float = 5, seed: int = 42):
        self.fps = fps
        self.sim_time = sim_time
        self.dt = 1.0 / fps
        self.total_frames = int(sim_time * fps) + 1
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Создаем шаблон результата
        self.result_template = VisualizationTemplate.create_standard_physics_schema()

        # Статистика
        self.numerical_issues = 0
        self.collision_count = 0
        self.min_time_step = self.dt
        self.max_time_step = 0

        # Инициализируем balls, walls и pegs как None
        self.balls = None
        self.walls = None
        self.pegs = None

    def set_objects(self, balls: List, walls: List, pegs: List = None):
        """Установка объектов симуляции и заполнение статических данных"""
        self.balls = [b.copy() for b in balls]
        self.walls = walls
        self.pegs = pegs if pegs is not None else []

        # Заполняем статические объекты в шаблоне
        self._fill_static_objects()

    def _calculate_world_bounds(self):
        """Автоматически вычисляет границы мира на основе стен, шаров и пегов"""
        if not self.walls and not self.pegs:
            return {'xmin': -6, 'xmax': 6, 'ymin': -4, 'ymax': 4}

        # Собираем все точки стен
        all_points = []
        for A, B in self.walls:
            all_points.append(A)
            all_points.append(B)

        # Собираем позиции и радиусы шаров
        ball_positions = []
        ball_radii = []
        if self.balls:
            for ball in self.balls:
                ball_positions.append(ball[0])  # позиция
                ball_radii.append(ball[2])  # радиус
        
        # Собираем позиции и радиусы пегов
        if self.pegs:
            for peg in self.pegs:
                all_points.append(peg[0])
                ball_radii.append(peg[1])

        # Преобразуем в numpy массив для удобства
        points_array = np.array(all_points)

        # Вычисляем границы
        xmin = np.min(points_array[:, 0])
        xmax = np.max(points_array[:, 0])
        ymin = np.min(points_array[:, 1])
        ymax = np.max(points_array[:, 1])

        # Учитываем радиусы шаров и пегов
        if ball_radii:
            max_radius = max(ball_radii)
            xmin -= max_radius
            xmax += max_radius
            ymin -= max_radius
            ymax += max_radius

        # Добавляем небольшой отступ (% от размера)
        x_padding = (xmax - xmin) * 0.05
        y_padding = (ymax - ymin) * 0.05

        return {
            'xmin': float(xmin - x_padding),
            'xmax': float(xmax + x_padding),
            'ymin': float(ymin - y_padding),
            'ymax': float(ymax + y_padding)
        }

    def _fill_static_objects(self):
        """Заполняет статические объекты в схеме"""
        # Очищаем предыдущие статические объекты
        self.result_template['static_objects']['walls'] = []
        if 'pegs' not in self.result_template['static_objects']:
            self.result_template['static_objects']['pegs'] = []
        else:
            self.result_template['static_objects']['pegs'] = []

        # Добавляем стены
        for i, (A, B) in enumerate(self.walls):
            wall_obj = {
                'type': 'wall',
                'id': f'wall_{i}',
                'start_pos': A.tolist() if hasattr(A, 'tolist') else A,
                'end_pos': B.tolist() if hasattr(B, 'tolist') else B,
                'color': 'black',
                'linewidth': 2,
                'style': '-',
                'tags': ['always']
            }
            self.result_template['static_objects']['walls'].append(wall_obj)

        # Добавляем пеги
        for i, (pos, r, color) in enumerate(self.pegs):
            peg_obj = {
                'type': 'circle',
                'id': f'peg_{i}',
                'position': pos.tolist() if hasattr(pos, 'tolist') else pos,
                'radius': float(r),
                'color': color,
                'edgecolor': 'black',
                'linewidth': 1.5,
                'tags': ['always'],
                'properties': {'static': True}
            }
            self.result_template['static_objects']['pegs'].append(peg_obj)

        # Вычисляем автоматические границы мира
        world_bounds = self._calculate_world_bounds()

        # Заполняем конфигурацию
        self.result_template['config'] = {
            'fps': self.fps,
            'sim_time': self.sim_time,
            'total_frames': self.total_frames,
            'seed': self.seed,
            'world_bounds': world_bounds,  # Теперь вычисляется автоматически
            'dt': self.dt
        }


    def _create_frame_data(self, balls_state, frame_idx, collisions=None, trajectory_points=None):
        """Создает структуру данных для одного кадра"""
        frame_time = frame_idx / self.fps

        frame_data = {
            'frame_number': frame_idx,
            'timestamp': frame_time,
            'objects': {
                'circles': [],
                'velocity_arrows': []
            },
            'events': {
                'collisions': [],
                'trajectories': []
            },
            'statistics': {
                'collisions_this_frame': len(collisions) if collisions else 0
            }
        }

        # Добавляем шары и векторы скорости (без изменений)
        for i, ball_data in enumerate(balls_state):
            pos, vel, r, color = ball_data
            if isinstance(color, (list, tuple)) and len(color) == 3:
                color = tuple(color) + (1,)

            circle_obj = {
                'type': 'circle',
                'id': f'ball_{i}',
                'position': pos.tolist() if hasattr(pos, 'tolist') else pos,
                'velocity': vel.tolist() if hasattr(vel, 'tolist') else vel,
                'radius': float(r),
                'color': color,
                'edgecolor': 'black',
                'linewidth': 1,
                'tags': ['always'],
                'properties': {
                    'speed': float(np.linalg.norm(vel)),
                    'mass': float(r * 10)
                }
            }
            frame_data['objects']['circles'].append(circle_obj)

            velocity_arrow = {
                'type': 'velocity_arrow',
                'id': f'velocity_{i}',
                'position': pos.tolist() if hasattr(pos, 'tolist') else pos,
                'velocity': vel.tolist() if hasattr(vel, 'tolist') else vel,
                'scale': 0.4,
                'color': 'darkred',
                'linewidth': 1,
                'tags': ['velocity_vectors']
            }
            frame_data['objects']['velocity_arrows'].append(velocity_arrow)

        # Добавляем столкновения (без изменений)
        if collisions:
            for collision in collisions:
                if collision['type'] == 'wall':
                    hit_point = {
                        'type': 'hit_point',
                        'id': f'collision_wall_{collision["ball_index"]}_{frame_idx}',
                        'position': collision['point'].tolist() if hasattr(collision['point'], 'tolist') else collision[
                            'point'],
                        'size': 8,
                        'color': 'red',
                        'edgecolor': 'darkred',
                        'marker_style': 'o',
                        'tags': ['collision_points']
                    }
                    frame_data['events']['collisions'].append(hit_point)

                    # Добавляем нормаль столкновения
                    normal_end = collision['point'] + collision['normal'] * 0.8
                    normal_line = {
                        'type': 'collision_normal',
                        'id': f'normal_{collision["ball_index"]}_{frame_idx}',
                        'start_pos': collision['point'].tolist() if hasattr(collision['point'], 'tolist') else
                        collision['point'],
                        'end_pos': normal_end.tolist() if hasattr(normal_end, 'tolist') else normal_end,
                        'color': 'purple',
                        'linewidth': 2,
                        'tags': ['collision_normals']
                    }
                    frame_data['events']['collisions'].append(normal_line)

                elif collision['type'] == 'circle':
                    hit_point = {
                        'type': 'hit_point',
                        'id': f'collision_circle_{collision["ball_indices"][0]}_{collision["ball_indices"][1]}_{frame_idx}',
                        'position': collision['point'].tolist() if hasattr(collision['point'], 'tolist') else collision[
                            'point'],
                        'size': 10,
                        'color': 'yellow',
                        'edgecolor': 'orange',
                        'marker_style': 'o',
                        'tags': ['collision_points']
                    }
                    frame_data['events']['collisions'].append(hit_point)

        # Добавляем траектории через центры кругов
        if trajectory_points and frame_idx > 0:
            for i, points in enumerate(trajectory_points):
                if len(points) > 1:  # Если есть промежуточные точки
                    # Преобразуем точки в список
                    point_list = []
                    for point in points:
                        if hasattr(point, 'tolist'):
                            point_list.append(point.tolist())
                        else:
                            point_list.append(point)

                    color = balls_state[i][3]
                    trajectory_line = {
                        'type': 'trajectory_line',
                        'id': f'trajectory_frame_{i}_{frame_idx}',
                        'points': point_list,
                        'color': color,
                        'linewidth': 2,
                        'linestyle': '-',
                        'alpha': 0.7,
                        'tags': ['trajectory_per_frame']
                    }
                    frame_data['events']['trajectories'].append(trajectory_line)

        return frame_data

    def _add_trajectory_data(self, frame_data, balls_state, frame_idx, collisions=None):
        """Добавляет данные траекторий в кадр с учетом столкновений"""
        if frame_idx == 0:
            return  # Для первого кадра нет траектории

        for i, ball_data in enumerate(balls_state):
            current_pos = ball_data[0]

            # Получаем предыдущую позицию из предыдущего кадра
            prev_frame = self.result_template['frames'][frame_idx - 1]
            prev_ball = prev_frame['objects']['circles'][i]
            prev_pos = prev_ball['position']

            # Создаем точки траектории
            trajectory_points = [prev_pos]

            # Если были столкновения в этом кадре, добавляем точки столкновений
            if collisions:
                # Фильтруем столкновения для текущего шара
                ball_collisions = []
                for collision in collisions:
                    if (collision['type'] == 'wall' and collision['ball_index'] == i) or \
                            (collision['type'] == 'circle' and i in collision['ball_indices']):
                        ball_collisions.append(collision)

                # Сортируем столкновения по расстоянию от начальной точки
                if ball_collisions:
                    # Вычисляем расстояния от начальной точки до точек столкновения
                    for collision in ball_collisions:
                        collision_point = collision['point']
                        if hasattr(collision_point, 'tolist'):
                            collision_point = collision_point.tolist()
                        distance = np.linalg.norm(np.array(collision_point) - np.array(prev_pos))
                        collision['distance'] = distance

                    # Сортируем по расстоянию
                    ball_collisions.sort(key=lambda x: x['distance'])

                    # Добавляем точки столкновений в траекторию
                    for collision in ball_collisions:
                        collision_point = collision['point']
                        if hasattr(collision_point, 'tolist'):
                            collision_point = collision_point.tolist()
                        trajectory_points.append(collision_point)

            # Добавляем конечную позицию
            trajectory_points.append(current_pos.tolist() if hasattr(current_pos, 'tolist') else current_pos)

            # Создаем траекторию за кадр
            trajectory_line = {
                'type': 'trajectory_line',
                'id': f'trajectory_frame_{i}_{frame_idx}',
                'points': trajectory_points,
                'color': ball_data[3],
                'linewidth': 2,
                'linestyle': '-',
                'alpha': 0.7,
                'tags': ['trajectory_per_frame']
            }
            frame_data['events']['trajectories'].append(trajectory_line)

    def compute_state_hash(self, frame_idx: int = -1) -> str:
        """Вычисление хеша состояния для верификации детерминированности"""
        if frame_idx == -1:
            # Хеш всей симуляции (включая результат, если он есть)
            if self.result_template['frames']:
                # Хешируем позиции всех объектов в последнем кадре
                last_frame = self.result_template['frames'][-1]
                positions = []
                for obj in last_frame['objects']['circles']:
                    positions.append((obj['position'], obj['velocity']))
                state_data = f"{self.balls}{self.walls}{self.pegs}{self.seed}{positions}"
            else:
                state_data = f"{self.balls}{self.walls}{self.pegs}{self.fps}{self.sim_time}{self.seed}"
        else:
            # Хеш конкретного кадра
            frame = self.result_template['frames'][frame_idx]
            state_data = str([(obj['position'], obj['velocity']) for obj in frame['objects']['circles']])

        return hashlib.md5(state_data.encode()).hexdigest()

    def simulate(self) -> Dict[str, Any]:
        """Запуск симуляции с заполнением данных"""
        print("=" * 50)
        print("ЗАПУСК СИМУЛЯЦИИ С УНИВЕРСАЛЬНЫМ ФОРМАТОМ")
        print("=" * 50)

        # Проверяем что объекты установлены
        if self.balls is None or self.walls is None:
            raise ValueError("Сначала установите объекты с помощью set_objects()")

        start_time = time.time()

        # Очищаем предыдущие кадры
        self.result_template['frames'] = []

        balls_state = [ball.copy() for ball in self.balls]

        # Добавляем начальный кадр
        initial_frame = self._create_frame_data(balls_state, 0)
        self.result_template['frames'].append(initial_frame)
        print(f"Добавлен кадр 0")

        # Основной цикл симуляции
        progress_interval = max(1, self.total_frames // 10)

        for frame in range(1, self.total_frames):
            # Передаем self.pegs в update_physics
            balls_state, collisions, trajectory_points = self.update_physics(balls_state, self.walls, self.pegs, self.dt)

            # Передаем trajectory_points в создание кадра
            frame_data = self._create_frame_data(balls_state, frame, collisions, trajectory_points)
            self.result_template['frames'].append(frame_data)

            if frame % progress_interval == 0:
                progress = frame / self.total_frames * 100
                print(f"Прогресс: {progress:.1f}% ({frame}/{self.total_frames}), столкновений: {len(collisions)}")

        # Заполняем статистику
        elapsed = time.time() - start_time
        self.result_template['stats'] = {
            'compute_time': elapsed,
            'collision_count': self.collision_count,
            'numerical_issues': self.numerical_issues,
            'min_time_step': float(self.min_time_step),
            'max_time_step': float(self.max_time_step),
            'state_hash': self.compute_state_hash(),
            'frames_generated': len(self.result_template['frames'])
        }

        print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
        print(f"Создано кадров: {len(self.result_template['frames'])}")
        print(f"Всего столкновений: {self.collision_count}")
        return self.result_template

    def save_result(self, filename: str = 'simulation_result.json'):
        """Сохраняет результат в JSON файл"""
        # Проверяем что есть кадры для сохранения
        if not self.result_template['frames']:
            print("Предупреждение: нет кадров для сохранения")

        # Убедимся, что расширение .json
        if not filename.endswith('.json'):
            filename += '.json'

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.result_template, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

        print(f"Результат сохранен в {filename}")
        print(f"Кадров сохранено: {len(self.result_template['frames'])}")
        print(f"Размер файла: {os.path.getsize(filename) / 1024:.2f} KB")


    # ФИЗИЧЕСКИЕ МЕТОДЫ (без изменений)
    def reflect_vector(self, v, n):
        """Отражение вектора относительно нормали"""
        n = n / np.linalg.norm(n)
        return v - 2 * np.dot(v, n) * n

    def collide_circle_time(self, p1, v1, r1, p2, v2, r2):
        """Время столкновения двух движущихся кругов"""
        dp = p1 - p2
        dv = v1 - v2
        a = np.dot(dv, dv)
        b = 2 * np.dot(dp, dv)
        c = np.dot(dp, dp) - (r1 + r2) ** 2

        if abs(a) < 1e-12:
            return None

        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        # Если шары уже накладываются (c < 0) и движутся навстречу друг другу (b < 0),
        # мы должны вернуть t=0 для немедленного разрешения.
        if c < 0 and b < 0:
            return 0.0

        t_candidates = [t for t in (t1, t2) if 0 <= t <= 1.0]
        return min(t_candidates) if t_candidates else None

    def collide_line_time_with_normal(self, p, v, r, A, B):
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

        t_A = self.collide_circle_time(p, v, r, A, np.zeros(2), 1e-8)
        t_B = self.collide_circle_time(p, v, r, B, np.zeros(2), 1e-8)

        candidates = []

        if t_line is not None:
            # Если шарик уже зашел за стену (t_line < 0), но движется в её сторону,
            # мы должны обработать это как столкновение в момент t=0, чтобы отразить его.
            if t_line < 0:
                t_line = 0.0
            
            contact_point = p + v * t_line
            AP = contact_point - A
            proj = np.dot(AP, AB_unit)
            if -1e-8 <= proj <= AB_len + 1e-8:
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

    def resolve_circle_collision(self, pos1, vel1, r1, pos2, vel2, r2):
        """Разрешение столкновения двух кругов"""
        collision_vector = pos1 - pos2
        distance = np.linalg.norm(collision_vector)

        if distance < 1e-12:
            # Шары в одной точке - добавляем небольшое смещение
            offset = self.rng.rand(2) * 0.001
            pos1 += offset
            collision_vector = pos1 - pos2
            distance = np.linalg.norm(collision_vector)
            self.numerical_issues += 1

        normal = collision_vector / distance
        rel_vel = vel1 - vel2
        vel_along_normal = np.dot(rel_vel, normal)

        if vel_along_normal > 0:
            return vel1, vel2

        # Упругое столкновение
        total_mass = r1 + r2
        impulse = 2 * vel_along_normal / total_mass
        new_vel1 = vel1 - impulse * normal * r2
        new_vel2 = vel2 + impulse * normal * r1

        return new_vel1, new_vel2

    def update_physics(self, balls, walls, pegs, dt):
        """Обновление физики за временной шаг с использованием пространственной сетки (Broad-phase)"""
        new_balls = [b.copy() for b in balls]
        collision_events = []
        trajectory_points = [[] for _ in range(len(balls))]

        for i in range(len(new_balls)):
            trajectory_points[i].append(new_balls[i][0].copy())

        remaining_time = dt
        iterations = 0
        max_iterations = 100

        # Определяем размер ячейки сетки (максимальный диаметр + запас на движение)
        max_r = 0
        if balls:
            max_r = max(max_r, max(b[2] for b in balls))
        if pegs:
            max_r = max(max_r, max(p[1] for p in pegs))
        
        # Размер ячейки должен быть достаточно большим, чтобы покрыть движение шара за шаг
        # Но не слишком большим, чтобы сохранять эффективность.
        cell_size = max_r * 4 if max_r > 0 else 1.0

        def get_cell(pos):
            return (int(pos[0] / cell_size), int(pos[1] / cell_size))

        while remaining_time > 1e-12 and iterations < max_iterations:
            iterations += 1
            earliest_t = remaining_time
            collision_info = None

            # Построение сетки для текущей итерации
            grid = {}
            # Добавляем шары в сетку
            for i, (pos, vel, r, color) in enumerate(new_balls):
                cell = get_cell(pos)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        neighbor = (cell[0] + dx, cell[1] + dy)
                        if neighbor not in grid: grid[neighbor] = {'balls': [], 'pegs': [], 'walls': []}
                        grid[neighbor]['balls'].append(i)
            
            # Добавляем пеги в сетку (они статичны, но для простоты добавим в ту же сетку)
            for i, (pos, r, color) in enumerate(pegs):
                cell = get_cell(pos)
                if cell not in grid: grid[cell] = {'balls': [], 'pegs': [], 'walls': []}
                grid[cell]['pegs'].append(i)

            # Проверка столкновений шаров со стенами (стены пока проверяем все, так как они длинные)
            for i, (pos, vel, r, color) in enumerate(new_balls):
                for A, B in walls:
                    result = self.collide_line_time_with_normal(pos, vel, r, A, B)
                    if result is not None and result[0] <= earliest_t:
                        earliest_t = result[0]
                        collision_info = ('wall', i, result[1], result[2], A, B)

            # Проверка столкновений шаров с пегами (используем сетку)
            for i, (pos, vel, r, color) in enumerate(new_balls):
                cell = get_cell(pos)
                if cell in grid:
                    for peg_idx in grid[cell]['pegs']:
                        peg_pos, peg_r, _ = pegs[peg_idx]
                        t_col = self.collide_circle_time(pos, vel, r, peg_pos, np.zeros(2), peg_r)
                        if t_col is not None and t_col <= earliest_t:
                            earliest_t = t_col
                            collision_info = ('peg', i, peg_idx)

            # Проверка столкновений между шарами (используем сетку)
            checked_pairs = set()
            for cell_coord, cell_data in grid.items():
                cell_balls = cell_data['balls']
                for idx_in_list, i in enumerate(cell_balls):
                    for j in cell_balls[idx_in_list + 1:]:
                        pair = tuple(sorted((i, j)))
                        if pair in checked_pairs: continue
                        checked_pairs.add(pair)
                        
                        pos1, vel1, r1, color1 = new_balls[i]
                        pos2, vel2, r2, color2 = new_balls[j]
                        t_col = self.collide_circle_time(pos1, vel1, r1, pos2, vel2, r2)
                        if t_col is not None and t_col <= earliest_t:
                            earliest_t = t_col
                            collision_info = ('circle', i, j)

            if collision_info is None:
                for i in range(len(new_balls)):
                    pos, vel, r, color = new_balls[i]
                    new_balls[i][0] = pos + vel * remaining_time
                    trajectory_points[i].append(new_balls[i][0].copy())
                break

            # Двигаем все шары до момента столкновения
            for i in range(len(new_balls)):
                pos, vel, r, color = new_balls[i]
                new_balls[i][0] = pos + vel * earliest_t
                trajectory_points[i].append(new_balls[i][0].copy())

            # Обрабатываем столкновение
            if collision_info[0] == 'wall':
                _, ball_idx, normal, col_type, A, B = collision_info
                collision_point = new_balls[ball_idx][0] - normal * new_balls[ball_idx][2]
                collision_events.append({
                    'type': 'wall',
                    'ball_index': ball_idx,
                    'point': collision_point.copy(),
                    'normal': normal.copy(),
                    'wall_type': col_type,
                    'wall_points': (A.copy(), B.copy())
                })
                new_balls[ball_idx][1] = self.reflect_vector(new_balls[ball_idx][1], normal)
                self.collision_count += 1
            elif collision_info[0] == 'peg':
                _, ball_idx, peg_idx = collision_info
                peg_pos, peg_r, _ = pegs[peg_idx]
                collision_vector = new_balls[ball_idx][0] - peg_pos
                dist = np.linalg.norm(collision_vector)
                normal = collision_vector / dist if dist > 1e-12 else np.array([0, 1])
                
                collision_point = peg_pos + normal * peg_r
                collision_events.append({
                    'type': 'wall', # Пеги отображаем как столкновение со стеной для простоты
                    'ball_index': ball_idx,
                    'point': collision_point.copy(),
                    'normal': normal.copy(),
                    'wall_type': 'peg'
                })
                new_balls[ball_idx][1] = self.reflect_vector(new_balls[ball_idx][1], normal)
                self.collision_count += 1
            else:
                _, i, j = collision_info
                pos1, vel1, r1, color1 = new_balls[i]
                pos2, vel2, r2, color2 = new_balls[j]
                collision_point = (pos1 + pos2) / 2
                collision_events.append({
                    'type': 'circle',
                    'ball_indices': (i, j),
                    'point': collision_point.copy()
                })
                new_vel1, new_vel2 = self.resolve_circle_collision(pos1, vel1, r1, pos2, vel2, r2)
                new_balls[i][1] = new_vel1
                new_balls[j][1] = new_vel2
                self.collision_count += 1

            remaining_time -= earliest_t

        if iterations == max_iterations and remaining_time > 1e-8:
            self.numerical_issues += 1
            print(f"Предупреждение: достигнут предел итераций. Осталось времени: {remaining_time:.6f}")

        return new_balls, collision_events, trajectory_points
