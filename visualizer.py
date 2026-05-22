import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider, CheckButtons
import matplotlib
import time
from typing import Dict, List, Any
import traceback


class UniversalRenderer:
    """Универсальный рендерер объектов на основе схемы"""

    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
        self.custom_renderers = self._compile_custom_renderers()

    def _compile_custom_renderers(self) -> Dict[str, callable]:
        """Компилирует кастомные рендереры из схемы"""
        custom_renderers = {}

        for name, renderer_def in self.schema.get('custom_renderers', {}).items():
            # Извлекаем код функции из описания
            code = renderer_def['implementation'].strip()

            print(f"🔧 Компиляция кастомного рендерера '{name}'...")
            print(f"Код функции:\n{code}\n")

            # Создаем локальное пространство для выполнения
            local_namespace = {}
            global_namespace = {
                'np': np,
                'plt': plt,
                'matplotlib': matplotlib,
                'Circle': Circle,
                'Polygon': Polygon,
                'Line2D': Line2D
            }

            try:
                # Выполняем код функции
                exec(code, global_namespace, local_namespace)
                print(f"✅ Успешно скомпилирован кастомный рендерер '{name}'")

                # Ищем функцию в созданном пространстве имен
                function_found = False
                for obj_name, obj in local_namespace.items():
                    if callable(obj) and not obj_name.startswith('_'):
                        custom_renderers[name] = obj
                        function_found = True
                        print(f"✅ Найдена функция: {obj_name}")
                        break

                if not function_found:
                    print(f"⚠️ В коде не найдена callable функция для рендерера '{name}'")
                    # Создаем заглушку
                    custom_renderers[name] = lambda *args, **kwargs: None


            except Exception as e:
                print(f"❌ Ошибка компиляции кастомного рендерера '{name}': {e}")
                print(f"Трассировка ошибки:")
                traceback.print_exc()
                # Используем fallback вместо заглушки
                if name == 'custom_arrowhead':
                    custom_renderers[name] = self._create_fallback_arrowhead()
                    print(f"✅ Использован fallback для arrowhead")
                else:
                    custom_renderers[name] = lambda *args, **kwargs: None

        print(f"📋 Итог компиляции: {len(custom_renderers)} кастомных рендереров")
        return custom_renderers

    def _resolve_parameters(self, template: Any, obj_data: Dict[str, Any]) -> Any:
        """Заменяет параметры вида @parameter на реальные значения"""
        if isinstance(template, str):
            if template.startswith('@'):
                # Извлекаем путь к параметру
                param_path = template[1:]
                # Проверяем, не является ли это выражением
                if any(op in param_path for op in ['+', '-', '*', '/', '(', ')']):
                    # Это выражение, нужно вычислить
                    return self._evaluate_expression(param_path, obj_data)
                else:
                    # Это простой путь
                    value = self._get_nested_value(obj_data, param_path)
                    return value if value is not None else template
            else:
                # Проверяем, не содержит ли строка выражения с @
                if any(op in template for op in ['+', '-', '*', '/']) and any('@' in part for part in template.split()):
                    # Это строковое выражение с параметрами
                    return self._evaluate_expression(template, obj_data)
                return template

        elif isinstance(template, list):
            return [self._resolve_parameters(item, obj_data) for item in template]

        elif isinstance(template, dict):
            return {key: self._resolve_parameters(value, obj_data) for key, value in template.items()}

        else:
            return template

    def _evaluate_expression(self, expression: str, obj_data: Dict[str, Any]) -> Any:
        """Вычисляет выражение с параметрами"""
        try:
            # Создаем локальное пространство имен с значениями из obj_data
            local_vars = {}
            for key, value in obj_data.items():
                if isinstance(value, (int, float, list, np.ndarray)):
                    # Добавляем переменные в локальное пространство имен
                    local_vars[key] = value

            # Заменяем @parameters на обращения к переменным
            evaluated_expression = expression
            for key in obj_data.keys():
                placeholder = f"@{key}"
                if placeholder in evaluated_expression:
                    evaluated_expression = evaluated_expression.replace(placeholder, key)

            # Вычисляем выражение в созданном пространстве имен
            return eval(evaluated_expression, {}, local_vars)
        except Exception as e:
            print(f"❌ Ошибка вычисления выражения '{expression}': {e}")
            # Возвращаем значения по умолчанию в зависимости от контекста
            if 'position' in expression or 'start_pos' in expression or 'end_pos' in expression:
                return [0, 0]  # Для позиций возвращаем нулевой вектор
            else:
                return 0  # Для других значений возвращаем 0

    def _get_nested_value(self, obj: Dict, path: str) -> Any:
        """Получает значение из вложенного словаря по пути"""
        try:
            # Если путь содержит numpy-синтаксис с двоеточием, обрабатываем особо
            if '[:, ' in path or ':' in path:
                return self._handle_numpy_syntax(obj, path)

            keys = path.split('.')
            current = obj
            for key in keys:
                # Обработка индексов массивов [0]
                if '[' in key and key.endswith(']'):
                    base_key = key.split('[')[0]
                    index_part = key.split('[')[1].split(']')[0]
                    # Проверяем, является ли индекс числом
                    if index_part.isdigit():
                        index = int(index_part)
                        current = current[base_key][index]
                    else:
                        # Если не число, возвращаем весь массив
                        current = current[base_key]
                else:
                    current = current[key]
            return current
        except (KeyError, IndexError, TypeError) as e:
            print(f"❌ Ошибка получения значения по пути '{path}': {e}")
            return None

    def _handle_numpy_syntax(self, obj: Dict, path: str) -> Any:
        """Обрабатывает numpy-подобный синтаксис для работы с массивами"""
        try:
            # Пример: 'points[:, 0]' -> получить все X координаты
            if path.endswith('[:, 0]'):
                base_path = path[:-6]  # Убираем '[:, 0]'
                points = self._get_nested_value(obj, base_path)
                if points and isinstance(points, list):
                    return [point[0] for point in points]  # Все X координаты

            elif path.endswith('[:, 1]'):
                base_path = path[:-6]  # Убираем '[:, 1]'
                points = self._get_nested_value(obj, base_path)
                if points and isinstance(points, list):
                    return [point[1] for point in points]  # Все Y координаты

            # Если синтаксис не распознан, возвращаем исходный массив
            base_path = path.split('[')[0] if '[' in path else path
            return self._get_nested_value(obj, base_path)

        except Exception as e:
            print(f"❌ Ошибка обработки numpy-синтаксиса '{path}': {e}")
            return None

    def _get_matplotlib_method(self, method_path: str):
        """Получает метод matplotlib по пути"""
        parts = method_path.split('.')

        # Если путь начинается с matplotlib, убираем первый элемент
        if parts[0] == 'matplotlib':
            parts = parts[1:]

        current = matplotlib
        for part in parts:
            current = getattr(current, part)
        return current

    def render_object(self, obj_data: Dict[str, Any], active_modes: List[str], ax: plt.Axes) -> List[Any]:
        obj_type = obj_data['type']

        # Проверяем, нужно ли отображать этот объект
        if not self._should_render(obj_data, active_modes):
            return []

        type_schema = self.schema['object_types'].get(obj_type)
        if not type_schema:
            print(f"Предупреждение: неизвестный тип объекта '{obj_type}'")
            return []

        render_method = type_schema['render_method']
        instructions = type_schema['drawing_instructions']

        if render_method.startswith('primitive_'):
            return [self._render_primitive(obj_data, instructions, ax)]
        elif render_method.startswith('composite_'):
            return self._render_composite(obj_data, instructions, ax)
        else:
            print(f"Предупреждение: неизвестный метод рендеринга '{render_method}'")
            return []

    def _should_render(self, obj_data: Dict[str, Any], active_modes: List[str]) -> bool:
        """Проверяет, нужно ли отображать объект based on tags and active modes"""
        tags = obj_data.get('tags', [])

        # Если нет тегов или есть тег 'always' - отображаем всегда
        if not tags or 'always' in tags:
            return True

        # Проверяем, есть ли пересечение тегов объекта с активными режимами
        return any(tag in active_modes for tag in tags)

    def _render_primitive(self, obj_data: Dict[str, Any], instructions: Dict[str, Any], ax: plt.Axes) -> Any:
        """Рендерит простой объект (один вызов библиотеки)"""
        method_name = instructions['library_method']
        args_template = instructions['constructor_args']

        # Разрешаем параметры
        resolved_args = self._resolve_parameters(args_template, obj_data)

        try:
            # Получаем метод matplotlib
            render_func = self._get_matplotlib_method(method_name)

            # Создаем объект
            artist = render_func(**resolved_args)

            # Добавляем на axes
            if hasattr(artist, 'axes') and hasattr(artist, 'figure'):
                # Это уже Artist объект (например, Circle)
                ax.add_artist(artist)
            else:
                # Это может быть список линий (например, от plot)
                if isinstance(artist, list):
                    for art in artist:
                        ax.add_artist(art)
                else:
                    ax.add_artist(artist)

            # Устанавливаем zorder если указан
            if 'zorder' in instructions:
                artist.set_zorder(instructions['zorder'])

            return artist
        except AttributeError as e:
            print(f"❌ Ошибка: метод '{method_name}' не найден в matplotlib: {e}")
            return None
        except Exception as e:
            print(f"❌ Ошибка при создании объекта '{obj_data.get('type', 'unknown')}': {e}")
            return None

    def _render_composite(self, obj_data: Dict[str, Any], instructions: Dict[str, Any], ax: plt.Axes) -> List[Any]:
        artists = []

        for step in instructions['steps']:
            step_type = step['type']
            method_name = step['method']
            args_template = step['args']

            # Разрешаем параметры
            resolved_args = self._resolve_parameters(args_template, obj_data)

            if method_name.startswith('custom_'):
                custom_name = method_name
                if custom_name in self.custom_renderers:
                    try:
                        artist = self.custom_renderers[custom_name](ax, **resolved_args)
                        # КАСТОМНЫЕ РЕНДЕРЕРЫ УЖЕ ДОБАВЛЯЮТ НА AXES!
                        if artist:
                            if isinstance(artist, list):
                                artists.extend(artist)  # НЕ вызываем ax.add_artist
                            else:
                                artists.append(artist)  # НЕ вызываем ax.add_artist
                    except Exception as e:
                        print(f"❌ Ошибка в кастомном рендерере '{custom_name}': {e}")
            else:
                # СТАНДАРТНЫЕ МЕТОДЫ нужно добавлять на axes
                try:
                    render_func = self._get_matplotlib_method(method_name)
                    artist = render_func(**resolved_args)
                    if artist:
                        if isinstance(artist, list):
                            for art in artist:
                                ax.add_artist(art)  # ВЫЗЫВАЕМ ax.add_artist
                                artists.append(art)
                        else:
                            ax.add_artist(artist)  # ВЫЗЫВАЕМ ax.add_artist
                            artists.append(artist)
                except Exception as e:
                    print(f"❌ Ошибка в стандартном методе '{method_name}': {e}")

        return artists


class StaticFrameVisualizer:
    """Универсальный визуализатор статических кадров"""

    def __init__(self, simulation_file: str):
        print("🚀 Инициализация визуализатора...")
        # Загружаем данные симуляции из JSON
        self.simulation_data = self._load_simulation_data(simulation_file)

        self.schema = self.simulation_data['visualization_schema']
        self.config = self.simulation_data['config']
        self.stats = self.simulation_data['stats']
        self.frames = self.simulation_data['frames']
        self.static_objects = self.simulation_data['static_objects']

        print("🔧 Инициализация рендерера...")
        # Инициализируем рендерер
        self.renderer = UniversalRenderer(self.schema)

        # Активные режимы отображения
        self.active_modes = self._get_default_modes()

        # Хранилища для artists
        self.static_artists = []  # Статические объекты (стены и т.д.)
        self.frame_artists = []  # Динамические объекты (кадр)

        # Создаем UI
        self.setup_ui()

        print("✅ Визуализатор инициализирован")

    def _load_simulation_data(self, filename: str) -> Dict[str, Any]:
        """Загружает данные симуляции из JSON файла"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"✅ Успешно загружен JSON файл: {filename}")
            print(f"📋 Версия схемы: {data['metadata']['schema_version']}")
            print(f"🎯 Тип симуляции: {data['metadata']['simulation_type']}")

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"❌ Ошибка декодирования JSON файла: {e}")
        except Exception as e:
            raise ValueError(f"❌ Ошибка загрузки файла {filename}: {e}")

    def _get_default_modes(self) -> List[str]:
        """Получает режимы отображения по умолчанию"""
        default_modes = []
        for mode_id, mode_def in self.schema['display_modes'].items():
            if mode_def.get('default', False):
                default_modes.append(mode_id)
        return default_modes

    def setup_ui(self):
        """Создает пользовательский интерфейс с новой компоновкой"""
        print("🖼️ Создание пользовательского интерфейса...")

        # Создаем фигуру с новой компоновкой - основная сцена слева, панель справа
        self.fig = plt.figure(figsize=(12, 7))  # Шире для боковой панели

        # Основная область визуализации (левая часть)
        self.ax = plt.axes([0.08, 0.15, 0.65, 0.75])  # left, bottom, width, height

        # Область для информационной панели (правая часть)
        self.info_ax = plt.axes([0.75, 0.15, 0.22, 0.75])
        self.info_ax.set_facecolor('#f8f9fa')
        self.info_ax.set_xlim(0, 1)
        self.info_ax.set_ylim(0, 1)
        self.info_ax.axis('off')  # Скрываем оси

        # Настройка основной области отрисовки
        world_bounds = self.config.get('world_bounds', {'xmin': -6, 'xmax': 6, 'ymin': -4, 'ymax': 4})
        self.ax.set_xlim(world_bounds['xmin'], world_bounds['xmax'])
        self.ax.set_ylim(world_bounds['ymin'], world_bounds['ymax'])
        self.ax.set_aspect('equal')
        self.ax.set_title("Универсальная визуализация физической симуляции (JSON)", fontsize=14, pad=10)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_facecolor('#f8f9fa')

        # Информационная панель
        self.setup_info_panel()

        # Элементы управления
        self.setup_controls()

        # Чекбоксы для режимов отображения
        self.setup_display_modes()
        self.draw_static_objects()
        print("✅ Пользовательский интерфейс создан")

    def setup_info_panel(self):
        """Создает информационную панель в правой части"""
        # Основная информация о симуляции
        metadata = self.simulation_data.get('metadata', {})
        info_text = (
            f"Симуляция: {self.config.get('sim_time', 'N/A')}с, {self.config.get('fps', 'N/A')} FPS\n"
            f"Кадров: {len(self.frames)}, Seed: {self.config.get('seed', 'N/A')}\n"
            f"Столкновений: {self.stats.get('collision_count', 0)}\n"
            f"Время расчета: {self.stats.get('compute_time', 0):.2f}с\n"
            f"Типы объектов: {len(self.schema['object_types'])}\n"
            f"Создано: {metadata.get('created_at', 'N/A')}"
        )

        # Добавляем текст на информационную панель
        self.info_ax.text(0.05, 0.95, "ИНФОРМАЦИЯ О СИМУЛЯЦИИ",
                          transform=self.info_ax.transAxes, fontsize=11, fontweight='bold',
                          verticalalignment='top')

        self.info_ax.text(0.05, 0.9, info_text, transform=self.info_ax.transAxes,
                          verticalalignment='top', fontsize=11, fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Область для информации о текущем кадре
        self.info_ax.text(0.05, 0.65, "ТЕКУЩИЙ КАДР",
                          transform=self.info_ax.transAxes, fontsize=11, fontweight='bold',
                          verticalalignment='top')

        self.frame_info_text = self.info_ax.text(0.05, 0.6, '', transform=self.info_ax.transAxes,
                                                 verticalalignment='top', fontsize=9,
                                                 fontfamily='monospace',
                                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def setup_controls(self):
        """Создает элементы управления кадрами с центральным расположением"""
        # Ползунок для навигации по кадрам - центрируем под основной сценой
        ax_slider = plt.axes([0.08, 0.08, 0.65, 0.03])  # Под основной сценой
        self.slider = Slider(ax_slider, 'Кадр', 0, len(self.frames) - 1,
                             valinit=0, valfmt='%d')
        # Настройки для текста слайдера (сдвиг под слайдер и центрирование)
        self.slider.valtext.set_position((1, -0.5))  # Центр по горизонтали, сдвиг вниз
        self.slider.valtext.set_horizontalalignment('right')

        # Кнопки управления - центрируем под слайдером
        button_y = 0.02
        button_width = 0.08
        button_height = 0.04

        # Центральное расположение кнопок под основной сценой
        total_buttons_width = 5 * button_width + 4 * 0.01  # 5 кнопок с отступами
        start_x = 0.08 + (0.65 - total_buttons_width) / 2  # Центрируем в области основной сцены

        buttons = [
            ('|◀◀', self.rewind, start_x),
            ('◀◀', self.prev_frame, start_x + button_width + 0.01),
            ('▶', self.play_pause, start_x + 2 * (button_width + 0.01)),
            ('▶▶', self.next_frame, start_x + 3 * (button_width + 0.01)),
            ('▶▶|', self.fast_forward, start_x + 4 * (button_width + 0.01))
        ]

        self.buttons = {}
        for text, callback, x_pos in buttons:
            ax = plt.axes([x_pos, button_y, button_width, button_height])
            btn = Button(ax, text)
            btn.on_clicked(callback)
            self.buttons[text] = btn

        # Переменные для анимации
        self.playing = False
        self.last_play_time = 0

    def setup_display_modes(self):
        """Создает чекбоксы для режимов отображения с переносом текста"""
        if not self.schema['display_modes']:
            return

        # Подготавливаем данные для CheckButtons
        mode_labels = []
        mode_active = []

        for mode_id, mode_def in self.schema['display_modes'].items():
            # Добавляем переносы длинных названий
            label = mode_def['name']
            # Если название длиннее 17 символов, добавляем перенос
            if len(label) > 17:
                # Пробуем разбить по пробелам
                words = label.split()
                if len(words) > 1:
                    # Находим середину для разбиения
                    mid = len(words) // 2
                    label = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
                else:
                    # Просто разбиваем пополам
                    mid = len(label) // 2
                    label = label[:mid] + '\n' + label[mid:]

            mode_labels.append(label)
            mode_active.append(mode_id in self.active_modes)

        # Создаем чекбоксы в информационной панели с увеличенной областью и сдвигом вниз для избежания перекрытия
        ax_checkboxes = plt.axes([0.75, 0.02, 0.22, 0.40])  # Сдвинут вниз (bottom=0.01) и увеличена высота

        # Настройки для увеличения размера квадратов чекбоксов (уменьшены примерно вдвое)
        num_labels = len(mode_labels)
        frame_props = {
            'sizes': [150] * num_labels,  # Уменьшенный размер рамки квадратов (было 300)
            'linewidths': [1] * num_labels,  # Толщина линии рамки
            'edgecolors': ['black'] * num_labels,
            'facecolors': ['none'] * num_labels
        }
        check_props = {
            'sizes': [100] * num_labels,  # Уменьшенный размер галочки (было 200)
            'linewidths': [1] * num_labels,  # Толщина линии галочки
            'facecolors': ['black'] * num_labels
        }
        label_props = {
            'fontsize': [12] * num_labels  # Уменьшенный шрифт текста (было 14)
        }

        self.checkboxes = CheckButtons(ax_checkboxes, mode_labels, mode_active,
                                       frame_props=frame_props, check_props=check_props, label_props=label_props)

        # Подключаем обработчик
        self.checkboxes.on_clicked(self.on_mode_changed)

        # Заголовок для чекбоксов с увеличенным шрифтом
        ax_checkboxes.set_title('Режимы отображения:', fontsize=14, pad=5)  # Увеличен до 14
        ax_checkboxes.set_facecolor('#f0f0f0')

    def on_mode_changed(self, label):
        """Обработчик изменения режимов отображения"""
        # Обновляем активные режимы
        self.active_modes = []
        for i, (mode_id, mode_def) in enumerate(self.schema['display_modes'].items()):
            if self.checkboxes.get_status()[i]:
                self.active_modes.append(mode_id)

        # Перерисовываем текущий кадр
        self.draw_frame(self.current_frame)

    def clear_frame_artists(self):
        """Очищает только динамические artists текущего кадра"""
        for artist in self.frame_artists:
            try:
                if hasattr(artist, 'remove'):
                    artist.remove()
            except Exception as e:
                print(f"⚠️ Ошибка при удалении artist: {e}")

        self.frame_artists = []

    def draw_static_objects(self):
        """Отрисовывает статические объекты один раз при инициализации"""
        for category, objects in self.static_objects.items():
            for obj in objects:
                artists = self.renderer.render_object(obj, ['always'], self.ax)
                self.static_artists.extend(artists)  # Добавляем в статические
        print(f"✅ Создано {len(self.static_artists)} статических объектов")

    def draw_frame(self, frame_idx: int):
        """Отрисовывает указанный кадр"""
        if frame_idx < 0 or frame_idx >= len(self.frames):
            return

        self.current_frame = frame_idx
        frame_data = self.frames[frame_idx]

        # Очищаем только динамические объекты предыдущего кадра
        self.clear_frame_artists()

        # Статические объекты НЕ перерисовываем - они уже нарисованы

        # Рисуем объекты кадра (динамические)
        for category, objects in frame_data.get('objects', {}).items():
            for obj in objects:
                artists = self.renderer.render_object(obj, self.active_modes, self.ax)
                self.frame_artists.extend(artists)

        # Рисуем события кадра (динамические)
        for category, events in frame_data.get('events', {}).items():
            for event in events:
                artists = self.renderer.render_object(event, self.active_modes, self.ax)
                self.frame_artists.extend(artists)

        # Обновляем информацию о кадре
        self.update_frame_info(frame_data)

        # Обновляем текст слайдера вручную
        self.slider.valtext.set_text(f'{frame_idx}/{len(self.frames) - 1}')

        self.fig.canvas.draw_idle()

    def update_frame_info(self, frame_data: Dict[str, Any]):
        """Обновляет информацию о текущем кадре на правой панели"""
        frame_num = frame_data['frame_number']
        timestamp = frame_data['timestamp']
        stats = frame_data.get('statistics', {})

        # Собираем информацию о шарах
        ball_info = []
        circles = frame_data.get('objects', {}).get('circles', [])
        for i, circle in enumerate(circles):
            speed = circle.get('properties', {}).get('speed', 0)
            ball_info.append(f"Шар {i + 1}: v={speed:.2f}")

        info_text = (f"Кадр: {frame_num}/{len(self.frames) - 1}\n"
                     f"Время: {timestamp:.2f}с\n"
                     f"Столкновений: {stats.get('collisions_this_frame', 0)}\n" +
                     "\n".join(ball_info[:3]))  # Показываем до 3 шаров

        self.frame_info_text.set_text(info_text)

    # Обработчики событий
    def on_slider_change(self, val):
        self.draw_frame(int(val))

    def play_pause(self, event):
        self.playing = not self.playing
        self.buttons['▶'].label.set_text('❚❚' if self.playing else '▶')
        self.last_play_time = time.time()

    def prev_frame(self, event):
        new_frame = max(0, self.current_frame - 1)
        self.slider.set_val(new_frame)

    def next_frame(self, event):
        new_frame = min(len(self.frames) - 1, self.current_frame + 1)
        self.slider.set_val(new_frame)

    def rewind(self, event):
        self.slider.set_val(0)

    def fast_forward(self, event):
        self.slider.set_val(len(self.frames) - 1)

    def animate(self):
        """Главный цикл анимации"""
        frame_interval = 1.0 / self.config.get('fps', 30)  # Значение по умолчанию

        while plt.fignum_exists(self.fig.number):
            current_time = time.time()

            if self.playing and current_time - self.last_play_time >= frame_interval:
                self.current_frame += 1
                if self.current_frame >= len(self.frames):
                    self.current_frame = 0  # Зацикливание
                self.slider.set_val(self.current_frame)
                self.last_play_time = current_time

            plt.pause(0.01)

    def show(self):
        """Запускает визуализацию"""
        print("=" * 60)
        print("🚀 ЗАПУСК УНИВЕРСАЛЬНОЙ ВИЗУАЛИЗАЦИИ ИЗ JSON")
        print("=" * 60)
        print(f"📊 Загружено кадров: {len(self.frames)}")
        print(f"🎨 Режимы отображения: {list(self.schema['display_modes'].keys())}")
        print(f"🔧 Типы объектов: {list(self.schema['object_types'].keys())}")
        print(f"🏗️ Статические объекты: {list(self.static_objects.keys())}")
        print(f"🔐 Хеш состояния: {self.stats.get('state_hash', 'N/A')}")
        print("=" * 60)

        # Подключаем обработчики
        self.slider.on_changed(self.on_slider_change)

        # Рисуем начальный кадр
        self.draw_frame(0)

        # Запускаем анимацию
        plt.show(block=False)
        self.animate()


class JSONSimulationAnalyzer:
    """Анализатор JSON файлов симуляции"""

    @staticmethod
    def analyze_simulation_file(filename: str):
        """Анализирует файл симуляции и выводит статистику"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print("📊 АНАЛИЗ ФАЙЛА СИМУЛЯЦИИ")
            print("=" * 50)

            # Метаданные
            metadata = data.get('metadata', {})
            print(f"Версия схемы: {metadata.get('schema_version', 'N/A')}")
            print(f"Тип симуляции: {metadata.get('simulation_type', 'N/A')}")
            print(f"Создано: {metadata.get('created_at', 'N/A')}")

            # Конфигурация
            config = data.get('config', {})
            print(f"\n⚙️ КОНФИГУРАЦИЯ:")
            print(f"  FPS: {config.get('fps', 'N/A')}")
            print(f"  Время симуляции: {config.get('sim_time', 'N/A')}с")
            print(f"  Всего кадров: {config.get('total_frames', 'N/A')}")
            print(f"  Seed: {config.get('seed', 'N/A')}")

            # Статистика
            stats = data.get('stats', {})
            print(f"\n📈 СТАТИСТИКА:")
            print(f"  Время расчета: {stats.get('compute_time', 0):.2f}с")
            print(f"  Столкновений: {stats.get('collision_count', 0)}")
            print(f"  Числовых проблем: {stats.get('numerical_issues', 0)}")
            print(f"  Хеш состояния: {stats.get('state_hash', 'N/A')[:16]}...")

            # Визуализация
            schema = data.get('visualization_schema', {})
            print(f"\n🎨 ВИЗУАЛИЗАЦИЯ:")
            print(f"  Режимы отображения: {len(schema.get('display_modes', {}))}")
            print(f"  Типы объектов: {len(schema.get('object_types', {}))}")
            print(f"  Кастомные рендереры: {len(schema.get('custom_renderers', {}))}")

            # Кадры
            frames = data.get('frames', [])
            print(f"\n🎞️ КАДРЫ:")
            print(f"  Всего кадров: {len(frames)}")
            if frames:
                first_frame = frames[0]
                print(f"  Объектов в первом кадре: {sum(len(obj) for obj in first_frame.get('objects', {}).values())}")
                print(f"  Событий в первом кадре: {sum(len(evt) for evt in first_frame.get('events', {}).values())}")

            # Статические объекты
            static_objects = data.get('static_objects', {})
            print(f"\n🏗️ СТАТИЧЕСКИЕ ОБЪЕКТЫ:")
            for obj_type, objects in static_objects.items():
                print(f"  {obj_type}: {len(objects)}")

        except Exception as e:
            print(f"Ошибка анализа файла: {e}")


# Функции для быстрой визуализации
def visualize_simulation(simulation_file: str):
    """Быстрая визуализация симуляции из файла"""
    visualizer = StaticFrameVisualizer(simulation_file)
    visualizer.show()
    return visualizer


def analyze_simulation(simulation_file: str):
    """Анализирует файл симуляции"""
    JSONSimulationAnalyzer.analyze_simulation_file(simulation_file)
