#!/usr/bin/env python3
"""
Анализатор кадров с максимальным количеством столкновений.
Ищет в JSON-файле симуляции кадры с наибольшим числом collision events.
"""

import json
import sys
from typing import List, Tuple
import argparse


def find_frames_with_most_collisions(filename: str, top_n: int = 100) -> List[Tuple[int, float, int]]:
    """
    Анализирует JSON файл и возвращает список кортежей (номер_кадра, время, количество_столкновений)
    отсортированный по убыванию количества столкновений.

    Args:
        filename: путь к JSON файлу с результатами симуляции
        top_n: сколько топовых кадров вернуть (по умолчанию 100)

    Returns:
        список кортежей (frame_number, timestamp, collision_count)
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл {filename} не найден!")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка декодирования JSON: {e}")
        return []

    frames = data.get('frames', [])
    if not frames:
        print("❌ В файле нет кадров!")
        return []

    print(f"📊 Анализ {len(frames)} кадров из {filename}")

    # Собираем статистику по кадрам
    frame_collisions = []

    for frame in frames:
        frame_num = frame.get('frame_number', 0)
        timestamp = frame.get('timestamp', 0.0)

        # Берем количество столкновений из statistics
        stats = frame.get('statistics', {})
        collision_count = stats.get('collisions_this_frame', 0)

        # Для надежности можно также проверить наличие events.collisions
        # но используем statistics как основной источник

        if collision_count > 0:  # Сохраняем только кадры со столкновениями
            frame_collisions.append((frame_num, timestamp, collision_count))

    # Сортируем по убыванию количества столкновений
    frame_collisions.sort(key=lambda x: x[2], reverse=True)

    return frame_collisions[:top_n]


def print_collision_stats(frame_collisions: List[Tuple[int, float, int]], top_n: int):
    """
    Красиво выводит статистику столкновений.

    Args:
        frame_collisions: список кортежей (frame_number, timestamp, collision_count)
        top_n: сколько кадров выводится
    """
    if not frame_collisions:
        print("❌ Нет данных для отображения")
        return

    print("\n" + "=" * 70)
    print(f"🏆 ТОП-{len(frame_collisions)} КАДРОВ ПО КОЛИЧЕСТВУ СТОЛКНОВЕНИЙ")
    print("=" * 70)

    # Находим максимальное количество для форматирования
    max_count = frame_collisions[0][2]
    max_frame = frame_collisions[0][0]
    max_time = frame_collisions[0][1]

    # Заголовок таблицы
    print(f"{'№':<4} {'Кадр':<8} {'Время (с)':<15} {'Столкновений':<12}")
    print("-" * 70)

    for i, (frame_num, timestamp, count) in enumerate(frame_collisions, 1):
        print(f"{i:<2} {frame_num:<8} {timestamp:<15.3f} {count:<12}")

    print("=" * 70)

    # Дополнительная статистика
    total_collisions = sum(count for _, _, count in frame_collisions)
    avg_collisions = total_collisions / len(frame_collisions)

    print(f"\n📈 Дополнительная статистика:")
    print(f"   Всего столкновений в топе: {total_collisions}")
    print(f"   Среднее количество: {avg_collisions:.2f}")
    print(f"   Максимум: {max_count} (кадр {max_frame}, время {max_time:.3f}с)")

    # Группировка по диапазонам
    print(f"\n📊 Распределение:")
    # Определяем диапазоны на основе данных
    if max_count > 50:
        ranges = [(0, 10), (11, 20), (21, 30), (31, 50), (51, 100), (101, float('inf'))]
    elif max_count > 20:
        ranges = [(0, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, float('inf'))]
    else:
        ranges = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 15), (16, float('inf'))]

    for low, high in ranges:
        count_in_range = sum(1 for _, _, cnt in frame_collisions if low <= cnt <= high)
        if count_in_range > 0:
            range_str = f"{low}-{high}" if high != float('inf') else f">{low-1}"
            percentage = (count_in_range / len(frame_collisions)) * 100
            print(f"   {range_str:>6} столкновений: {count_in_range:3} кадров ({percentage:5.1f}%)")

    # Первые 10 кадров с самым большим количеством столкновений
    print(f"\n🎯 ТОП-10 (детально):")
    print("-" * 50)
    for i, (frame_num, timestamp, count) in enumerate(frame_collisions[:10], 1):
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        time_str = f"{minutes}:{seconds:06.3f}".replace('.', ',')
        print(f"{i:2}. Кадр {frame_num:5} | Время {time_str:>10} мин:сек | {count:2} столкновений")


def export_to_csv(frame_collisions: List[Tuple[int, float, int]], filename: str):
    """Экспортирует результаты в CSV файл"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("rank,frame,timestamp_seconds,collisions\n")
            for i, (frame_num, timestamp, count) in enumerate(frame_collisions, 1):
                f.write(f"{i},{frame_num},{timestamp:.3f},{count}\n")
        print(f"\n✅ Результаты экспортированы в {filename}")
    except Exception as e:
        print(f"\n❌ Ошибка при экспорте в CSV: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Анализирует JSON файл симуляции и выводит топ кадров по количеству столкновений"
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="complex_simulation_result.json",
        help="Путь к JSON файлу с результатами симуляции (по умолчанию: complex_simulation_result.json)"
    )
    parser.add_argument(
        "-n", "--top",
        type=int,
        default=100,
        help="Количество топовых кадров для вывода (по умолчанию: 100)"
    )
    parser.add_argument(
        "--csv",
        help="Экспортировать результаты в указанный CSV файл"
    )
    parser.add_argument(
        "--min-collisions",
        type=int,
        default=0,
        help="Минимальное количество столкновений для учета (по умолчанию: 0)"
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Не выводить заголовок и статистику, только номера кадров столбиком"
    )

    args = parser.parse_args()

    if not args.no_header:
        print("🔍 АНАЛИЗАТОР КАДРОВ С МАКСИМАЛЬНЫМИ СТОЛКНОВЕНИЯМИ")
        print("=" * 70)

    # Ищем топ кадров
    top_frames = find_frames_with_most_collisions(args.filename, args.top * 2)  # Запрашиваем с запасом

    if not top_frames:
        return 1

    # Фильтруем по минимальному количеству столкновений
    if args.min_collisions > 0:
        top_frames = [(f, t, c) for f, t, c in top_frames if c >= args.min_collisions]
        top_frames = top_frames[:args.top]

    if args.no_header:
        # Просто выводим номера кадров столбиком
        for frame_num, _, _ in top_frames[:args.top]:
            print(frame_num)
    else:
        # Полный вывод со статистикой
        print_collision_stats(top_frames, args.top)

        # Экспорт в CSV если нужно
        if args.csv:
            export_to_csv(top_frames, args.csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())