import requests
import webbrowser
import time
import urllib.parse
import re
import threading
import os
import json
import hdbscan
import numpy as np
from typing import List, Dict, Optional, Tuple
from openai import OpenAI
from sklearn.metrics import pairwise_distances
from json import JSONDecodeError

# Настройки моделей
USED_CHAT_GPT_MODEL = "gpt-4o-mini"  # gpt-4o or gpt-4o-mini
USED_EMBEDDINGS_MODEL = "text-embedding-3-large"  # text-embedding-3-small or text-embedding-3-large

# Настройки API
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_API_ORGANIZATION_KEY = os.environ.get("OPENAI_API_ORGANIZATION_KEY")

# Загрузка промпта для ChatGPT
with open("old_prompt.txt", 'r') as file:
    prompt = file.read()

client = OpenAI(
    organization=OPENAI_API_ORGANIZATION_KEY,
    api_key=OPENAI_API_KEY
)

# Переменные кэша
CACHE_FILE = 'VK-topics-analyzer-tools.json'
tags_str_set = set()
tags_embeddings: Dict[str, 'Tag'] = {}
messenger_posts_list: List['MessengerPost'] = []

research_result_text = []


# Классы для хранения тегов и постов
class Tag:
    """
    Класс для представления тега, его эмбеддинга и информации о кластеризации.

    Атрибуты:
    - text (str): Текст тега.
    - embedding (List[float]): Эмбеддинг тега.
    - cluster (int): Идентификатор кластера, к которому относится тег.
    - distance_to_center (float): Расстояние до центра кластера.
    """
    def __init__(self, text: str = '', embedding: Optional[List[float]] = None, cluster: Optional[int] = None, distance_to_center: Optional[float] = None) -> None:
        self.text = text
        self.embedding = embedding
        self.cluster = cluster
        self.distance_to_center = distance_to_center

    def to_dict(self) -> Dict:
        """
        Преобразует объект Tag в словарь для сериализации.

        :return: Словарь, представляющий объект Tag.
        """
        return {
            'text': self.text,
            'embedding': self.embedding,
            'cluster': self.cluster,
            'distance_to_center': self.distance_to_center
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Tag':
        """
        Создает объект Tag из словаря.

        :param data: Словарь с данными для инициализации Tag.
        :return: Объект Tag.
        """
        return cls(
            text=data.get('text', ''),
            embedding=data.get('embedding'),
            cluster=data.get('cluster'),
            distance_to_center=data.get('distance_to_center')
        )


class MessengerPost:
    """
    Класс для представления поста и связанных с ним тегов.

    Атрибуты:
    - text (str): Текст поста.
    - images (List[str]): Список URL изображений в посте.
    - tags (List[Tag]): Список тегов, связанных с постом.
    - scores_for_clusters (Dict[int, float]): Оценки для каждого кластера.
    """
    def __init__(self, text: str = '', images: Optional[List[str]] = None) -> None:
        self.text = text
        self.images = images if images else []
        self.tags: List[Tag] = []
        self.scores_for_clusters: Dict[int, float] = {}

    def to_dict(self) -> Dict:
        """
        Преобразует объект MessengerPost в словарь для сериализации.

        :return: Словарь, представляющий объект MessengerPost.
        """
        return {
            'text': self.text,
            'images': self.images,
            'tags': [tag.to_dict() for tag in self.tags],
            'scores_for_clusters': self.scores_for_clusters
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MessengerPost':
        """
        Создает объект MessengerPost из словаря.

        :param data: Словарь с данными для инициализации MessengerPost.
        :return: Объект MessengerPost.
        """
        post = cls(
            text=data.get('text', ''),
            images=data.get('images', [])
        )

        # Обновление логики десериализации тегов
        post.tags = []
        for tag_data in data.get('tags', []):
            tag_text = tag_data.get('text', '')
            if tag_text in tags_embeddings:
                # Используем уже существующий объект Tag из tags_embeddings
                post.tags.append(tags_embeddings[tag_text])
            else:
                # Создаем новый объект Tag, если его нет в tags_embeddings
                new_tag = Tag.from_dict(tag_data)
                tags_embeddings[tag_text] = new_tag
                post.tags.append(new_tag)

        post.scores_for_clusters = data.get('scores_for_clusters', {})
        return post


# Работа с кэшем
def save_cache(domains: List[str], count_posts: int) -> None:
    """
    Сохраняет кэшированные данные в JSON файл.

    :param domains: Список доменов групп ВКонтакте.
    :param count_posts: Количество постов для обработки.
    """
    print('Кеширование данных начато')

    cache_data = {
        'domains': domains,
        'count_posts': count_posts,
        'tags_str_set': list(tags_str_set),
        'tags_embeddings': {k: v.to_dict() for k, v in tags_embeddings.items()},
        'messenger_posts_list': [post.to_dict() for post in messenger_posts_list]
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_data, f)


def load_cache() -> Tuple[Optional[List[str]], Optional[int]]:
    """
    Загружает кэшированные данные из JSON файла.

    :return: Кэшированные данные (domains, count_posts).
    """
    try:
        with open(CACHE_FILE, 'r') as f:
            if os.stat(CACHE_FILE).st_size == 0:
                return None, None
            cache_data = json.load(f)

        domains = cache_data['domains']
        count_posts = cache_data['count_posts']
        global tags_str_set, tags_embeddings, messenger_posts_list
        tags_str_set = set(cache_data['tags_str_set'])
        tags_embeddings = {k: Tag.from_dict(v) for k, v in cache_data['tags_embeddings'].items()}
        messenger_posts_list = [MessengerPost.from_dict(post) for post in cache_data['messenger_posts_list']]

        return domains, count_posts
    except FileNotFoundError:
        return None, None
    except JSONDecodeError:
        print("Ошибка: не удалось декодировать JSON. Файл может быть поврежден.")
        return None, None


def get_vk_wall_posts(vk_api_key: str, domain: str, page_number: int) -> List[Dict]:
    """
    Получает посты со стены ВКонтакте.

    :param vk_api_key: Ключ доступа к VK API.
    :param domain: Короткий адрес сообщества или пользователя, со стены которого необходимо получить посты.
    :param page_number: Номер страницы (по 100 постов) для сканирования.
    :return: Список постов со стены ВКонтакте.
    """
    params = {
        'access_token': vk_api_key,
        'v': '5.131',
        'domain': domain,
        'count': 100,
        'offset': page_number * 100
    }

    try:
        response = requests.get('https://api.vk.com/method/wall.get', params=params)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при выполнении запроса: {e}, page_number:{page_number}")
        return []

    response_data = response.json()
    if 'error' in response_data:
        print(f"Ошибка: {response_data['error']['error_msg']}")
        return []

    return response_data['response']['items']


def process_vk_wall_posts(posts: List[Dict]) -> List[MessengerPost]:
    """
    Обрабатывает список постов ВКонтакте, извлекая текст и изображения.

    :param posts: Список постов, полученных с помощью функции get_vk_wall_posts.
    :return: Список объектов MessengerPost с текстом и изображениями.
    """
    processed_posts = []

    for post in posts:
        text = post.get('text', '')
        images = []

        if 'attachments' in post:
            for attachment in post['attachments']:
                if attachment['type'] == 'photo':
                    photo_sizes = attachment['photo']['sizes']
                    suitable_photos = [size for size in photo_sizes if size['width'] < 512 and size['height'] < 512]
                    best_photo = max(suitable_photos, key=lambda x: x['width'] * x['height']) if suitable_photos else max(photo_sizes, key=lambda x: x['width'] * x['height'])
                    images.append(best_photo['url'])

        processed_post = MessengerPost(text=text, images=images)
        processed_posts.append(processed_post)

    return processed_posts


# Обработка постов и тегов
def get_tags_list_for_post(messenger_post: MessengerPost) -> List[str]:
    """
    Получает список тегов для поста, используя модель ChatGPT.

    :param messenger_post: Объект MessengerPost.
    :return: Список тегов для поста.
    """
    def parse_tags(response_content: str) -> List[str]:
        match = re.search(r"<start>(.*?)<end>", response_content)
        if not match:
            raise ValueError("Неверный формат ответа от ChatGPT. Ожидается, что теги будут заключены в <start> и <end>.")

        tags = match.group(1).strip().split(", ")
        tags = [tag.strip() for tag in tags if tag.strip()]
        if not tags:
            # raise ValueError("Ответ не содержит тегов.")
            return []
        return tags

    messages = [
        {
            "role": 'system',
            "content": prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": messenger_post.text},
            ]
        }
    ]

    for img_url in messenger_post.images:
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": img_url,
                "detail": "low"
            }
        })

    count = 0
    while True:
        count += 1
        try:
            response = client.chat.completions.create(
                model=USED_CHAT_GPT_MODEL,
                messages=messages,
                temperature=0.0,
                stream=False,
                max_tokens=1024,
            )

            tag_string = response.choices[0].message.content
            return parse_tags(tag_string)
        except Exception as e:
            # if count % 30 == 1:
            if True:
                print(f"Ошибка в get_tags_list_for_post: {e}")
            time.sleep(60)


def get_embedding(text: str) -> List[float]:
    """
    Получает эмбеддинг текста, используя модель эмбеддингов OpenAI.

    :param text: Текст для получения эмбеддинга.
    :return: Эмбеддинг текста в виде одномерного списка.
    """
    count = 0
    while True:
        count += 1
        try:
            response = client.embeddings.create(
                input=text,
                model=USED_EMBEDDINGS_MODEL
            )

            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            # if count % 30 == 1:
            if True:
                print(f"Ошибка в get_embedding: {e}")
            time.sleep(60)


def post_handler(post: MessengerPost) -> None:
    """
    Обрабатывает пост, извлекая теги и эмбеддинги для каждого тега.

    :param post: Объект MessengerPost для обработки.
    """
    tags_str_list = get_tags_list_for_post(post)

    for tag_str in tags_str_list:
        if tag_str not in tags_str_set:
            tag = Tag(text=tag_str, embedding=get_embedding(tag_str))
            tags_str_set.add(tag_str)
            tags_embeddings[tag_str] = tag

            print(f'Тег "{tag_str}" обработан. Его примерный номер: {len(tags_str_set)}')
        else:
            tag = tags_embeddings[tag_str]
            print(f'Тег "{tag_str}" уже был обработан.')

        post.tags.append(tag)
    messenger_posts_list.append(post)


# Многопоточность для обработки постов
def parallel_hundred_posts_handler(vk_api_key: str, domain: str, page_number: int) -> None:
    """
    Обрабатывает 100 постов из указанного домена ВКонтакте параллельно.

    :param vk_api_key: Ключ доступа к VK API.
    :param domain: Домен группы ВКонтакте.
    :param page_number: Номер страницы для обработки.
    """
    posts = process_vk_wall_posts(get_vk_wall_posts(vk_api_key, domain, page_number))
    thr_for_post_list = []

    for post in posts:
        thr_for_post = threading.Thread(target=post_handler, args=(post,))
        thr_for_post_list.append(thr_for_post)
        thr_for_post.start()

    print(f"Создано {len(thr_for_post_list)} потоков.")

    for thr_obj in thr_for_post_list:
        thr_obj.join()


def synchronous_hundred_posts_handler(vk_api_key: str, domain: str, page_number: int) -> None:
    """
    Обрабатывает 100 постов из указанного домена ВКонтакте синхронно.

    :param vk_api_key: Ключ доступа к VK API.
    :param domain: Домен группы ВКонтакте.
    :param page_number: Номер страницы для обработки.
    """
    print('Запуск synchronous_hundred_posts_handler')

    posts = process_vk_wall_posts(get_vk_wall_posts(vk_api_key, domain, page_number))

    for post in posts:
        post_handler(post)


def parallel_groups_handler(vk_api_key: str, domains: List[str], count_posts: int, suppress_excessive_parallelization: bool = False) -> None:
    """
    Параллельно обрабатывает посты из нескольких доменов ВКонтакте.

    :param suppress_excessive_parallelization: Выключает распараллеливание на уровне отдельных постов.
    :param vk_api_key: Ключ доступа к VK API.
    :param domains: Список доменов для обработки.
    :param count_posts: Общее количество постов для обработки в каждом домене.
    """
    print('Запуск parallel_groups_handler')

    thr_list = []

    for domain in domains:
        for i in range(int(count_posts / 100)):
            if suppress_excessive_parallelization:
                thr = threading.Thread(target=synchronous_hundred_posts_handler, args=(vk_api_key, domain, i))
            else:
                thr = threading.Thread(target=parallel_hundred_posts_handler, args=(vk_api_key, domain, i))
            thr_list.append(thr)
            thr.start()

    print(f"Создано {len(thr_list)} потоков.")

    for thr_obj in thr_list:
        thr_obj.join()

    save_cache(domains, count_posts)


# Кластеризация тегов и вычисление оценок для постов
def cluster_tags_with_hdbscan(min_cluster_size: int, num_display_tags=5) -> None:
    """
    Выполняет кластеризацию тегов с использованием алгоритма HDBSCAN и сохраняет результаты в tags_embeddings.
    """
    print("Выполняется кластеризация тегов")

    # Сбор эмбеддингов для кластеризации
    embeddings = [tag.embedding for tag in tags_embeddings.values() if tag.embedding is not None]
    if not embeddings:
        print("Нет доступных эмбеддингов для кластеризации.")
        return

    # Выполнение кластеризации с использованием HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)

    # Присваивание кластера каждому тегу
    for (tag_text, tag), label in zip(tags_embeddings.items(), labels):
        tag.cluster = label

    # Определение уникальных меток кластеров и центров кластеров
    unique_labels = set(labels)
    global cluster_centers
    cluster_centers = {}
    for label in unique_labels:
        if label != -1:
            cluster_points = [tag.embedding for tag in tags_embeddings.values() if tag.cluster == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers[label] = cluster_center

    # Вычисление расстояния каждого тега до центра его кластера
    for tag in tags_embeddings.values():
        if tag.cluster != -1:
            tag.distance_to_center = np.linalg.norm(np.array(tag.embedding) - cluster_centers[tag.cluster])

    # Сбор информации о кластерах
    clusters_info = {}
    for tag in tags_embeddings.values():
        if tag.cluster != -1:
            if tag.cluster not in clusters_info:
                clusters_info[tag.cluster] = []
            clusters_info[tag.cluster].append(tag)

    # Вывод результатов
    print(f"Количество обрабатываемых тегов: {len(tags_str_set)}")
    print("Кластеризация завершена. Результаты добавлены в tags_embeddings.")

    print(f"Найдено кластеров: {len(clusters_info)}")
    for cluster_label, tags in sorted(clusters_info.items()):
        text = f"Кластер {cluster_label}:\nКоличество тегов = {len(tags)}\n"
        top_num_display_tags = sorted(tags, key=lambda t: t.distance_to_center)[:num_display_tags]
        text += "Представительные теги кластера:"
        for tag in top_num_display_tags:
            text += f"  - Тег: {tag.text}, Расстояние до центра кластера: {tag.distance_to_center}\n"

        text += '\n'
        research_result_text.append(text)


    # Подсчет количества тегов, не попавших ни в один кластер
    noise_count = sum(1 for tag in tags_embeddings.values() if tag.cluster == -1)
    print(f"Количество тегов, не попавших ни в один кластер: {noise_count}")


def calculate_scores_for_posts(k1: float = 1.5, k2: float = 3.5) -> None:
    """
    Вычисляет scores_for_clusters для каждого поста на основе кластеров тегов с использованием обновленной формулы.

    :param k1: Коэффициент для регулирования влияния расстояния между кластерами.
    :param k2: Коэффициент для регулирования влияния расстояния от мусорного тега до центра кластера.
    """
    cluster_centers_np = {label: np.array(center) for label, center in cluster_centers.items() if label != -1}
    cluster_labels = list(cluster_centers_np.keys())
    cluster_distances = pairwise_distances(list(cluster_centers_np.values()), metric='euclidean')
    cluster_distance_dict = {
        (cluster_labels[i], cluster_labels[j]): cluster_distances[i, j]
        for i in range(len(cluster_labels))
        for j in range(len(cluster_labels))
    }

    for post in messenger_posts_list:
        for cluster in cluster_labels:
            post_score = 0
            has_cluster_tag = False  # Проверяем, есть ли в посте теги, принадлежащие текущему кластеру

            for tag in post.tags:
                if tag.cluster == cluster:
                    # Если тег принадлежит текущему кластеру, добавляем расстояние до центра кластера
                    post_score += tag.distance_to_center if tag.distance_to_center is not None else 0
                    has_cluster_tag = True  # Найден тег, относящийся к текущему кластеру
                elif tag.cluster != -1:
                    # Если тег принадлежит другому кластеру, добавляем расстояние между кластерами, умноженное на k1
                    post_score += cluster_distance_dict.get((cluster, tag.cluster), 0) * k1
                elif tag.cluster == -1:
                    # Если тег не принадлежит ни одному кластеру (мусорный тег),
                    # добавляем расстояние от центра рассматриваемого кластера до эмбеддинга тега
                    distance_to_cluster_center = np.linalg.norm(np.array(tag.embedding) - cluster_centers[cluster])
                    post_score += distance_to_cluster_center * k2 if tag.embedding is not None else 0

            # Проверяем наличие хотя бы одного тега, относящегося к текущему кластеру
            if has_cluster_tag:
                # Рассчитываем среднее значение для кластера и сохраняем обратное значение, если `post_score` не равно 0
                if post_score > 0:
                    post.scores_for_clusters[cluster] = 1 / (post_score / len(post.tags))
                else:
                    post.scores_for_clusters[cluster] = 0

    print("Оценки для кластеров постов успешно вычислены.")


# Поиск представителей кластеров
def find_representative_posts(num_display_posts=1) -> None:
    """
    Находит представителей с самым высоким score для каждого кластера и выводит эти посты.
    """
    # Сначала собираем посты для каждого кластера
    representative_posts = {}
    for post in messenger_posts_list:
        for cluster, score in post.scores_for_clusters.items():
            if cluster not in representative_posts:
                representative_posts[cluster] = []
            representative_posts[cluster].append((post, score))

    # Затем выводим посты с самым высоким score для каждого кластера в порядке возрастания кластеров
    sorted_representative_posts = sorted(representative_posts.keys())
    for cluster_number in range(len(sorted_representative_posts)):
        posts = sorted(representative_posts[sorted_representative_posts[cluster_number]], key=lambda x: x[1], reverse=True)[:num_display_posts]
        research_result_text[cluster_number] += f"Представительные посты этого кластера:\n"
        for post, score in posts:
            research_result_text[cluster_number] += f"  Пост с score ({score}):\n"
            research_result_text[cluster_number] += f"  Текст поста: \n```\n{post.text}\n```\n"
        research_result_text[cluster_number] += "---\n\n"


# Функции работы с VK API
def get_vk_access_token() -> str:
    """
    Открывает браузер для получения access_token через OAuth авторизацию ВКонтакте.

    :return: access_token, извлеченный из введенной пользователем ссылки.
    """
    client_id = '52469526'
    redirect_uri = 'https://google.com'
    scope = 'wall,groups,photos'
    state = 'random_state_string'

    auth_url = (
        f"https://oauth.vk.com/authorize?client_id={client_id}"
        f"&redirect_uri={redirect_uri}&response_type=token"
        f"&scope={scope}&state={state}&v=5.131"
    )

    print("После авторизации скопируйте всю ссылку и вставьте её сюда.")
    time.sleep(10)
    print("Открываю браузер для авторизации...")
    webbrowser.open(auth_url)

    full_url = input("Введите полный URL после авторизации: ")
    parsed_url = urllib.parse.urlparse(full_url)
    fragment_params = urllib.parse.parse_qs(parsed_url.fragment)
    access_token = fragment_params.get('access_token', [None])[0]

    if not access_token:
        print("Не удалось извлечь access_token. Проверьте введенную ссылку.")
    return access_token


def get_groups_id():
    """
    Получение от пользователя списка id групп в VK для анализа

    :return: список id групп в VK
    """
    print('Введите через запятую идентификаторы групп в VK, которые нужно проанализировать')
    print('Например, для группы по ссылке: https://vk.com/overhearspbsu, id будет: overhearspbsu')

    inp = input("Введите: ")

    # Проверка на пустой ввод
    if not inp.strip():
        print("Ошибка: Введите хотя бы один идентификатор группы.")
        return get_groups_id()

    # Удаление пробелов, точек и других неуместных символов
    ids = [re.sub(r'[^a-zA-Z0-9_]', '', group_id.strip()) for group_id in inp.split(',')]

    # Уведомление, если какие-то идентификаторы были изменены
    cleaned_ids = [group_id for group_id in ids if group_id]
    if len(cleaned_ids) < len(ids):
        print("Некоторые символы были удалены. Проверьте идентификаторы.")

    return cleaned_ids


# Основная программа
def main() -> None:
    """
    Основная функция программы, управляющая процессом анализа постов, кластеризации и сохранения данных.
    """
    print("Предупреждение: в версии кода для консоли есть не устранённая ошибка при кластеризации.\nПредлагаем использовать gui версию, запустив main.py")

    domains, count_posts = load_cache()
    if domains is None or count_posts is None:
        domains = get_groups_id()
        count_posts = 0
        while count_posts <= 0 or count_posts % 100 != 0:
            count_posts = int(input("Введите количество постов, которое нужно получить из каждого паблика (должно быть больше 0 и кратно 100): "))
        vk_api_key = get_vk_access_token()
        parallel_groups_handler(vk_api_key, domains, count_posts, suppress_excessive_parallelization=True)
        save_cache(domains, count_posts)
    else:
        print("Данные успешно загружены из кэша. Пропускаем сбор данных.")

    cluster_tags_with_hdbscan(min_cluster_size=100, num_display_tags=1)
    calculate_scores_for_posts()
    find_representative_posts(num_display_posts=5)

    for output_text in research_result_text:
        print(output_text)


if __name__ == "__main__":
    main()
