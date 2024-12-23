import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog, ttk
from tkinter.scrolledtext import ScrolledText
import webbrowser
import urllib.parse
from vk_topics_analyzer_tools import run_analysis_with_gui, load_cache


class VKAnalyzerApp(tk.Tk):
    CLUSTERING_HYPERPARAMS = {
        'KMeans': {'n_clusters': int},
        'MeanShift': {'bandwidth': float},
        'DBSCAN': {'eps': float, 'min_samples': int},
        'HDBSCAN': {'min_cluster_size': int}
    }

    def __init__(self):
        super().__init__()

        self.title("VK Topics Analyzer")
        self.geometry("700x800")

        self.vk_api_key = None

        self.create_input_frame()
        self.create_settings_frame()
        self.create_buttons_frame()
        self.create_result_frame()

    def create_input_frame(self):
        input_frame = tk.Frame(self)
        input_frame.pack(pady=10, padx=10, fill="x")

        self.label_domains = tk.Label(input_frame, text="Введите ID групп через запятую:")
        self.label_domains.pack(anchor="w")
        self.entry_domains = tk.Entry(input_frame, width=60)
        self.entry_domains.pack(anchor="w", pady=5)

        self.label_count_posts = tk.Label(input_frame, text="Введите количество постов (кратно 100):")
        self.label_count_posts.pack(anchor="w")
        self.entry_count_posts = tk.Entry(input_frame, width=10)
        self.entry_count_posts.pack(anchor="w", pady=5)

    def create_settings_frame(self):
        settings_frame = tk.LabelFrame(self, text="Настройки кластеризации", padx=10, pady=10)
        settings_frame.pack(pady=10, padx=10, fill="x")

        self.label_clustering_method = tk.Label(settings_frame, text="Выберите метод кластеризации:")
        self.label_clustering_method.pack(anchor="w")
        self.clustering_method = tk.StringVar()
        clustering_methods = list(self.CLUSTERING_HYPERPARAMS.keys())
        self.clustering_method.set(clustering_methods[0])  # Значение по умолчанию
        self.dropdown_clustering_method = ttk.OptionMenu(
            settings_frame, self.clustering_method, self.clustering_method.get(), *clustering_methods, command=self.update_hyperparameter_fields
        )
        self.dropdown_clustering_method.pack(anchor="w", pady=5)

        self.hyperparameters_frame = tk.Frame(settings_frame)
        self.hyperparameters_frame.pack(fill="x")

        self.hyperparameter_entries = {}

        self.create_hyperparameter_fields()

        self.label_num_display_tags = tk.Label(settings_frame, text="Количество тегов для отображения на кластер:")
        self.label_num_display_tags.pack(anchor="w")
        self.entry_num_display_tags = tk.Entry(settings_frame, width=10)
        self.entry_num_display_tags.insert(tk.END, "5")
        self.entry_num_display_tags.pack(anchor="w", pady=5)

        self.label_num_display_posts = tk.Label(settings_frame, text="Количество постов для отображения на кластер:")
        self.label_num_display_posts.pack(anchor="w")
        self.entry_num_display_posts = tk.Entry(settings_frame, width=10)
        self.entry_num_display_posts.insert(tk.END, "1")
        self.entry_num_display_posts.pack(anchor="w", pady=5)

    def create_buttons_frame(self):
        buttons_frame = tk.Frame(self)
        buttons_frame.pack(pady=10, padx=10, fill="x")

        self.btn_get_vk_key = tk.Button(buttons_frame, text="Получить VK API ключ", command=self.get_vk_access_token)
        self.btn_get_vk_key.pack(side="left", padx=5)

        self.btn_run = tk.Button(buttons_frame, text="Запустить анализ", command=self.run_analysis)
        self.btn_run.pack(side="left", padx=5)

        self.btn_load_cache = tk.Button(buttons_frame, text="Загрузить кэш", command=self.load_cache)
        self.btn_load_cache.pack(side="left", padx=5)

        self.btn_save_result = tk.Button(buttons_frame, text="Сохранить результат", command=self.save_result)
        self.btn_save_result.pack(side="left", padx=5)

    def create_result_frame(self):
        result_frame = tk.Frame(self)
        result_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.result_text = ScrolledText(result_frame, wrap=tk.WORD, height=15, width=80)
        self.result_text.pack(fill="both", expand=True)

    def get_vk_access_token(self):
        self.vk_api_key = get_vk_access_token_gui()
        if self.vk_api_key:
            messagebox.showinfo("Успешно", "VK API ключ успешно получен.")

    def load_cache(self):
        domains, count_posts = load_cache()
        if domains is None or count_posts is None:
            messagebox.showinfo("Кэш", "Нет доступных кэшированных данных.")
        else:
            self.entry_domains.delete(0, tk.END)
            self.entry_domains.insert(tk.END, ", ".join(domains))
            self.entry_count_posts.delete(0, tk.END)
            self.entry_count_posts.insert(tk.END, str(count_posts))
            messagebox.showinfo("Кэш", "Данные успешно загружены из кэша.")

    def run_analysis(self):
        domains_input = self.entry_domains.get()
        count_posts_input = self.entry_count_posts.get()
        clustering_method = self.clustering_method.get()
        num_display_tags = self.entry_num_display_tags.get()
        num_display_posts = self.entry_num_display_posts.get()

        if not domains_input.strip():
            messagebox.showerror("Ошибка", "Пожалуйста, введите ID групп.")
            return

        if not count_posts_input.isdigit() or int(count_posts_input) <= 0 or int(count_posts_input) % 100 != 0:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректное количество постов (кратно 100).")
            return

        if not self.vk_api_key:
            messagebox.showerror("Ошибка", "Необходимо получить VK API ключ.")
            return

        clustering_params = {}
        expected_params = self.CLUSTERING_HYPERPARAMS.get(clustering_method, {})
        for param_name, param_type in expected_params.items():
            entry = self.hyperparameter_entries.get(param_name)
            if entry:
                value_str = entry.get().strip()
                if not value_str:
                    messagebox.showerror("Ошибка", f"Пожалуйста, введите значение для '{param_name}'.")
                    return
                try:
                    if param_type == int:
                        value = int(value_str)
                        if value <= 0:
                            raise ValueError
                    elif param_type == float:
                        value = float(value_str)
                        if value <= 0:
                            raise ValueError
                    else:
                        raise ValueError(f"Неизвестный тип для параметра '{param_name}'.")
                    clustering_params[param_name] = value
                except ValueError:
                    messagebox.showerror(
                        "Ошибка",
                        f"Неверное значение для '{param_name}'. Ожидается {param_type.__name__}."
                    )
                    return

        try:
            num_display_tags = int(num_display_tags)
            if num_display_tags <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Количество тегов для отображения должно быть положительным целым числом.")
            return

        try:
            num_display_posts = int(num_display_posts)
            if num_display_posts <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Количество постов для отображения должно быть положительным целым числом.")
            return

        domains = [d.strip() for d in domains_input.split(",") if d.strip()]
        count_posts = int(count_posts_input)

        try:
            result = run_analysis_with_gui(
                domains=domains,
                count_posts=count_posts,
                vk_api_key=self.vk_api_key,
                clustering_method=clustering_method,
                clustering_params=clustering_params,
                num_display_tags=num_display_tags,
                num_display_posts=num_display_posts
            )

            self.result_text.delete(1.0, tk.END)
            result_text = "\n".join(result)
            self.result_text.insert(tk.END, result_text)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

    def save_result(self):
        result_text = self.result_text.get(1.0, tk.END).strip()
        if not result_text:
            messagebox.showinfo("Информация", "Нет данных для сохранения.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(result_text)
                messagebox.showinfo("Сохранение", "Результат успешно сохранен.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")

    def update_hyperparameter_fields(self, *args):
        for widget in self.hyperparameters_frame.winfo_children():
            widget.destroy()

        self.hyperparameter_entries.clear()

        self.create_hyperparameter_fields()

    def create_hyperparameter_fields(self):
        clustering_method = self.clustering_method.get()
        hyperparams = self.CLUSTERING_HYPERPARAMS.get(clustering_method, {})

        for param_name, param_type in hyperparams.items():
            label = tk.Label(self.hyperparameters_frame, text=f"{param_name}:")
            label.pack(anchor="w")
            entry = tk.Entry(self.hyperparameters_frame, width=20)
            default_values = {
                'n_clusters': "5",
                'bandwidth': "",
                'eps': "0.5",
                'min_samples': "5",
                'min_cluster_size': "5"
            }
            entry.insert(tk.END, default_values.get(param_name, ""))
            entry.pack(anchor="w", pady=5)
            self.hyperparameter_entries[param_name] = entry


def get_vk_access_token_gui():
    client_id = '52469526'
    redirect_uri = 'https://google.com'
    scope = 'wall,groups,photos'
    state = 'random_state_string'

    auth_url = (
        f"https://oauth.vk.com/authorize?client_id={client_id}"
        f"&redirect_uri={redirect_uri}&response_type=token"
        f"&scope={scope}&state={state}&v=5.131"
    )

    messagebox.showinfo("Авторизация ВКонтакте", "Сейчас откроется браузер для авторизации ВКонтакте. "
                                                 "После этого скопируйте и вставьте URL с access_token.")
    webbrowser.open(auth_url)

    root = tk.Tk()
    root.withdraw()

    full_url = simpledialog.askstring("Вставьте URL", "Введите полный URL после авторизации:")

    if not full_url:
        messagebox.showerror("Ошибка", "Не введен URL!")
        return None

    parsed_url = urllib.parse.urlparse(full_url)
    fragment_params = urllib.parse.parse_qs(parsed_url.fragment)
    access_token = fragment_params.get('access_token', [None])[0]

    if not access_token:
        messagebox.showerror("Ошибка", "Не удалось извлечь access_token. Проверьте введенную ссылку.")
        return None

    return access_token


if __name__ == "__main__":
    app = VKAnalyzerApp()
    app.mainloop()
