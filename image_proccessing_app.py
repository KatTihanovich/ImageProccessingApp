import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from collections import deque

# ------------------- Константы -------------------
ROTATE_ANGLE = 45           # Угол поворота изображения по умолчанию
PIXEL_SHIFT = 30            # Сдвиг изображения на заданное количество пикселей
THRESHOLD_VALUE = 128       # Значение порога для бинаризации
GAUSSIAN_KERNEL = 5         # Размер ядра для размытия Гаусса
SOBEL_KSIZE = 3             # Размер ядра для оператора Собеля
LAPLACIAN_KSIZE = 3         # Размер ядра для оператора Лапласа

# ------------------- Класс приложения -------------------
class ImageProcessorApp:
    def __init__(self, root):
        """Инициализация основного окна и параметров приложения"""
        self.root = root
        root.title("Image Processing Lab")
        root.configure(bg="#F3E9E0")

        # Цветовая схема интерфейса
        self.colors = {
            "bg": "#F3E9E0",
            "frame": "#E8D7C3",
            "button": "#8D6E63",
            "button_hover": "#6D4C41",
            "text": "#3E2723",
            "entry_bg": "#F7EFE7"
        }

        # Настройка стилей ttk
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background=self.colors["frame"])
        self.style.configure('TLabel', background=self.colors["frame"], foreground=self.colors["text"], font=('Arial', 10))
        self.style.configure('TButton', background=self.colors["button"], foreground="white",
                             font=('Calibri', 10, 'bold'), padding=6, borderwidth=0, relief="flat")
        self.style.map('TButton', background=[('active', self.colors["button_hover"])])
        self.style.configure('TCombobox', font=('Arial', 10), fieldbackground=self.colors["entry_bg"],
                             background=self.colors["entry_bg"], foreground=self.colors["text"])
        self.style.configure('BG.TFrame', background=self.colors["bg"])

        # ------------------- Переменные изображений -------------------
        self.original = None       # Исходное изображение
        self.processed = None      # Обработанное изображение
        self.gray_last = None      # Последнее серое изображение
        self.use_gray_vars = {}    # Переменные для использования grayscale

        # Методы, которые требуют серого изображения
        self.grayscale_methods = {
            "Threshold", "Otsu", "Normalize", "Equalize", "Stretch",
            "Gaussian Blur", "Laplacian Sharp", "Sobel Edges",
            "Hough Lines", "Hough Circles",
            "Local Mean", "Local Std", "Local Contrast"
        }

        # Словарь методов обработки и их функций
        self.processing_methods = {
            "Gray Avg": self.gray_avg,
            "Gray HSV": self.gray_hsv,
            "Threshold": self.threshold,
            "Otsu": self.otsu,
            "Normalize": self.normalize,
            "Equalize": self.equalize,
            "Stretch": self.stretch,
            "Gaussian Blur": self.gaussian_blur,
            "Laplacian Sharp": self.laplacian_sharp,
            "Sobel Edges": self.sobel_edges,
            "Shift Horizontal": self.shift_horizontal,
            "Shift Vertical": self.shift_vertical,
            "Rotate": self.rotate,
            "Hough Lines": self.hough_lines,
            "Hough Circles": self.hough_circles,
            "Local Mean": self.local_mean,
            "Local Std": self.local_std,
            "Local Contrast": self.local_contrast,
            "Flood Fill Intensity": self.flood_fill_intensity,
            "Region Grow Texture": self.region_grow_texture
        }

        # Параметры методов с дефолтными значениями
        self.method_params = {
            "Gray Avg": {},
            "Gray HSV": {},
            "Threshold": {"thresh": THRESHOLD_VALUE},
            "Otsu": {},
            "Normalize": {},
            "Equalize": {},
            "Stretch": {},
            "Gaussian Blur": {"kernel": GAUSSIAN_KERNEL},
            "Laplacian Sharp": {"ksize": LAPLACIAN_KSIZE},
            "Sobel Edges": {"ksize": SOBEL_KSIZE},
            "Shift Horizontal": {"pixels": PIXEL_SHIFT},
            "Shift Vertical": {"pixels": PIXEL_SHIFT},
            "Rotate": {"angle": ROTATE_ANGLE},
            "Hough Lines": {"canny_thresh1": 50, "canny_thresh2": 150, "rho": 1, "theta": np.pi / 180, "threshold": 100},
            "Hough Circles": {"dp": 1.2, "min_dist": 20, "param1": 60, "param2": 40, "min_radius": 250, "max_radius": 300},
            "Local Mean": {"window_size": 5},
            "Local Std": {"window_size": 5},
            "Local Contrast": {"window_size": 5},
            "Flood Fill Intensity": {"seed_xy": "100,100", "lo": 10, "up": 10},
            "Region Grow Texture": {"seed_xy": "100,100", "win": 9, "std_tol": 5.0, "inten_tol": 15}
        }

        # Создание GUI
        self.create_widgets()

    # ------------------- GUI -------------------
    def create_widgets(self):
        """Создание всех виджетов интерфейса"""
        container = ttk.Frame(self.root, style='BG.TFrame')
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(container, bg=self.colors["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, style='BG.TFrame')
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.canvas.bind_all("<MouseWheel>", lambda e: self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        frame = self.scrollable_frame

        # Canvas для исходного, серого и обработанного изображений
        self.canvas_original = tk.Label(frame, text='Original Image', bg='white', relief='groove', borderwidth=2)
        self.canvas_original.grid(row=0, column=0, padx=10, pady=10)
        self.canvas_gray = tk.Label(frame, text='Grayscale Image', bg='white', relief='groove', borderwidth=2)
        self.canvas_gray.grid(row=0, column=1, padx=10, pady=10)
        self.canvas_gray.grid_remove()
        self.canvas_processed = tk.Label(frame, text='Processed Image', bg='white', relief='groove', borderwidth=2)
        self.canvas_processed.grid(row=0, column=2, padx=10, pady=10)

        # Кнопки загрузки и сохранения
        ttk.Button(frame, text="Load Image", command=self.load_image).grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Save Original", command=self.save_original).grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Save Grayscale", command=self.save_gray).grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Save Processed", command=self.save_processed).grid(row=2, column=2, sticky="ew", padx=5, pady=5)

        # Dropdown для методов
        self.methods = list(self.processing_methods.keys())
        self.method_var = tk.StringVar(value=self.methods[0])
        self.dropdown = ttk.Combobox(frame, values=self.methods, textvariable=self.method_var, state="readonly", width=30)
        self.dropdown.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_param_entries())

        # Dropdown для выбора направления отображения Собеля
        self.sobel_display_var = tk.StringVar(value="Combined")
        self.sobel_display_dropdown = ttk.Combobox(frame, values=["X", "Y", "Combined"], textvariable=self.sobel_display_var,
                                                   state="readonly", width=10)
        self.sobel_display_dropdown.grid(row=3, column=2, sticky="ew", padx=5, pady=5)

        # Кнопки "Apply" и "Visualize"
        ttk.Button(frame, text="Apply", command=self.apply_method).grid(row=4, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        ttk.Button(frame, text="Show All Steps", command=self.visualize).grid(row=5, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Фрейм для параметров методов
        self.param_frame = ttk.Frame(frame, padding=10, style='TFrame')
        self.param_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
        self.param_vars = {}
        self.use_gray_vars = {}

        self.create_param_entries()

    # ------------------- Вспомогательные функции -------------------
    def update_param_entries(self):
        """Обновление полей параметров при смене метода"""
        self.create_param_entries()
        self.update_grayscale_canvas()

    def create_param_entries(self):
        """Создает поля для ввода параметров каждого метода"""
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_vars = {}
        self.use_gray_vars = {}
        col_count = 2
        row_positions = [0] * col_count
        col = 0
        for method_name, params in self.method_params.items():
            lbl_method = tk.Label(self.param_frame, text=f"{method_name} parameters:", bg=self.colors["frame"],
                                  fg=self.colors["text"], font=('Arial', 10, 'bold'))
            lbl_method.grid(row=row_positions[col], column=col * 2, columnspan=2, sticky="w", padx=5, pady=(5, 0))
            row_positions[col] += 1

            # Флажок использования grayscale
            if method_name in self.grayscale_methods:
                use_gray_var = tk.BooleanVar(value=True)
                chk = tk.Checkbutton(self.param_frame, text="Use Grayscale", variable=use_gray_var,
                                     bg=self.colors["frame"], fg=self.colors["text"], selectcolor=self.colors["entry_bg"])
                chk.grid(row=row_positions[col], column=col * 2, columnspan=2, sticky="w", padx=20)
                self.use_gray_vars[method_name] = use_gray_var
                row_positions[col] += 1

            for pname, pdefault in params.items():
                tk.Label(self.param_frame, text=f"{pname}:", bg=self.colors["frame"], fg=self.colors["text"]).grid(
                    row=row_positions[col], column=col * 2, sticky="w", padx=20)
                var = tk.StringVar(value=str(pdefault))
                entry = tk.Entry(self.param_frame, textvariable=var, width=13)
                entry.grid(row=row_positions[col], column=col * 2 + 1, sticky="w", padx=5)
                self.param_vars[(method_name, pname)] = var
                row_positions[col] += 1
            col = (col + 1) % col_count

    def update_grayscale_canvas(self):
        """Показывает или скрывает canvas для grayscale в зависимости от выбранного метода"""
        method_name = self.method_var.get()
        if method_name in self.grayscale_methods or method_name in {"Gray Avg", "Gray HSV"}:
            self.canvas_gray.grid()
        else:
            self.canvas_gray.grid_remove()

    def get_gray_or_original(self, method_name):
        """Возвращает серое изображение, если выбран флаг использования grayscale"""
        use_gray = self.use_gray_vars.get(method_name, tk.BooleanVar(value=False)).get()
        if use_gray:
            gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
            self.gray_last = gray
            self.show_image(gray, self.canvas_gray)
            return gray
        else:
            self.canvas_gray.config(image='')
            self.gray_last = None
            return self.original

    def safe_cast(self, val, pname=None):
        """Преобразует строку в int, float или tuple для seed_xy"""
        if pname == 'seed_xy':
            if isinstance(val, tuple):
                return val
            try:
                if ',' in val:
                    vals = [int(v.strip()) for v in val.split(',')]
                    return tuple(vals) if len(vals) == 2 else (0, 0)
                elif val.startswith('[') and val.endswith(']'):
                    vals = [int(v.strip()) for v in val[1:-1].split(',')]
                    return tuple(vals) if len(vals) == 2 else (0, 0)
                else:
                    return tuple(map(int, eval(val)))
            except Exception:
                return (0, 0)
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    # ------------------- Загрузка и отображение -------------------
    def load_image(self):
        """Загрузка изображения с диска"""
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.processed = None
                self.gray_last = None
                self.show_image(self.original, self.canvas_original)
                self.canvas_processed.config(image='')
                self.canvas_gray.config(image='')
                self.update_grayscale_canvas()

    def show_image(self, img, canvas):
        """Отображает изображение на canvas с масштабированием до 300x300"""
        if img is not None:
            max_size = 300
            h, w = img.shape[:2]
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 else img)
            im = im.resize((new_w, new_h))
            background = Image.new("RGB", (max_size, max_size), (255, 255, 255))
            offset = ((max_size - new_w) // 2, (max_size - new_h) // 2)
            background.paste(im, offset)
            imgtk = ImageTk.PhotoImage(background)
            canvas.imgtk = imgtk
            canvas.config(image=imgtk)

    def save_image(self, img, default_name):
        """Сохраняет изображение на компьютер"""
        if img is None:
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
                                                 initialfile=default_name)
        if file_path:
            if len(img.shape) == 2:
                cv2.imwrite(file_path, img)
            else:
                cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def save_original(self):
        self.save_image(self.original, "original.png")

    def save_gray(self):
        self.save_image(self.gray_last, "grayscale.png")

    def save_processed(self):
        self.save_image(self.processed, "processed.png")

    # ------------------- Применение методов -------------------
    def apply_method(self):
        """Применяет выбранный метод обработки к изображению"""
        if self.original is None:
            return
        method_name = self.method_var.get()
        params = {pname: self.safe_cast(var.get(), pname)
                  for (mname, pname), var in self.param_vars.items() if mname == method_name}
        self.update_grayscale_canvas()
        self.processed = self.processing_methods[method_name](**params)
        self.show_image(self.processed, self.canvas_processed)

    # ------------------- Методы обработки -------------------
    def gray_avg(self):
        """Преобразование в серое изображение по среднему"""
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.gray_last = gray
        self.show_image(gray, self.canvas_gray)
        return gray

    def gray_hsv(self):
        """Преобразование в серое изображение через HSV"""
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2HSV)[:, :, 2]
        self.gray_last = gray
        self.show_image(gray, self.canvas_gray)
        return gray

    def threshold(self, thresh=THRESHOLD_VALUE):
        """Бинаризация по порогу"""
        img = self.get_gray_or_original("Threshold")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, result = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        return result

    def otsu(self):
        """Бинаризация методом Отсу"""
        img = self.get_gray_or_original("Otsu")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, result = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    def normalize(self):
        """Нормализация значений пикселей"""
        img = self.get_gray_or_original("Normalize")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

    def equalize(self):
        """Гистограмма выравнивания"""
        img = self.get_gray_or_original("Equalize")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.equalizeHist(img)

    def stretch(self):
        """Растягивание контраста"""
        img = self.get_gray_or_original("Stretch")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        min_val = np.min(img)
        max_val = np.max(img)
        return ((img - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)

    def gaussian_blur(self, kernel=GAUSSIAN_KERNEL):
        """Размытие Гаусса"""
        img = self.get_gray_or_original("Gaussian Blur")
        k = kernel if kernel % 2 == 1 else kernel + 1
        return cv2.GaussianBlur(img, (k, k), 0)

    def laplacian_sharp(self, ksize=LAPLACIAN_KSIZE):
        """Резкость через Лаплас"""
        img = self.get_gray_or_original("Laplacian Sharp")
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=ksize)
        return cv2.convertScaleAbs(img - lap)

    def sobel_edges(self, ksize=SOBEL_KSIZE):
        """Выделение границ оператором Собеля"""
        img = self.get_gray_or_original("Sobel Edges")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        choice = self.sobel_display_var.get()
        if choice == "X":
            return abs_grad_x
        elif choice == "Y":
            return abs_grad_y
        else:
            return combined

    def shift_horizontal(self, pixels=PIXEL_SHIFT):
        """Горизонтальный сдвиг изображения"""
        return np.roll(self.original, pixels, axis=1)

    def shift_vertical(self, pixels=PIXEL_SHIFT):
        """Вертикальный сдвиг изображения"""
        return np.roll(self.original, pixels, axis=0)

    def rotate(self, angle=ROTATE_ANGLE):
        """Поворот изображения на заданный угол"""
        h, w = self.original.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(self.original, M, (w, h))

    # ------------------- Методы Hough, FloodFill и RegionGrow -------------------
    def hough_lines(self, canny_thresh1=100, canny_thresh2=200,
                    rho=1, theta=np.pi / 180, threshold=120,
                    min_line_len=50, max_line_gap=10):
        """Выделение прямых методом Hough"""
        img = self.get_gray_or_original("Hough Lines")
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()
        edges = cv2.Canny(img, canny_thresh1, canny_thresh2)
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
        overlay = img_bgr.copy()
        if lines is not None:
            for l in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, l)
                cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return overlay

    def hough_circles(self, dp=1.2, min_dist=40,
                      param1=120, param2=30,
                      min_radius=10, max_radius=0):
        """Выделение окружностей методом Hough"""
        img = self.get_gray_or_original("Hough Circles")
        if len(img.shape) == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img.copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp, min_dist,
                                   param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
        overlay = img_bgr.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles[0]))
            for x, y, r in circles:
                cv2.circle(overlay, (x, y), r, (0, 0, 255), 2)
                cv2.circle(overlay, (x, y), 2, (0, 255, 255), 3)
        return overlay

    def local_mean(self, window_size=5):
        """Локальное среднее в окне"""
        img = self.get_gray_or_original("Local Mean")
        mean = cv2.blur(img, (window_size, window_size))
        return mean

    def local_std(self, window_size=5):
        """Локальная дисперсия/стандартное отклонение"""
        img = self.get_gray_or_original("Local Std").astype(np.float32)
        mean = cv2.blur(img, (window_size, window_size))
        sqr_mean = cv2.blur(img ** 2, (window_size, window_size))
        diff = sqr_mean - mean ** 2
        diff = np.where(diff < 0, 0, diff)
        std = np.sqrt(diff)
        return cv2.convertScaleAbs(std)

    def local_contrast(self, window_size=5):
        """Локальный контраст (макс - мин в окне)"""
        img = self.get_gray_or_original("Local Contrast")
        kernel = (window_size, window_size)
        local_max = cv2.dilate(img, np.ones(kernel, np.uint8))
        local_min = cv2.erode(img, np.ones(kernel, np.uint8))
        contrast = local_max - local_min
        return contrast

    def region_grow_texture(self, seed_xy="0,0", win=9, std_tol=5.0, inten_tol=15):
        """Рост региона по текстуре"""
        img = self.get_gray_or_original("Region Grow Texture")
        if len(img.shape) == 2:
            image_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = img.copy()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if isinstance(seed_xy, str):
            seed_xy = self.safe_cast(seed_xy, pname='seed_xy')

        def local_stats(gray, ksize=win):
            """Вычисление локального std"""
            mean = cv2.blur(gray.astype(np.float32), (ksize, ksize))
            sqr_mean = cv2.blur((gray ** 2).astype(np.float32), (ksize, ksize))
            diff = sqr_mean - mean ** 2
            diff = np.where(diff < 0, 0, diff)
            std = np.sqrt(diff)
            return {"std": std}

        stats = local_stats(gray, ksize=win)
        std_map = stats['std']
        h, w = gray.shape
        sx, sy = int(seed_xy[0]), int(seed_xy[1])
        sx = np.clip(sx, 0, w - 1)
        sy = np.clip(sy, 0, h - 1)
        seed_std = float(std_map[sy, sx])
        seed_int = int(gray[sy, sx])
        visited = np.zeros((h, w), np.uint8)
        mask = np.zeros((h, w), np.uint8)
        q = deque()
        q.append((sx, sy))
        visited[sy, sx] = 1
        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        while q:
            x, y = q.popleft()
            mask[y, x] = 255
            for dx, dy in neigh:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    visited[ny, nx] = 1
                    if abs(float(std_map[ny, nx]) - seed_std) <= std_tol and abs(
                            int(gray[ny, nx]) - seed_int) <= inten_tol:
                        q.append((nx, ny))
        overlay = image_bgr.copy()
        overlay[mask == 255] = (0, 255, 0)
        return overlay

    def flood_fill_intensity(self, seed_xy="0,0", lo=10, up=10):
        """Flood fill по интенсивности"""
        img = self.get_gray_or_original("Flood Fill Intensity")
        if len(img.shape) == 2:
            image_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = img.copy()
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if isinstance(seed_xy, str):
            seed_xy = self.safe_cast(seed_xy, pname='seed_xy')
        mask = np.zeros((h + 2, w + 2), np.uint8)
        img2 = gray.copy()
        cv2.floodFill(img2, mask, seedPoint=tuple(seed_xy), newVal=255, loDiff=lo, upDiff=up, flags=4)
        ff_mask = (mask[1:-1, 1:-1] * 255).astype(np.uint8)
        overlay = image_bgr.copy()
        overlay[ff_mask > 0] = (0, 0, 255)
        return overlay

    # ------------------- Визуализация всех шагов -------------------
    def visualize(self):
        """Визуализирует все этапы обработки изображения"""
        if self.original is None:
            return
        imgs = [self.original]
        titles = ["Original"]
        for name, func in self.processing_methods.items():
            params = {}
            for pname in self.method_params.get(name, {}):
                var = self.param_vars.get((name, pname))
                if var:
                    params[pname] = self.safe_cast(var.get(), pname)
            imgs.append(func(**params))
            titles.append(name)

        win = tk.Toplevel(self.root)
        win.title("All Processing Steps")
        max_columns = 5
        thumb_size = (120, 120)

        def show_full_image(img, title):
            top = tk.Toplevel(win)
            top.title(title)
            if len(img.shape) == 2:
                image = Image.fromarray(img).convert('L')
            else:
                image = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image.resize((400, 400)))
            label = tk.Label(top, image=imgtk)
            label.image = imgtk
            label.pack()

        for i, (img, title) in enumerate(zip(imgs, titles)):
            if img is not None:
                if len(img.shape) == 2:
                    thumb = Image.fromarray(img).convert('L').resize(thumb_size)
                else:
                    thumb = Image.fromarray(img).resize(thumb_size)
                imgtk = ImageTk.PhotoImage(thumb)
                btn = tk.Button(win, image=imgtk, command=lambda i=i: show_full_image(imgs[i], titles[i]))
                btn.image = imgtk
                btn.grid(row=i // max_columns * 2, column=i % max_columns, padx=5, pady=5)
                lbl = tk.Label(win, text=title, bg=self.colors["frame"], fg=self.colors["text"])
                lbl.grid(row=i // max_columns * 2 + 1, column=i % max_columns)

# ------------------- Запуск приложения -------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
