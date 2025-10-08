import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk

# Значения по умолчанию
ROTATE_ANGLE = 45
PIXEL_SHIFT = 30
THRESHOLD_VALUE = 128
GAUSSIAN_KERNEL = 5
SOBEL_KSIZE = 3
LAPLACIAN_KSIZE = 3

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        root.title("Image Processing Lab")
        root.configure(bg="#F3E9E0")

        # Цветовая палитра
        self.colors = {
            "bg": "#F3E9E0",
            "frame": "#E8D7C3",
            "button": "#8D6E63",
            "button_hover": "#6D4C41",
            "text": "#3E2723",
            "entry_bg": "#F7EFE7"
        }

        # Настройка стиля ttk
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background=self.colors["frame"])
        self.style.configure('TLabel', background=self.colors["frame"], foreground=self.colors["text"], font=('Arial', 10))
        self.style.configure('TButton',
                             background=self.colors["button"],
                             foreground="white",
                             font=('Calibri', 10, 'bold'),
                             padding=6,
                             borderwidth=0,
                             relief="flat")
        self.style.map('TButton', background=[('active', self.colors["button_hover"])] )
        self.style.configure('TCombobox',
                             font=('Arial', 10),
                             fieldbackground=self.colors["entry_bg"],
                             background=self.colors["entry_bg"],
                             foreground=self.colors["text"])

        self.original = None
        self.processed = None

        # Методы, которым требуется grayscale
        self.grayscale_methods = {
            "Gray Avg", "Gray HSV", "Threshold", "Otsu",
            "Normalize", "Equalize", "Stretch",
            "Gaussian Blur", "Laplacian Sharp", "Sobel Edges"
        }

        # Словарь методов обработки
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
            "Rotate": self.rotate
        }

        # Словарь параметров с дефолтными значениями
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
            "Rotate": {"angle": ROTATE_ANGLE}
        }

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=10, style='TFrame')
        frame.pack(fill=tk.BOTH, expand=True)

        # Канвасы для изображений
        self.canvas_original = tk.Label(frame, text='Original Image', bg='white', relief='groove', borderwidth=2)
        self.canvas_original.grid(row=0, column=0, padx=10, pady=10)
        self.canvas_processed = tk.Label(frame, text='Processed Image', bg='white', relief='groove', borderwidth=2)
        self.canvas_processed.grid(row=0, column=1, padx=10, pady=10)
        self.canvas_gray = tk.Label(frame, text='Grayscale Image', bg='white', relief='groove', borderwidth=2)
        self.canvas_gray.grid(row=0, column=2, padx=10, pady=10)
        self.canvas_gray.grid_remove()

        # Кнопка загрузки изображения
        btn_load = ttk.Button(frame, text="Load Image", command=self.load_image)
        btn_load.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Выпадающий список методов обработки
        self.methods = list(self.processing_methods.keys())
        self.method_var = tk.StringVar(value=self.methods[0])
        self.dropdown = ttk.Combobox(frame, values=self.methods, textvariable=self.method_var, state="readonly", width=30)
        self.dropdown.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", lambda e: self.create_param_entries())

        # Кнопка применения метода
        btn_apply = ttk.Button(frame, text="Apply", command=self.apply_method)
        btn_apply.grid(row=1, column=2, sticky="ew", padx=5, pady=5)

        # Кнопка визуализации всех шагов
        btn_visualize = ttk.Button(frame, text="Show All Steps", command=self.visualize)
        btn_visualize.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)

        # Фрейм для параметров метода
        self.param_frame = ttk.Frame(self.root, padding=10, style='TFrame')
        self.param_frame.pack(fill=tk.BOTH, expand=True)
        self.param_vars = {}
        self.create_param_entries()

    # Создание полей параметров для выбранного метода
    def create_param_entries(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        method_name = self.method_var.get()
        params = self.method_params.get(method_name, {})
        self.param_vars = {}
        for i, (pname, pdefault) in enumerate(params.items()):
            tk.Label(self.param_frame, text=f"{pname}:", bg=self.colors["frame"], fg=self.colors["text"]).grid(row=i, column=0, sticky="w", padx=5)
            var = tk.StringVar(value=str(pdefault))
            entry = tk.Entry(self.param_frame, textvariable=var, width=10)
            entry.grid(row=i, column=1, sticky="w", padx=5)
            self.param_vars[pname] = var

    # Работа с изображениями
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                self.original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.processed = None
                self.show_image(self.original, self.canvas_original)
                self.canvas_processed.config(image='')
                self.canvas_gray.config(image='')
                self.canvas_gray.grid_remove()

    def show_image(self, img, canvas):
        if img is not None:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            im = Image.fromarray(img)
            im = im.resize((300, 300))
            imgtk = ImageTk.PhotoImage(im)
            canvas.imgtk = imgtk
            canvas.config(image=imgtk)

    # Применение метода
    def apply_method(self):
        if self.original is None:
            return
        method_name = self.method_var.get()
        if method_name in self.grayscale_methods:
            self.canvas_gray.grid()
        else:
            self.canvas_gray.grid_remove()

        # Получаем параметры из полей
        params = {k: self.safe_cast(v.get()) for k, v in self.param_vars.items()} if self.param_vars else {}
        self.processed = self.processing_methods[method_name](**params)
        self.show_image(self.processed, self.canvas_processed)

    # Безопасное преобразование строки в число
    def safe_cast(self, val, typ=int):
        try:
            return typ(val)
        except ValueError:
            return val

    # Методы обработки с промежуточным grayscale
    def gray_avg(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        return gray

    def gray_hsv(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2HSV)[:, :, 2]
        self.show_image(gray, self.canvas_gray)
        return gray

    def threshold(self, thresh=THRESHOLD_VALUE):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        _, result = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        return result

    def otsu(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return result

    def normalize(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    def equalize(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        return cv2.equalizeHist(gray)

    def stretch(self):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        return ((gray - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)

    def gaussian_blur(self, kernel=GAUSSIAN_KERNEL):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        k = kernel if kernel % 2 == 1 else kernel + 1  # ядро должно быть нечетным
        return cv2.GaussianBlur(gray, (k, k), 0)

    def laplacian_sharp(self, ksize=LAPLACIAN_KSIZE):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        return cv2.convertScaleAbs(gray - lap)

    def sobel_edges(self, ksize=SOBEL_KSIZE):
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        self.show_image(gray, self.canvas_gray)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # Геометрические преобразования
    def shift_horizontal(self, pixels=PIXEL_SHIFT):
        return np.roll(self.original, pixels, axis=1)

    def shift_vertical(self, pixels=PIXEL_SHIFT):
        return np.roll(self.original, pixels, axis=0)

    def rotate(self, angle=ROTATE_ANGLE):
        h, w = self.original.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(self.original, M, (w, h))

    # Визуализация всех шагов
    def visualize(self):
        if self.original is None:
            return
        imgs = [self.original]
        titles = ["Original"]
        for name, func in self.processing_methods.items():
            # используем дефолтные параметры
            params = self.method_params.get(name, {})
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

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
