"""OptiPlot desktop workbench. Local data stays on this computer."""

from pathlib import Path
import io
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import replace

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from optiplot import analyze_file, recommend
from optiplot.render import render, FIT_CHOICES, FITTABLE
from optiplot.export import export_bundle

BG = "#F1F5F9"
INK = "#17324D"
ACCENT = "#0072B2"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OptiPlot  |  光学科研绘图工作台")
        self.geometry(
            f"{min(1400,self.winfo_screenwidth()-60)}x{min(900,self.winfo_screenheight()-110)}"
        )
        self.minsize(1060, 680)
        self.configure(bg=BG)
        self.profile = None
        self.recs = []
        self.active = None
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.thumbnails = []
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei", 10), background=BG, foreground=INK)
        style.configure("TButton", padding=(10, 7))
        style.configure("Accent.TButton", background=ACCENT, foreground="white", padding=(12, 9))
        style.configure("TNotebook.Tab", padding=(16, 9))
        style.configure("Treeview", rowheight=27, background="white", fieldbackground="white")
        header = tk.Frame(self, bg=INK, padx=22, pady=16)
        header.pack(fill="x")
        tk.Label(header, text="OptiPlot", font=("Segoe UI", 24, "bold"), fg="white", bg=INK).pack(
            side="left"
        )
        tk.Label(
            header,
            text="光学科研绘图工作台   /   DATA → FIGURE",
            fg="#ADC9DB",
            bg=INK,
            font=("Microsoft YaHei", 11),
        ).pack(side="left", padx=22)
        tk.Label(header, text="本地处理 · 可解释推荐 · 可复现导出", fg="#ADC9DB", bg=INK).pack(
            side="right"
        )
        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=16, pady=14)
        self.sidebar = ttk.Frame(main, width=290, padding=8)
        self.sidebar.pack_propagate(False)
        main.add(self.sidebar, weight=0)
        right = ttk.Frame(main)
        main.add(right, weight=1)
        ttk.Label(self.sidebar, text="01  导入与识别", font=("Microsoft YaHei", 13, "bold")).pack(
            anchor="w", pady=(0, 10)
        )
        ttk.Button(
            self.sidebar, text="打开实验数据…", style="Accent.TButton", command=self.open_file
        ).pack(fill="x")
        ttk.Label(
            self.sidebar, text="CSV / TSV / TXT / XLSX / NPY / MAT", font=("Segoe UI", 9)
        ).pack(anchor="w", pady=7)
        self.header_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.sidebar, text="文本首行为列名", variable=self.header_var).pack(
            anchor="w"
        )
        row = ttk.Frame(self.sidebar)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Excel 工作表（名称/序号）").pack(side="left")
        self.sheet = tk.StringVar(value="0")
        ttk.Entry(row, textvariable=self.sheet, width=7).pack(side="right")
        ttk.Label(self.sidebar, text="先试一组模拟数据").pack(anchor="w", pady=(12, 4))
        self.examples = sorted(ROOT.joinpath("examples").glob("*.csv"))
        self.demo = ttk.Combobox(
            self.sidebar, values=[p.stem for p in self.examples], state="readonly"
        )
        self.demo.pack(fill="x")
        self.demo.bind(
            "<<ComboboxSelected>>", lambda _: self.load_file(self.examples[self.demo.current()])
        )
        self.summary = tk.StringVar(value="尚未导入数据")
        ttk.Label(self.sidebar, textvariable=self.summary, wraplength=265, justify="left").pack(
            anchor="w", pady=15
        )
        ttk.Separator(self.sidebar).pack(fill="x", pady=6)
        ttk.Label(self.sidebar, text="02  候选图形", font=("Microsoft YaHei", 13, "bold")).pack(
            anchor="w", pady=10
        )
        self.rec_list = tk.Listbox(
            self.sidebar,
            bg="white",
            fg=INK,
            selectbackground=ACCENT,
            borderwidth=0,
            highlightthickness=0,
            font=("Microsoft YaHei", 11),
            height=7,
            activestyle="none",
        )
        self.rec_list.pack(fill="x")
        self.rec_list.bind("<<ListboxSelect>>", self.on_select)
        self.reason = tk.StringVar(value="推荐等级表示数据满足该图型前提的程度，不是统计置信度。")
        ttk.Label(self.sidebar, textvariable=self.reason, wraplength=265, justify="left").pack(
            anchor="w", pady=12
        )
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        self.gallery_tab = ttk.Frame(self.tabs, padding=12)
        self.editor_tab = ttk.Frame(self.tabs, padding=10)
        self.data_tab = ttk.Frame(self.tabs, padding=10)
        for tab, title in [
            (self.gallery_tab, "候选预览"),
            (self.editor_tab, "绘图与导出"),
            (self.data_tab, "数据检查"),
        ]:
            self.tabs.add(tab, text=title)
        self.gallery_canvas = tk.Canvas(self.gallery_tab, bg=BG, highlightthickness=0)
        gallery_scroll = ttk.Scrollbar(
            self.gallery_tab, orient="vertical", command=self.gallery_canvas.yview
        )
        self.gallery_canvas.configure(yscrollcommand=gallery_scroll.set)
        gallery_scroll.pack(side="right", fill="y")
        self.gallery_canvas.pack(side="left", fill="both", expand=True)
        self.gallery = ttk.Frame(self.gallery_canvas)
        gallery_window = self.gallery_canvas.create_window((0, 0), window=self.gallery, anchor="nw")
        self.gallery.bind(
            "<Configure>",
            lambda _: self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all")),
        )
        self.gallery_canvas.bind(
            "<Configure>",
            lambda event: self.gallery_canvas.itemconfigure(gallery_window, width=event.width),
        )
        ttk.Label(
            self.gallery, text="导入数据，自动比较多种表达方式。", font=("Microsoft YaHei", 18)
        ).pack(pady=100)
        self._build_editor()
        self.status = tk.StringVar(value="就绪  |  未对数据做平滑、归一化或拟合")
        ttk.Label(self, textvariable=self.status, padding=(22, 8)).pack(side="bottom", fill="x")
        main.pack_forget()
        main.pack(fill="both", expand=True, padx=16, pady=14)
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_editor(self):
        controls = ttk.Frame(self.editor_tab)
        controls.pack(fill="x")
        self.x = tk.StringVar()
        self.y = tk.StringVar()
        self.z = tk.StringVar()
        self.axis_boxes = {}
        for label, var in [("X / θ", self.x), ("Y / R", self.y), ("Z / 误差 / 分组", self.z)]:
            ttk.Label(controls, text=label).pack(side="left", padx=(6, 4))
            cb = ttk.Combobox(
                controls,
                textvariable=var,
                width=14,
                state="normal" if var is self.y else "readonly",
            )
            cb.pack(side="left")
            self.axis_boxes[label] = cb
        ttk.Button(controls, text="应用变量", command=self.apply_mapping).pack(side="left", padx=10)
        row = ttk.Frame(self.editor_tab)
        row.pack(fill="x", pady=8)
        self.size = tk.StringVar(value="double")
        self.fit = tk.StringVar(value="none")
        self.angle = tk.StringVar(value="deg")
        self.error = tk.StringVar(value="sd")
        for label, var, values, width in [
            ("尺寸", self.size, ["single", "double", "slide"], 8),
            # straight from the renderer, so the list can never offer a fit the
            # engine would silently ignore
            ("拟合", self.fit, list(FIT_CHOICES), 18),
            ("角度", self.angle, ["deg", "rad"], 5),
            ("重复测量误差", self.error, ["sd", "sem"], 5),
        ]:
            ttk.Label(row, text=label).pack(side="left", padx=(6, 3))
            ttk.Combobox(row, textvariable=var, values=values, state="readonly", width=width).pack(
                side="left"
            )
        self.xlog = tk.BooleanVar(value=False)
        self.ylog = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="X log", variable=self.xlog).pack(side="left", padx=5)
        ttk.Checkbutton(row, text="Y log", variable=self.ylog).pack(side="left")
        ttk.Button(row, text="刷新", command=self.redraw).pack(side="left", padx=8)
        row2 = ttk.Frame(self.editor_tab)
        row2.pack(fill="x", pady=(0, 8))
        self.title_var = tk.StringVar()
        self.xlabel = tk.StringVar()
        self.ylabel = tk.StringVar()
        for label, var, width in [
            ("标题", self.title_var, 25),
            ("X 标签", self.xlabel, 18),
            ("Y 标签", self.ylabel, 18),
        ]:
            ttk.Label(row2, text=label).pack(side="left", padx=(6, 3))
            ttk.Entry(row2, textvariable=var, width=width).pack(side="left")
        ttk.Label(
            self.editor_tab,
            text="single: 88.9 mm · double: 182.9 mm · slide: 16:9；曲线 Y 多列用 ; 分隔。",
            font=("Microsoft YaHei", 9),
        ).pack(anchor="w", padx=6)
        self.plot_frame = ttk.Frame(self.editor_tab)
        footer = ttk.Frame(self.editor_tab)
        footer.pack(side="bottom", fill="x")
        ttk.Button(footer, text="导出 PNG / SVG / PDF", command=self.export_figure).pack(
            side="right", padx=5
        )
        ttk.Button(
            footer, text="导出可复现 ZIP", style="Accent.TButton", command=self.export_zip
        ).pack(side="right", padx=5)
        ttk.Label(footer, text="ZIP：数据 + 脚本 + 参数 + 图片").pack(side="left")
        self.plot_frame.pack(fill="both", expand=True, pady=8)

    def open_file(self):
        p = filedialog.askopenfilename(
            filetypes=[("实验数据", "*.csv *.tsv *.txt *.xlsx *.npy *.mat"), ("所有文件", "*.*")]
        )
        if p:
            self.load_file(p)

    def load_file(self, path):
        try:
            sheet = self.sheet.get()
            sheet = int(sheet) if sheet.isdecimal() else sheet
            p = analyze_file(path, header=self.header_var.get(), sheet_name=sheet)
            recs = recommend(p)
        except Exception as exc:
            messagebox.showerror("无法导入数据", str(exc))
            return
        self.profile = p
        self.recs = recs
        self.rec_list.delete(0, "end")
        for r in recs:
            self.rec_list.insert("end", f"{r.tier_label}   {r.title}")
        self.summary.set(
            f"{Path(path).name}\n{p.n_rows:,} 行 × {p.n_cols} 列 · {len(p.numeric_columns)} 个数值变量\n"
            + "\n".join(p.notes[:3])
        )
        for cb in self.axis_boxes.values():
            cb.configure(values=[""] + p.columns)
        self._populate_data()
        self._populate_gallery()
        if recs:
            self.rec_list.selection_set(0)
            self.select_rec(0, show_tab=False)
        self.tabs.select(self.gallery_tab)
        self.status.set("数据已识别  |  点击预览卡片可编辑与导出")

    def _populate_data(self):
        for w in self.data_tab.winfo_children():
            w.destroy()
        p = self.profile
        ttk.Label(
            self.data_tab,
            text="前 200 行预览；导出包保留全部导入数据。\n" + "\n".join(p.notes),
            wraplength=900,
        ).pack(anchor="w", pady=8)
        frame = ttk.Frame(self.data_tab)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=p.columns, show="headings")
        xs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        ys = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        tree.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        for c in p.columns:
            tree.heading(c, text=c)
            tree.column(c, width=150, stretch=False)
        for row in p.data.head(200).itertuples(index=False, name=None):
            tree.insert("", "end", values=[str(v) for v in row])

    def _populate_gallery(self):
        for w in self.gallery.winfo_children():
            w.destroy()
        self.thumbnails = []
        for i, r in enumerate(self.recs[:4]):
            card = tk.Frame(
                self.gallery,
                bg="white",
                padx=12,
                pady=10,
                highlightbackground="#D9E4ED",
                highlightthickness=1,
            )
            card.grid(row=i // 2, column=i % 2, sticky="nsew", padx=5, pady=5)
            tk.Label(
                card,
                text=f"{r.title}   ·   推荐等级 {r.tier_label}",
                bg="white",
                fg=INK,
                font=("Microsoft YaHei", 12, "bold"),
            ).pack(anchor="w")
            try:
                fig = render(self.profile, r)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=80)
                fig.clear()
                buf.seek(0)
                im = Image.open(buf)
                im.thumbnail((410, 205), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(im)
                self.thumbnails.append(photo)
                tk.Label(card, image=photo, bg="white").pack(fill="both", expand=True)
            except Exception as exc:
                tk.Label(card, text=str(exc), bg="white", fg="#AD4A24", wraplength=380).pack(
                    expand=True, pady=20
                )
            tk.Label(
                card,
                text=r.reason,
                bg="white",
                fg="#5D7487",
                wraplength=390,
                justify="left",
                font=("Microsoft YaHei", 9),
            ).pack(anchor="w", pady=4)
            ttk.Button(card, text="选择并编辑 →", command=lambda idx=i: self.select_rec(idx)).pack(
                anchor="e"
            )
        for i in [0, 1]:
            self.gallery.columnconfigure(i, weight=1)
            self.gallery.rowconfigure(i, weight=1)

    def on_select(self, event=None):
        if self.rec_list.curselection():
            self.select_rec(self.rec_list.curselection()[0])

    def select_rec(self, index, show_tab=True):
        self.active = self.recs[index]
        e = self.active.encodings
        self.rec_list.selection_clear(0, "end")
        self.rec_list.selection_set(index)
        self.reason.set(self.active.reason)
        self.x.set(e.get("x", e.get("theta", e.get("source", ""))))
        y = e.get("y", e.get("r", e.get("value", e.get("target", ""))))
        self.y.set("; ".join(y) if isinstance(y, list) else y)
        self.z.set(e.get("z", e.get("yerr", e.get("error", e.get("group", "")))))
        self.angle.set(e.get("angle_unit", "deg"))
        self.fit.set("none")
        self.error.set("sd")
        self.title_var.set("")
        self.xlabel.set("")
        self.ylabel.set("")
        self.xlog.set(False)
        self.ylog.set(False)
        self.redraw()
        if show_tab:
            self.tabs.select(self.editor_tab)

    def apply_mapping(self):
        if not self.active:
            return
        e = dict(self.active.encodings)
        kind = self.active.id
        if kind == "polar":
            e.update(theta=self.x.get(), r=self.y.get())
        elif kind in ["distribution", "box"]:
            e["value"] = self.y.get()
            e.update({"group": self.z.get()} if kind == "box" else {})
        elif kind == "flow":
            e.update(source=self.x.get(), target=self.y.get())
        elif kind not in ["correlation", "table", "matrix_heatmap"]:
            e.update(
                x=self.x.get(),
                y=(
                    [v.strip() for v in self.y.get().split(";") if v.strip()]
                    if kind == "spectrum_lines"
                    else self.y.get()
                ),
            )
            if kind in ["heatmap", "contour"]:
                e["z"] = self.z.get()
            if kind == "errorbar":
                e.pop("error", None)
                e.pop("yerr", None)
                if self.z.get():
                    e["yerr"] = self.z.get()
        self.active = replace(self.active, encodings=e)
        self.redraw()

    def options(self):
        return dict(
            size=self.size.get(),
            fit=self.fit.get(),
            angle_unit=self.angle.get(),
            error_type=self.error.get(),
            title=self.title_var.get(),
            xlabel=self.xlabel.get(),
            ylabel=self.ylabel.get(),
            xlog=self.xlog.get(),
            ylog=self.ylog.get(),
            dpi=300,
        )

    def redraw(self):
        if not self.profile or not self.active:
            return
        try:
            fig = render(self.profile, self.active, options=self.options())
        except Exception as exc:
            messagebox.showerror("无法绘制当前变量", str(exc))
            return
        if self.figure:
            self.figure.clear()
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame, pack_toolbar=False)
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar.update()
        hint = (
            ""
            if self.fit.get() == "none" or self.active.id in FITTABLE
            else "  |  此图型不接受拟合，当前拟合选择未生效"
        )
        self.status.set(f"当前：{self.active.title}  |  {self.active.encodings}" + hint)

    def export_figure(self):
        if not self.active:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".svg",
            initialfile=self.active.id,
            filetypes=[("SVG 矢量图", "*.svg"), ("PDF 矢量图", "*.pdf"), ("PNG 300 dpi", "*.png")],
        )
        if p:
            try:
                render(self.profile, self.active, p, options=self.options()).clear()
                self.status.set("已导出：" + p)
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))

    def export_zip(self):
        if not self.active:
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".zip",
            initialfile="optiplot_" + self.active.id,
            filetypes=[("可复现数据与脚本", "*.zip")],
        )
        if p:
            try:
                export_bundle(self.profile, self.active, p, self.options())
                self.status.set("已导出可复现包：" + p)
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))

    def close(self):
        if self.figure:
            self.figure.clear()
        self.destroy()


if __name__ == "__main__":
    app = App()
    import sys

    if len(sys.argv) > 1:
        app.after(100, lambda: app.load_file(sys.argv[1]))
    app.mainloop()
