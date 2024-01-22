import os
import tkinter as tk
import ttkbootstrap as ttk
from S import generate_ask_signal, generate_fsk_signal, generate_psk_signal
from H import generate_qpsk_signal, generate_qam_signal
from O import test

class user_UI(ttk.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 

        self.title("调制解调系统")
        self.check_and_creat_folder()

        container = ttk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (Binary, Qpsk_button, Qam, Ofdm):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(Binary)

        menu_bar = tk.Menu(self)
        menu_bar.add_command(label="二进制", command=lambda: self.show_frame(Binary))
        menu_bar.add_command(label="QPSK", command=lambda: self.show_frame(Qpsk_button))
        menu_bar.add_command(label="QAM", command=lambda: self.show_frame(Qam))
        menu_bar.add_command(label="OFDM", command=lambda: self.show_frame(Ofdm))
        self.config(menu=menu_bar)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

    def on_close(self):
        self.destroy()

    def check_and_creat_folder(self):
        temp_folder = "temp"
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        else:
            return

class Binary(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.content = None
        self.binary_sequence = None
        
        # 创建PanedWindow
        paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED) 
        paned_window.pack(expand=True, fill="both")

        # 左侧框架
        self.left_frame = tk.Frame(paned_window)
        entry_label = tk.Label(self.left_frame, text="序列输入:")
        entry_label.pack(pady=10)  
        self.entry = tk.Entry(self.left_frame)
        self.entry.pack(pady=5)

        ask_button = ttk.Button(self.left_frame, text="2 ASK", style="link", command=lambda :[self.get_entry_content(),generate_ask_signal(self.binary_sequence),self.display_ask_result()])
        ask_button.pack(pady=20)
        fsk_button = ttk.Button(self.left_frame, text="2 FSK", style="link", command=lambda :[self.get_entry_content(),generate_fsk_signal(self.binary_sequence),self.display_fsk_result()])
        fsk_button.pack(pady=20)
        psk_button = ttk.Button(self.left_frame, text="2 PSK", style="link", command=lambda :[self.get_entry_content(),generate_psk_signal(self.binary_sequence),self.display_psk_result()])
        psk_button.pack(pady=20)
        paned_window.add(self.left_frame, minsize = 200)

        # 右侧框架
        self.right_frame = tk.Frame(paned_window)

        paned_window.add(self.right_frame)

    def get_entry_content(self):
        self.content = self.entry.get()
        self.binary_sequence = [int(bit) for bit in self.content]

    def display_result(self, image_path):
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        
        img = tk.PhotoImage(file=image_path)
        img_label = tk.Label(self.right_frame, image=img)
        img_label.image = img
        img_label.pack()

    def display_ask_result(self):
        self.display_result("temp/ASK.png")

    def display_fsk_result(self):
        self.display_result("temp/FSK.png")

    def display_psk_result(self):
        self.display_result("temp/PSK.png")

class Qpsk_button(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # 创建PanedWindow
        paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        paned_window.pack(expand=True, fill="both")

        # 左侧框架
        self.left_frame = tk.Frame(paned_window)
        qpsk_button = ttk.Button(self.left_frame, text="Q PSK", style="link", command=lambda :[generate_qpsk_signal()])
        qpsk_button.pack(pady=20)
        self.input = ttk.Button(self.left_frame, text="输入信号", style="link", command=lambda :self.display_in_result())
        self.input.pack(pady=15)
        self.constellation = ttk.Button(self.left_frame, text="已调信号", style="link", command=lambda :self.display_re_result())
        self.constellation.pack(pady=15)
        self.output = ttk.Button(self.left_frame, text="接收信号", style="link", command=lambda :self.display_no_result())
        self.output.pack(pady=15)
        self.ber = ttk.Button(self.left_frame, text="解调信号", style="link", command=lambda :self.display_de_result())
        self.ber.pack(pady=15)
        self.powerspectrum = ttk.Button(self.left_frame, text="对比", style="link", command=lambda :self.display_co_result())
        self.powerspectrum.pack(pady=15)

        paned_window.add(self.left_frame, minsize = 200)
        # 右侧框架
        self.right_frame = tk.Frame(paned_window)

        paned_window.add(self.right_frame)

    def display_result(self, image_path):
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        
        img = tk.PhotoImage(file=image_path)
        img_label = tk.Label(self.right_frame, image=img)
        img_label.image = img
        img_label.pack()

    def display_in_result(self):
        self.display_result("temp/QPSK_in.png")
    def display_re_result(self):
        self.display_result("temp/QPSK_re.png")
    def display_no_result(self):
        self.display_result("temp/QPSK_no.png")
    def display_de_result(self):
        self.display_result("temp/QPSK_de.png")
    def display_co_result(self):
        self.display_result("temp/QPSK_co.png")
     
class Qam(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.selected_value = None

        # 创建PanedWindow
        paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        paned_window.pack(expand=True, fill="both")

        # 左侧框架
        self.left_frame = tk.Frame(paned_window)
        choose_label = tk.Label(self.left_frame, text="请选择:")
        choose_label.pack(pady=10)  
        self.choose_box = ttk.Combobox(self.left_frame, width=15, height=4)
        self.choose_box['value'] = ('QAM 16', 'QAM 64', 'QAM 256', 'QAM 1024')
        self.choose_box.pack(pady=10)

        self.input = ttk.Button(self.left_frame, text="发射码元", style="link", command=lambda :self.display_input_result())
        self.input.pack(pady=10)
        self.constellation = ttk.Button(self.left_frame, text="星座图", style="link", command=lambda :self.display_constellation_result())
        self.constellation.pack(pady=10)
        self.output = ttk.Button(self.left_frame, text="接收码元", style="link", command=lambda :self.display_output_result())
        self.output.pack(pady=10)
        self.ber = ttk.Button(self.left_frame, text="误码率", style="link", command=lambda :self.display_ber_result())
        self.ber.pack(pady=10)
        self.powerspectrum = ttk.Button(self.left_frame, text="功率谱", style="link", command=lambda :self.display_powerspectrum_result())
        self.powerspectrum.pack(pady=10)
        self.all = ttk.Button(self.left_frame, text="对比", style="link", command=lambda :self.display_all_result())
        self.all.pack(pady=10)

        paned_window.add(self.left_frame, minsize = 200)
        # 右侧框架
        self.right_frame = tk.Frame(paned_window)

        paned_window.add(self.right_frame)

        self.choose_box.bind("<<ComboboxSelected>>", self.get_value)

    def get_value(self,event):
        self.selected_value = self.choose_box.get().split()[-1]
        generate_qam_signal(int(self.selected_value))
        
    def display_result(self, image_path):
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        
        img = tk.PhotoImage(file=image_path)
        img_label = tk.Label(self.right_frame, image=img)
        img_label.image = img
        img_label.pack()

    def display_input_result(self):
        image_path = f"temp/QAM_{self.selected_value}_input.png"
        self.display_result(image_path)
    def display_constellation_result(self):
        image_path = f"temp/QAM_{self.selected_value}_constellation.png"
        self.display_result(image_path)
    def display_output_result(self):
        image_path = f"temp/QAM_{self.selected_value}_output.png"
        self.display_result(image_path)
    def display_ber_result(self):
        image_path = f"temp/QAM_{self.selected_value}_ber.png"
        self.display_result(image_path)
    def display_powerspectrum_result(self):
        image_path = f"temp/QAM_{self.selected_value}_powerspectrum.png"
        self.display_result(image_path)
    def display_all_result(self):
        image_path = f"temp/QAM_{self.selected_value}_all.png"
        self.display_result(image_path)

class Ofdm(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        # 创建PanedWindow
        paned_window = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        paned_window.pack(expand=True, fill="both")

        # 左侧框架
        self.left_frame = tk.Frame(paned_window)

        self.Briefintroduction = ttk.Button(self.left_frame, text="简介", style="link", command=lambda :[self.display_Briefintroduction(), test()])
        self.Briefintroduction.pack(pady=20)
        self.Principlesummary = ttk.Button(self.left_frame, text="原理概要", style="link", command=lambda :self.display_Principlesummary())
        self.Principlesummary.pack(pady=10)
        self.Codebuilding = ttk.Button(self.left_frame, text="构建流程", style="link", command=lambda :self.display_Codebuilding())
        self.Codebuilding.pack(pady=10)
        self.Testprocess = ttk.Button(self.left_frame, text="测试流程", style="link", command=lambda :self.display_Testprocess())
        self.Testprocess.pack(pady=10)
        self.orimage = ttk.Button(self.left_frame, text="测试图像", style="link", command=lambda :self.display_orimage())
        self.orimage.pack(pady=10)
        self.spectrum = ttk.Button(self.left_frame, text="信号结构", style="link", command=lambda :self.display_spectrum())
        self.spectrum.pack(pady=10)
        self.search = ttk.Button(self.left_frame, text="检索", style="link", command=lambda :self.display_search())
        self.search.pack(pady=10)
        self.deimage = ttk.Button(self.left_frame, text="输出图像", style="link", command=lambda :self.display_deimage())
        self.deimage.pack(pady=10)
        self.coimage = ttk.Button(self.left_frame, text="对比", style="link", command=lambda :self.display_coimage())
        self.coimage.pack(pady=10)

        paned_window.add(self.left_frame, minsize = 200)
        # 右侧框架
        self.right_frame = tk.Frame(paned_window)

        paned_window.add(self.right_frame)

    def display_result(self, image_path):
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        
        img = tk.PhotoImage(file=image_path)
        img_label = tk.Label(self.right_frame, image=img)
        img_label.image = img
        img_label.pack(expand=True, fill="both")

    def display_Briefintroduction(self):
        image_path = f"temp/Briefintroduction.png"
        self.display_result(image_path)

    def display_Principlesummary(self):
        image_path = f"temp/Principlesummary.png"
        self.display_result(image_path)

    def display_Codebuilding(self):
        image_path = f"temp/Codebuilding.png"
        self.display_result(image_path)

    def display_Testprocess(self):
        image_path = f"temp/Testprocess.png"
        self.display_result(image_path)

    def display_orimage(self):
        image_path = f"temp/OFDM_or.png"
        self.display_result(image_path)

    def display_spectrum(self):
        image_path = f"temp/OFDM_sp.png"
        self.display_result(image_path)

    def display_search(self):
        image_path = f"temp/OFDM_se.png"
        self.display_result(image_path)

    def display_deimage(self):
        image_path = f"temp/OFDM_de.png"
        self.display_result(image_path)

    def display_coimage(self):
        image_path = f"temp/OFDM_co.png"
        self.display_result(image_path)


if __name__ == "__main__":
    app = user_UI(themename="cosmo")
    app.geometry("1200x650")
    app.resizable(0,0)
    app.mainloop()