
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
from tkinter import filedialog
from utils import Utils
from threading import *
from PIL import Image, ImageTk
import os
import main
root = Tk()
root.title("Object Construction Using GAN")

root.geometry('500x450+400+100')
root.resizable(1, 0)

frame = Frame(width=300, height=300)
frame.pack()


heading = Label(frame, text="Object Construction Using GAN")
heading.grid(row=0, column=0, columnspan=2, padx=65, pady=10)
heading.config(font=("Arial", 14, "bold"))

style = Style()
style.configure('W.TButton', font=('Arial', 10), background='red')
style.configure('A.TButton', font=('Arial', 10), background='green')


def browseFiles():
    global filename
    filename = filedialog.askopenfilename(initialdir="/ObjectConstruction",
                                          title="Select a File",
                                          filetypes=(("Image files",
                                                      "*.png*"),
                                                     ("all files",
                                                      "*.*")))

    # Change label contents
    label_file_explorer.configure(text="File selected: " + os.path.basename(filename))

    return filename


def post():
    global cv
    cv = Canvas()
    path = Utils().image_sim()
    photo = ImageTk.PhotoImage(Image.open(path))
    generated.config(image=photo)
    generated.image = photo
    root.update_idletasks()


def show():

    try:
        progress.grid(row=6, columnspan=2, padx=5, pady=5)
        main.main(frame, progress, mask, filename)
    except Exception as e:
        print('Exited in gui', e)
    finally:
        post()


t1 = Thread()


def threading():
    global mask
    mask = clicked.get().lower()
    isMask = False
    isFile = True
    if mask == 'mask type':
        maskDetailLabel.config(text="No mask selected")
    else:
        isMask = True
        maskDetailLabel.config(text='Mask selected is ' + clicked.get())
    maskDetailLabel.config(font=("Arial", 10))
    if not filename:
        isFile = False
        label_file_explorer.config(text="No file selected")
    if isMask and isFile:
        global t1
        t1 = Thread(target=show)
        if t1.is_alive():
            t1.join()
        else:
            t1.start()


def disable_event():
    pass


def destroy_event():
    if t1 and t1.is_alive():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            root.destroy()
    else:
        root.destroy()


options = [
    "Mask Type",
    "Right",
    "Left",
    "Top",
    "Bottom",
    "Center",
    "Diagonal",
    "Random"
]

maskLabel = Label(frame, text="Select Mask")
maskLabel.grid(row=1, column=0, padx=5, pady=5)
maskLabel.config(font=("Arial", 10))
# datatype of menu text
clicked = StringVar()

# initial menu text
clicked.set("Random")

# Create Dropdown menu
drop = OptionMenu(frame, clicked, *options)
drop.grid(row=1, column=1, padx=5, pady=5)


# Create Label
maskDetailLabel = Label(frame, text=" ")
maskDetailLabel.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# PSNR Label
psnr_value = Label(frame, text=" ")
psnr_value.grid(row=7, column=0, columnspan=2, padx=5, pady=5)


browseFileLabel = Label(frame, text="Select Image")
browseFileLabel.grid(row=3, column=0, padx=5, pady=5)
browseFileLabel.config(font=("Arial", 10))


filename = Button(frame, text="Browse Files", command=browseFiles).grid(row=3, column=1, padx=5, pady=5)


label_file_explorer = Label(frame, text="")
label_file_explorer.grid(row=4, column=0, columnspan=2, padx=5, pady=5)
label_file_explorer.config(font=("Arial", 10))

generated = Label(frame, text="")
generated.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

button = Button(frame, text="Run", style='A.TButton', command=threading).grid(row=8, column=0, padx=5, pady=5, columnspan=2)

progress = Progressbar(frame, orient=HORIZONTAL, length=100, mode='determinate')

exitBtn = Button(frame, text='Exit', style='W.TButton', command=destroy_event)
exitBtn.grid(row=9, column=0, padx=5, pady=5, columnspan=2)


root.protocol("WM_DELETE_WINDOW", disable_event)
root.mainloop()
