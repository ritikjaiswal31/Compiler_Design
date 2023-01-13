# This is the GUI for Compiler made by Ritik Jaiswal

# Import the tkinter libraries

from tkinter import *
from tkinter.filedialog import asksaveasfilename, askopenfilename
import subprocess

compiler_ritikjaiswal = Tk()
compiler_ritikjaiswal.title('Compiler_Ritik Jaiswal_500084079')
a = ''

# Define the set file path function

def set_a(path):
    global a
    a = path

# Declare a function for opening of file

def open_file():
    path = askopenfilename(filetypes=[('Python Files', '*.py')])
    with open(path, 'r') as file:
        code = file.read()
        editor.delete('1.0', END)
        editor.insert('1.0', code)
        set_a(path)

# Declare a function for saving of file

def save_file():
    if a == '':
        path = asksaveasfilename(filetypes=[('Python Files', '*.py')])
    else:
        path = a
    with open(path, 'w') as file:
        code = editor.get('1.0', END)
        file.write(code)
        set_a(path)

# Declare a function for Running of file

def Run():
    if a == '':
        save_prompt = Toplevel()
        text = Label(save_prompt, text='Kindly Save Your Code')
        text.pack()
        return
    command = f'python {a}'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    code_output.insert('1.0', output)
    code_output.insert('1.0',  error)

# Create the menu operation and call it

menu_operation = Menu(compiler_ritikjaiswal)

file_menu = Menu(menu_operation, tearoff=0)
file_menu.add_command(label='Open', command=open_file)
file_menu.add_command(label='Save', command=save_file)
file_menu.add_command(label='Save As', command=save_file)
file_menu.add_command(label='Exit', command=exit)
menu_operation.add_cascade(label='File', menu=file_menu)

Run_bar = Menu(menu_operation, tearoff=0)
Run_bar.add_command(label='Run', command=Run)
menu_operation.add_cascade(label='Run', menu=Run_bar)

compiler_ritikjaiswal.config(menu=menu_operation)

editor = Text()
editor.pack()

code_output = Text(height=10)
code_output.pack()

compiler_ritikjaiswal.mainloop()