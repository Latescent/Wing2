import tkinter as tk
from tkinter import messagebox

def custom_function(*args):
    # This is where you can process the arguments.
    # For demonstration, we'll just show them in a message box.
    result = "Arguments:\n" + "\n".join(args)
    messagebox.showinfo("Submitted Values", result)

def on_submit():
    # Retrieve values from entry fields
    values = [entry.get() for entry in entries]
    custom_function(*values)

# Create the main window
root = tk.Tk()
root.title("Custom Function GUI")

# List to hold entry fields
entries = []

# Create 11 entry fields
for i in range(11):
    label = tk.Label(root, text=f"Field {i+1}:")
    label.grid(row=i, column=0, padx=10, pady=5)

    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

# Create a submit button
submit_button = tk.Button(root, text="Submit", command=on_submit)
submit_button.grid(row=11, columnspan=2, pady=10)

# Start the Tkinter event loop
root.mainloop()
