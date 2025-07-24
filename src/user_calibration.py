import csv
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from preprocessing.filter import process_bee_wing
from utils.helpers import noise_level_detection
from utils.helpers import string_to_tuple
from utils.helpers import tuple_to_string


def execute_function(*args):
    """
    This function is called when the 'Execute' button is pressed.

    It feeds the values to the filter script and shows the debugging arguments.
    """
    print("Executing function with the following inputs:")
    # Create a formatted string with all the inputs
    # Enumerate provides an index (i) and the value (arg)
    try:
        params = parse_args(args)
    except Exception as E:
        messagebox.showerror("Error", f"{E}")
        return

    print("Executing process_bee_wing with the following arguments:")
    print(params[2:13])
    print(f"debug_vars: {params[13]}")

    process_bee_wing(params[0], params[2:13], Path(params[14]), *params[13])

    # Display the inputs in a user-friendly message box
    print("-" * 20)


def submit_function(*args):
    """
    This function is called when the 'Submit' button is pressed.

    It adds the set of values to the output directory.
    """
    try:
        params = parse_args(args)

        noise_level = noise_level_detection(params[0])
        data_row = [params[1], noise_level[1]] + params[2:13]
        tuple_list = [5, 7, 10, 11, 12]
        for i in tuple_list:
            data_row[i] = tuple_to_string(data_row[i])
        file_path = Path(params[14]) / "output.csv"

        with open(file_path, "a", newline="", encoding="utf-8") as csvfile:
            # Create a writer object to convert data into delimited strings
            writer = csv.writer(csvfile)

            # Write the new data row
            writer.writerow(data_row)

    except Exception as E:
        messagebox.showerror("Error", f"{E}")
        return

    print(f"Added {data_row} to {file_path}")
    print("-" * 20)


def parse_args(args: tuple):
    result = []
    iter = 0
    for _ in range(2):
        result.append(args[iter])
        iter += 1

    for _ in range(3):
        result.append(int(args[iter]))
        iter += 1

    result.append(string_to_tuple(args[iter]))
    iter += 1

    result.append(float(args[iter]))
    iter += 1

    result.append(string_to_tuple(args[iter]))
    iter += 1

    result.append(int(args[iter]))
    iter += 1

    result.append(float(args[iter]))
    iter += 1

    for _ in range(3):
        result.append(string_to_tuple(args[iter]))
        iter += 1

    result.append(args[iter].replace(" ", "").split(","))
    iter += 1

    result.append(args[iter])
    iter += 1

    return result


def main():
    """
    Main function to set up and run the Tkinter application.
    """
    # --- Window Setup ---
    root = tk.Tk()
    root.title("User Calibration")
    root.geometry("1000x1000")  # Set a default size for the window

    # --- Main Frame ---
    main_frame = tk.Frame(root, padx=20, pady=20)
    main_frame.pack(fill="both", expand=True)

    # --- Input Fields ---
    # A list to hold the Entry widgets and their associated StringVars
    field_names = [
        "image_path",
        "folder_name",
        "nlm_h",
        "nlm_tws",
        "nlm_sws",
        "gb_kernel",
        "clahe_cl",
        "clahe_tgs",
        "thresh_bs",
        "thresh_c",
        "morphx_kernel",
        "kernel_open",
        "kernel_close",
        "debug_args",
        "output_destination",
    ]
    input_entries = []

    for i in range(15):
        # Create a frame for each row to group the label and entry field
        row_frame = tk.Frame(main_frame)
        row_frame.pack(
            fill="x", padx=10, pady=2
        )  # fill="x" makes the frame expand horizontally

        # Label for the input field
        label = tk.Label(row_frame, text=field_names[i], width=20)
        label.pack(side="left")

        # The Entry widget where the user can type
        entry = tk.Entry(row_frame)
        entry.pack(
            padx=10, side="left", fill="x", expand=True
        )  # expand=True allows the entry to fill the frame

        input_entries.append(entry)

    # --- Text Lines ---
    hello_label = tk.Label(
        main_frame,
        text="debug_args: gray, denoised, blured, clahe, enhanced_gray, thresh, closed_binary, cleaned_binary",
    )
    hello_label.pack(pady=(10, 0))

    hello_label = tk.Label(
        main_frame,
        text="opened_image, cleaned_image,output_image, skeleton, cropped_image",
    )
    hello_label.pack(pady=(10, 0))

    # --- Execute Button ---
    # The command uses a lambda function to gather the current text from all
    # entry widgets and pass them to the execute_function.
    # The '*' unpacks the list of strings into separate arguments.
    execute_button = tk.Button(
        main_frame,
        text="Execute",
        command=lambda: execute_function(*(entry.get() for entry in input_entries)),
        bg="#4CAF50",  # A nice green color
        fg="white",  # White text
        pady=5,
        padx=10,
        font=("Helvetica", 10, "bold"),
    )
    # Use pack with pady to add some space above the button
    execute_button.pack(pady=20)

    execute_button = tk.Button(
        main_frame,
        text="Submit",
        command=lambda: submit_function(*(entry.get() for entry in input_entries)),
        bg="#ffdd00",  # A nice yellow color
        fg="black",  # White text
        pady=5,
        padx=10,
        font=("Helvetica", 10, "bold"),
    )
    # Use pack with pady to add some space above the button
    execute_button.pack(pady=20)

    # --- Start the Application ---
    # The mainloop() call is what displays the window and listens for events
    # like button clicks and keyboard input.
    print("Starting the application...")
    root.mainloop()


if __name__ == "__main__":
    main()
