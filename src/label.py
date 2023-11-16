#!/usr/bin/env python3

import tkinter as tk


if __name__ == '__main__':
	window = tk.Tk()

	label = tk.Label(text="test text")
	label.pack()

	button = tk.Button(
		text="Click me",
		width=25,
		height=5,
		bg="blue",
		fg="white"
	)
	button.pack()

	window.mainloop()
