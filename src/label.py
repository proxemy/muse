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

	entry = tk.Entry(
		fg="yellow",
		bg="blue",
		width=50
	)
	entry.pack()

	text = tk.Text()
	text.pack()

	frame_a = tk.Frame()
	frame_b = tk.Frame()

	label_a = tk.Label(master=frame_a, text="I'm in Frame A")
	label_a.pack()

	label_b = tk.Label(master=frame_b, text="I'm in Frame B")
	label_b.pack()

	frame_a.pack()
	frame_b.pack()

	window.mainloop()
