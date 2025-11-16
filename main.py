import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os

# ===== MODEL PATH =====
PROTO = "models/colorization_deploy_v2.prototxt"        # model structure
MODEL = "models/colorization_release_v2.caffemodel"     # trained weights
PTS = "models/pts_in_hull.npy"                          # ab cluster centers

orig_img = None         # store original image
result_img = None       # store colorized result

zoom_before = 1.0       # zoom factor for before image
zoom_after = 1.0        # zoom factor for after image


# ===============================================================
#   COLORIZATION
# ===============================================================

def optimize_colors(img):
    return img        # placeholder (no extra color processing)


def load_colorization_model():
    if not (os.path.exists(PROTO) and os.path.exists(MODEL) and os.path.exists(PTS)):
        messagebox.showerror("Model Error", "Model is missing.")
        return None, None

    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)        # load model
    pts = np.load(PTS)                                  # load cluster points

    pts = pts.transpose().reshape(2,313,1,1)            # reshape cluster centers
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
        np.full([1,313], 2.606, dtype="float32")        # standard bias fix
    ]

    return net, pts


def colorize_image(img_pil):
    net, pts = load_colorization_model()
    if net is None:
        return None

    img = np.array(img_pil)                              # convert PIL → array
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype("float32") / 255.0

    h, w = img_bgr.shape[:2]                             # original size

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)       # convert to LAB
    L = lab[:,:,0]                                       # extract L channel

    L_rs = cv2.resize(L, (224,224))                      # resize for model
    L_rs -= 50                                           # normalize brightness

    net.setInput(cv2.dnn.blobFromImage(L_rs))            # forward pass
    ab_dec = net.forward()[0,:,:,:].transpose((1,2,0))   # decode ab channels

    ab_us = cv2.resize(ab_dec, (w,h))                    # upsample ab

    lab_out = np.zeros((h,w,3))                          # prepare LAB output
    lab_out[:,:,0] = L                                   # L stays original
    lab_out[:,:,1:] = ab_us                              # predicted ab

    rgb_out = cv2.cvtColor(lab_out.astype("float32"), cv2.COLOR_LAB2RGB)
    rgb_out = np.clip(rgb_out * 255, 0, 255).astype("uint8")

    return optimize_colors(rgb_out)                      # optional final adjust


# ===============================================================
#   GUI
# ===============================================================

def create_canvas(frame):
    canvas = tk.Canvas(frame, bg="#1e1e1e", highlightthickness=0)    # dark canvas
    canvas.pack(fill="both", expand=True)
    return canvas


def show_image_on_canvas(canvas, pil_img):
    canvas.delete("all")                                      # clear canvas
    tk_img = ImageTk.PhotoImage(pil_img)                      # convert to Tk image
    canvas.image = tk_img                                     # store reference
    canvas.create_image(0, 0, anchor="nw", image=tk_img)      # draw image
    canvas.config(scrollregion=canvas.bbox("all"))            # update scroll area


# ---------- DRAG HANDLER ----------
def start_drag(event, canvas):
    canvas.scan_mark(event.x, event.y)                        # mark drag origin

def drag_move(event, canvas):
    canvas.scan_dragto(event.x, event.y, gain=1)              # drag movement


# ---------- ZOOM ONLY ----------
def zoom(event, canvas, is_before=True):
    global zoom_before, zoom_after

    if canvas.image is None:
        return "break"

    factor = 1.1 if event.delta > 0 else 0.9                  # zoom direction

    canvas.scale("all", event.x, event.y, factor, factor)     # apply zoom
    canvas.config(scrollregion=canvas.bbox("all"))            # update area

    if is_before:
        zoom_before *= factor
    else:
        zoom_after *= factor

    return "break"


def enable_zoom_only(canvas, is_before):
    canvas.unbind_all("<MouseWheel>")                          # clear bindings

    canvas.bind("<MouseWheel>", lambda e: zoom(e, canvas, is_before))   # zoom bind
    canvas.bind("<Button-4>", lambda e: zoom(e, canvas, is_before))     # Linux zoom
    canvas.bind("<Button-5>", lambda e: zoom(e, canvas, is_before))

    canvas.bind("<MouseWheel>", lambda e: "break")             # disable scroll
    canvas.bind("<Button-4>", lambda e: "break")
    canvas.bind("<Button-5>", lambda e: "break")


# ===============================================================
#   OPEN / COLORIZE / SAVE
# ===============================================================

def open_image():
    global orig_img, result_img

    p = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])   # pick file
    if not p:
        return

    orig_img = Image.open(p).convert("RGB")               # load image
    result_img = None                                     # reset output
    show_image_on_canvas(canvas_before, orig_img)         # show before image
    canvas_after.delete("all")                            # clear after canvas


def run_colorize():
    global orig_img, result_img

    if orig_img is None:
        messagebox.showerror("Error", "Choose an image first.")
        return

    out = colorize_image(orig_img)                        # run model
    if out is None:
        return

    result_img = Image.fromarray(out)                     # convert back to PIL
    show_image_on_canvas(canvas_after, result_img)        # show result


def save_result():
    if result_img is None:
        messagebox.showerror("Error", "No output to save.")
        return

    p = filedialog.asksaveasfilename(defaultextension=".png")      # save dialog
    if p:
        result_img.save(p)                                         # save file
        messagebox.showinfo("Saved", "Image saved successfully.")


# ===============================================================
#   BUILD GUI
# ===============================================================

root = tk.Tk()
root.title("Retro → Modern Colorizer (Dark Mode)")         # window title
root.geometry("1250x700")                                  # window size
root.configure(bg="#121212")                               # dark theme

title = tk.Label(root, text="AI Retro Colorizer", fg="white", bg="#121212",
                 font=("Segoe UI", 18, "bold"))            # title label
title.pack(pady=10)

frame_split = tk.Frame(root, bg="#121212")                 # split container
frame_split.pack(fill="both", expand=True)

# BEFORE
frame_left = tk.Frame(frame_split, bg="#121212")           # left panel
frame_left.pack(side="left", fill="both", expand=True)

tk.Label(frame_left, text="BEFORE", fg="white", bg="#121212",
         font=("Segoe UI", 12)).pack()                     # header text

canvas_before = create_canvas(frame_left)                  # before canvas
enable_zoom_only(canvas_before, True)

canvas_before.bind("<ButtonPress-1>", lambda e: start_drag(e, canvas_before))   # drag start
canvas_before.bind("<B1-Motion>", lambda e: drag_move(e, canvas_before))       # drag move


# AFTER
frame_right = tk.Frame(frame_split, bg="#121212")          # right panel
frame_right.pack(side="right", fill="both", expand=True)

tk.Label(frame_right, text="AFTER", fg="white", bg="#121212",
         font=("Segoe UI", 12)).pack()                     # header text

canvas_after = create_canvas(frame_right)                  # after canvas
enable_zoom_only(canvas_after, False)

canvas_after.bind("<ButtonPress-1>", lambda e: start_drag(e, canvas_after))     # drag start
canvas_after.bind("<B1-Motion>", lambda e: drag_move(e, canvas_after))         # drag move


# BUTTONS
btn_frame = tk.Frame(root, bg="#121212")                   # button area
btn_frame.pack(pady=10)

btn_style = {"bg": "#333", "fg": "white", "font": ("Segoe UI", 11), "width": 15}  # button style

tk.Button(btn_frame, text="Open Image", command=open_image, **btn_style).grid(row=0, column=0, padx=10)
tk.Button(btn_frame, text="Colorize", command=run_colorize, **btn_style).grid(row=0, column=1, padx=10)
tk.Button(btn_frame, text="Save Result", command=save_result, **btn_style).grid(row=0, column=2, padx=10)

root.mainloop()                                            # run GUI
