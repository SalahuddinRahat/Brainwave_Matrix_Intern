import tkinter as tk
import customtkinter as ctk 

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline 

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Ar koto mama? Aktu ghuma") 
ctk.set_appearance_mode("dark") 

prompt = ctk.CTkEntry(master = app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white") 
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master = app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant="fp16", torch_dtype=torch.float16, use_auth_token=auth_token) 
pipe.to(device) 

from PIL import Image, ImageTk

def generate(): 
    with autocast(device): 
        image = pipe(prompt.get(), guidance_scale=8.5).images[0]
    
    # Save the full image
    image.save('generatedimage.png')
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    new_width = 512
    new_height = int(new_width / aspect_ratio)
    
    if new_height > 512:
        new_height = 512
        new_width = int(new_height * aspect_ratio)
    
    # Resize the image while maintaining the aspect ratio
    img_resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)  
    img = ImageTk.PhotoImage(img_resized)

    lmain.configure(image=img)
    lmain.image = img  # Prevent garbage collection



trigger = ctk.CTkButton(master=app,height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate) 
trigger.configure(text="Generate") 
trigger.place(x=206, y=60) 

app.mainloop()