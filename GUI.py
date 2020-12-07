import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import torch
import torchvision.models as models
from torchvision import transforms
from torch import nn

classes = [ "affenpinscher", 
            "afghan_hound",
            "african_hunting_dog",
            "airedale",
            "american_staffordshire_terrier",
            "appenzeller",
            "australian_terrier",
            "basenji",
            "basset",
            "beagle",
            "bedlington_terrier",
            "bernese_mountain_dog",
            "black-and-tan_coonhound",
            "blenheim_spaniel",
            "bloodhound",
            "bluetick",
            "border_collie",
            "border_terrier",
            "borzoi",
            "boston_bull",
            "bouvier_des_flandres",
            "boxer",
            "brabancon_griffon",
            "briard",
            "brittany_spaniel",
            "bull_mastiff",
            "cairn",
            "cardigan",
            "chesapeake_bay_retriever",
            "chihuahua",
            "chow",
            "clumber",
            "cocker_spaniel",
            "collie",
            "curly-coated_retriever",
            "dandie_dinmont",
            "dhole",
            "dingo",
            "doberman",
            "english_foxhound",
            "english_setter",
            "english_springer",
            "entlebucher",
            "eskimo_dog",
            "flat-coated_retriever",
            "french_bulldog",
            "german_shepherd",
            "german_short-haired_pointer",
            "giant_schnauzer",
            "golden_retriever",
            "gordon_setter",
            "great_dane",
            "great_pyrenees",
            "greater_swiss_mountain_dog",
            "groenendael",
            "ibizan_hound",
            "irish_setter",
            "irish_terrier",
            "irish_water_spaniel",
            "irish_wolfhound",
            "italian_greyhound",
            "japanese_spaniel",
            "keeshond",
            "kelpie",
            "kerry_blue_terrier",
            "komondor",
            "kuvasz",
            "labrador_retriever",
            "lakeland_terrier",
            "leonberg",
            "lhasa",
            "malamute",
            "malinois",
            "maltese_dog",
            "mexican_hairless",
            "miniature_pinscher",
            "miniature_poodle",
            "miniature_schnauzer",
            "newfoundland",
            "norfolk_terrier",
            "norwegian_elkhound",
            "norwich_terrier",
            "old_english_sheepdog",
            "otterhound",
            "papillon",
            "pekinese",
            "pembroke",
            "pomeranian",
            "pug",
            "redbone",
            "rhodesian_ridgeback",
            "rottweiler",
            "saint_bernard",
            "saluki",
            "samoyed",
            "schipperke",
            "scotch_terrier",
            "scottish_deerhound",
            "sealyham_terrier",
            "shetland_sheepdog",
            "shih-tzu",
            "siberian_husky",
            "silky_terrier",
            "soft-coated_wheaten_terrier",
            "staffordshire_bullterrier",
            "standard_poodle",
            "standard_schnauzer",
            "sussex_spaniel",
            "tibetan_mastiff",
            "tibetan_terrier",
            "toy_poodle",
            "toy_terrier",
            "vizsla",
            "walker_hound",
            "weimaraner",
            "welsh_springer_spaniel",
            "west_highland_white_terrier",
            "whippet",
            "wire-haired_fox_terrier",
            "yorkshire_terrier" ]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = models.resnet152(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)
model = model.to(device)
model.load_state_dict(torch.load("uniquename.pth", map_location=device))
model.eval()

# #initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Dog Breed Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    trf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.open(file_path)
    image = trf(image)
    output = model(image.unsqueeze(0).to(device))
    pred = classes[output.argmax(dim=1, keepdim=True)]
    print(pred)
    label.configure(foreground='#011638', text=pred) 

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know the Breed of You Dog!",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()