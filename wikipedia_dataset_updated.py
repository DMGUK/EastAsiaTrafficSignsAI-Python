import os
import requests
import cairosvg
from PIL import Image
import io
import time
import shutil
from urllib.parse import unquote, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import numpy as np
import random

# ——— ADD Albumentations for augmentations ——————————————————————————————
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=30, p=0.5),
    A.Blur(blur_limit=3, p=0.2),
    A.CLAHE(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
# ——————————————————————————————————————————————————————————————————————

# Directory and target number of images per class
DATASET_DIR = r"wikipedia_dataset1"
TARGET_COUNT = 150

# Wikipedia URLs to scrape
URLS = [
    "https://en.wikipedia.org/wiki/Road_signs_in_China#Prohibitory_signs",
    "https://en.wikipedia.org/wiki/Road_signs_in_China#Indicative_signs",
    "https://en.wikipedia.org/wiki/Road_signs_in_China#Warning_signs",
    "https://en.wikipedia.org/wiki/Road_signs_in_South_Korea#Warning_signs",
    "https://en.wikipedia.org/wiki/Road_signs_in_South_Korea#Prohibition_signs",
    "https://en.wikipedia.org/wiki/Road_signs_in_South_Korea#Mandatory_instruction_signs",
    "https://en.wikipedia.org/wiki/Road_signs_in_Japan#Warning_signs",
    "https://en.wikipedia.org/wiki/Road_signs_in_Japan#Regulatory_signs",
    "https://en.wikipedia.org/wiki/Road_signs_in_Japan#Instruction_signs"
]

def country_from_url(url):
    if "China" in url:
        return "China"
    elif "South_Korea" in url:
        return "South Korea"
    elif "Japan" in url:
        return "Japan"
    else:
        return "Unknown"

def setup_driver():
    print("🔧 Setting up ChromeDriver...")
    options = Options()
    options.headless = True
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    print("✅ ChromeDriver setup complete.")
    return driver

def convert_svg_to_jpg(svg_path):
    print(f"🔄 Converting SVG to JPG: {svg_path}")
    try:
        png_data = cairosvg.svg2png(url=svg_path)
        img = Image.open(io.BytesIO(png_data))
        jpg_path = svg_path.replace(".svg", ".jpg")
        background = Image.new("RGB", img.size, (255,255,255))
        if img.mode == "RGBA":
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img)
        background.save(jpg_path, "JPEG", quality=95)
        os.remove(svg_path)
        print(f"✅ Replaced {svg_path} with {jpg_path}")
        return jpg_path
    except Exception as e:
        print(f"❌ Error converting {svg_path} to JPG: {e}")
        return None

def update_image_background(image_path):
    try:
        img = Image.open(image_path)
        orig_w, orig_h = img.size
        new_size = (orig_w*4, orig_h*4)
        background = Image.new("RGB", new_size, (255,255,255))
        top_left = ((new_size[0]-orig_w)//2, (new_size[1]-orig_h)//2)
        background.paste(img, top_left)
        background.save(image_path, "JPEG", quality=95)
        print(f"✅ Updated image background for: {image_path}")
    except Exception as e:
        print(f"❌ Error updating background for {image_path}: {e}")

def calculate_yolo_bbox(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error loading image {image_path}.")
        return (0.5,0.5,1.0,1.0)
    h,w = img.shape[:2]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thr = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts,_ = cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print(f"⚠️ No contours for {image_path}.")
        return (0.5,0.5,1.0,1.0)
    x,y,box_w,box_h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
    cx,cy = x+box_w/2, y+box_h/2
    norm_w = random.uniform(0.40,0.55)
    norm_h = random.uniform(0.40,0.55)
    new_w, new_h = int(norm_w*w), int(norm_h*h)
    x_new = max(0, min(int(cx-new_w/2), w-new_w))
    y_new = max(0, min(int(cy-new_h/2), h-new_h))
    cx_n = (x_new+new_w/2)/w
    cy_n = (y_new+new_h/2)/h
    nw = new_w/w
    nh = new_h/h
    print(f"✅ BBox for {image_path}: {cx_n:.3f},{cy_n:.3f},{nw:.3f},{nh:.3f}")
    return (cx_n, cy_n, nw, nh)

def create_label(image_path, class_id, bbox):
    img_name   = os.path.basename(image_path)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_dir  = os.path.join(DATASET_DIR, "train", "labels")
    os.makedirs(label_dir, exist_ok=True)
    label_path = os.path.join(label_dir, label_name)
    x,y,w_box,h_box = bbox
    with open(label_path,"w",encoding="utf-8") as f:
        f.write(f"{class_id} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}")
    print(f"✅ Created label: {label_path}")

def extract_images_from_wiki(url, driver):
    print(f"🔄 Processing page: {url}")
    driver.get(url); time.sleep(2)
    soup = BeautifulSoup(driver.page_source,"html.parser")
    section_id = urlparse(url).fragment
    if not section_id:
        return []
    heading = soup.find(id=section_id)
    if not heading:
        return []
    items = heading.find_all_next("li",class_="gallerybox")
    images = []
    for it in items:
        prev = it.find_previous(["h2","h3"])
        if prev and prev.get("id")!=section_id:
            break
        img = it.find("img")
        if img and img.get("src"):
            img_url = "https:"+img["src"]
            txt_div = it.find("div",class_="gallerytext")
            name    = txt_div.get_text(strip=True) if txt_div else "unknown"
            images.append((img_url, name, section_id))
    print(f"✅ Found {len(images)} images for '{section_id}'.")
    return images[:5]

def balance_dataset_fixed_amount(dataset_dir, target_count=TARGET_COUNT):
    images_dir = os.path.join(dataset_dir,"train","images")
    labels_dir = os.path.join(dataset_dir,"train","labels")
    class_files, max_idx = {}, -1
    for fn in os.listdir(images_dir):
        if not fn.lower().endswith((".jpg",".jpeg",".png")): continue
        idx = int(fn.split("_")[1].split(".")[0])
        max_idx = max(max_idx, idx)
        lbl = os.path.join(labels_dir, fn.rsplit(".",1)[0]+".txt")
        if not os.path.exists(lbl): continue
        with open(lbl) as f:
            cid = f.readline().split()[0]
        class_files.setdefault(cid, []).append((os.path.join(images_dir,fn), lbl))
    if max_idx<0:
        print("⚠️ No images to balance."); return
    print(f"🔎 Balancing to {target_count} images per class.")
    nxt = max_idx+1
    for cid, pairs in class_files.items():
        cnt = len(pairs)
        print(f"  Class {cid}: {cnt} images")
        if cnt < target_count:
            for _ in range(target_count-cnt):
                src_img, src_lbl = random.choice(pairs)
                dst_img = os.path.join(images_dir, f"image_{nxt}.jpg")
                dst_lbl = os.path.join(labels_dir, f"image_{nxt}.txt")
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                nxt += 1
        elif cnt > target_count:
            for src_img, src_lbl in random.sample(pairs, cnt-target_count):
                os.remove(src_img); os.remove(src_lbl)
    print("✅ Balancing complete.")

def move_images_to_partition(images, partition, start_number):
    timg = os.path.join(DATASET_DIR,partition,"images")
    tlbl = os.path.join(DATASET_DIR,partition,"labels")
    os.makedirs(timg,exist_ok=True); os.makedirs(tlbl,exist_ok=True)
    for img_path in images:
        ext = os.path.splitext(img_path)[1]
        nm = f"image_{start_number}{ext}"
        dst_img = os.path.join(timg,nm)
        shutil.move(img_path, dst_img)
        lbl_src = os.path.join(os.path.dirname(img_path).replace("images","labels"),
                               os.path.basename(img_path).rsplit(".",1)[0]+".txt")
        lbl_dst = os.path.join(tlbl, f"image_{start_number}.txt")
        if os.path.exists(lbl_src):
            shutil.move(lbl_src, lbl_dst)
        start_number += 1
    return start_number

def partition_train_to_test_and_valid(train_images):
    random.shuffle(train_images)
    n = len(train_images)
    t_count = int(n*0.1); v_count = int(n*0.2)
    test_imgs  = train_images[:t_count]
    valid_imgs = train_images[t_count:t_count+v_count]
    print(f"📂 test={t_count}, valid={v_count}, train={n-t_count-v_count}")
    cnt = 0
    cnt = move_images_to_partition(test_imgs,  "test",  cnt)
    cnt = move_images_to_partition(valid_imgs, "valid", cnt)
    print("✅ Partitioning complete.")

def create_folders():
    print("🔄 Creating folder structure...")
    for sp in ["train","valid","test"]:
        os.makedirs(os.path.join(DATASET_DIR,sp,"images"),exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR,sp,"labels"),exist_ok=True)
    print("✅ Directories created.")

global_class_mapping = {}
next_class_id = 0

def main():
    create_folders()
    driver = setup_driver()
    processed = []
    counter = 0

    # 1) Download & label
    for url in URLS:
        country = country_from_url(url)
        entries = extract_images_from_wiki(url, driver)
        for img_url, sign, sec in entries:
            sign = f"{country}_{sign}"
            try:
                r = requests.get(img_url, headers={'User-Agent':'Mozilla/5.0'}, stream=True)
                if r.status_code != 200: continue
            except:
                continue
            ext = os.path.splitext(unquote(img_url).split("/")[-1])[1].lower()
            tmp = os.path.join(DATASET_DIR,"train","images", f"tmp_{counter}{ext}")
            with open(tmp,"wb") as f:
                for chunk in r.iter_content(1024): f.write(chunk)
            if ext == ".svg":
                conv = convert_svg_to_jpg(tmp)
                if not conv: continue
                tmp, ext = conv, ".jpg"
            update_image_background(tmp)
            bbox = calculate_yolo_bbox(tmp)
            global global_class_mapping, next_class_id
            if sign not in global_class_mapping:
                global_class_mapping[sign] = next_class_id
                next_class_id += 1
            cid = global_class_mapping[sign]
            idx = counter
            final_img = os.path.join(DATASET_DIR,"train","images",f"image_{idx}.jpg")
            os.rename(tmp, final_img)
            create_label(final_img, cid, bbox)
            processed.append(final_img)
            counter += 1

            # 2) Oversample 5 copies with Albumentations
            orig_img = final_img
            orig_lbl = os.path.join(DATASET_DIR,"train","labels",f"image_{idx}.txt")
            img = cv2.imread(orig_img)
            with open(orig_lbl,"r") as fh:
                parts = fh.readline().split()
                cls_int = int(parts[0])
                bbox0 = tuple(map(float, parts[1:]))

            for _ in range(5):
                cp_img = os.path.join(DATASET_DIR,"train","images",f"image_{counter}.jpg")
                cp_lbl = os.path.join(DATASET_DIR,"train","labels",f"image_{counter}.txt")

                aug = transform(image=img, bboxes=[bbox0], class_labels=[cls_int])
                aug_img = aug["image"]
                x2,y2,w2,h2 = aug["bboxes"][0]

                cv2.imwrite(cp_img, aug_img)
                with open(cp_lbl,"w",encoding="utf-8") as fh:
                    fh.write(f"{cls_int} {x2:.6f} {y2:.6f} {w2:.6f} {h2:.6f}")

                processed.append(cp_img)
                counter += 1

    driver.quit()
    print("✅ Download & processing done.")
    for s,c in global_class_mapping.items():
        print(f"Class {c}: {s}")

    # 3) Balance & partition
    balance_dataset_fixed_amount(DATASET_DIR, TARGET_COUNT)
    train_dir = os.path.join(DATASET_DIR,"train","images")
    train_imgs = [os.path.join(train_dir,f) for f in os.listdir(train_dir)
                  if f.lower().endswith((".jpg",".jpeg",".png"))]
    partition_train_to_test_and_valid(train_imgs)

    # 4) Write data.yaml with full Windows paths
    yaml_path = os.path.join(DATASET_DIR,"data.yaml")
    names = [None]*len(global_class_mapping)
    for s,c in global_class_mapping.items():
        names[c] = s
    abs_base = r"C:/Users/dmytr/dataset/wikipedia_dataset1"
    with open(yaml_path,"w",encoding="utf-8") as f:
        f.write(f"train: {abs_base}/train/images\n")
        f.write(f"val:   {abs_base}/valid/images\n")
        f.write(f"test:  {abs_base}/test/images\n\n")
        f.write(f"names: {names}\n")
        f.write(f"nc: {len(names)}\n")
    print(f"✅ data.yaml saved at {yaml_path}")
    print("✅ All done!")

if __name__ == "__main__":
    main()
