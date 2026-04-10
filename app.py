#!/usr/bin/env python3
# app.py - Surfing completo con Selenium (niente requests)

import os
import time
import threading
import gc
import cv2
import numpy as np
import faiss
import base64
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datasets import load_dataset

# ================ CONFIG =====================
DIM = 64
HEALTH_CHECK_PORT = int(os.environ.get('PORT', 10000))
MAX_CONSECUTIVE_FAILURES = 5

EASYHITS_EMAIL = "sandrominori50+uiszuzoqatr@gmail.com"
EASYHITS_PASSWORD = "DDnmVV45!!"

# Browserless WebSocket endpoint (per Selenium)
BROWSERLESS_KEYS = [
    "2TPBw78eoqITsdsc25e9ff6270092838010c06b1652627c8f",
    "2UB2mJ8Pu4KvAwya658a33c2af825bbe2f707870ba088d746",
    "2UB6xXPVzalwmFrdf68265d93b745fd095899467d21a32326",
    "2UB72G0jNe5RsxL6b2e845d0b94bb6897966e88f662bc99a7",
    "2UCe01EH3vUJLnP6d3f028660d770ed840a0c6b05b6dcf71e",
    "2UCyusO830dLAcyda29244c83c2bfa0217728908ff8810c42",
    "2UD3pQCcge39YhQce5797773c8508515a295a1298d0105b42",
    "2UDOf1dHJeNmeOl0a373211ade4280ba7e212cde93dfc9e20",
    "2UDOnpiBIFokFEBcb1017abfdd901756272f2ff182c4a9f32",
    "2UDPWeUf62vB2I8aa37152a5b515e5360c127d669b813f23c",
]

dataset = None
classes_fast = None
faiss_index = None
vector_dim = 33
server_ready = False

# ================ HEALTH CHECK =====================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, format, *args):
        pass

def run_health_server():
    global server_ready
    try:
        server = HTTPServer(('0.0.0.0', HEALTH_CHECK_PORT), HealthHandler)
        server_ready = True
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏥 Health check avviato su porta {HEALTH_CHECK_PORT}")
        server.serve_forever()
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Health check error: {e}")
        server_ready = False

health_thread = threading.Thread(target=run_health_server, daemon=True)
health_thread.start()
timeout = 10
while not server_ready and timeout > 0:
    time.sleep(0.5)
    timeout -= 0.5

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ================ DATASET =====================
def load_dataset_hf():
    global dataset, classes_fast, faiss_index
    log("📥 Caricamento dataset...")
    try:
        dataset = load_dataset("zenadazurli/easyhits4u-dataset", split="train", token=None)
        log(f"✅ Dataset caricato: {len(dataset)} vettori")
        class_names = dataset.features['y'].names
        classes_fast = {i: name for i, name in enumerate(class_names)}
        
        log("🔧 Costruzione indice FAISS...")
        index = faiss.IndexFlatL2(vector_dim)
        batch_size = 5000
        total = len(dataset)
        for i in range(0, total, batch_size):
            batch = dataset[i:i+batch_size]
            X_batch = np.array(batch['X'], dtype=np.float32)
            index.add(X_batch)
        log(f"✅ Indice FAISS creato con {index.ntotal} vettori")
        faiss_index = index
        gc.collect()
        return True
    except Exception as e:
        log(f"❌ Errore dataset: {e}")
        return False

# ================ FUNZIONI FEATURE =====================
def centra_figura(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (DIM, DIM))
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    aspect_ratio = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri != 0:
            circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0.0
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()
    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    raggi = [int(min(h,w)*r) for r in (0.2, 0.4, 0.6, 0.8)]
    radiale = []
    for r in raggi:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (cx,cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m/255.0 for m in mean])
    spaziale = []
    quadranti = [(0,0,cx,cy), (cx,0,w,cy), (0,cy,cx,h), (cx,cy,w,h)]
    for (x1,y1,x2,y2) in quadranti:
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m/255.0 for m in mean])
    vettore = radiale + spaziale + [circularity, aspect_ratio] + hu
    return np.array(vettore, dtype=float)

def predict(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None
    features = estrai_descrittori(img_crop).astype(np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(features, 1)
    best_idx = indices[0][0]
    true_label_idx = dataset['y'][best_idx]
    return classes_fast.get(int(true_label_idx), "errore")

def solve_captcha_with_selenium(driver, wait):
    """Riconosce le figure e clicca su quella duplicata"""
    try:
        # Attendi che le immagini siano caricate
        images = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img[src*='/simg/']")))
        if len(images) < 5:
            log(f"   ⚠️ Solo {len(images)} immagini trovate")
            return False
        
        labels = []
        for idx, img_elem in enumerate(images[:5]):
            src = img_elem.get_attribute("src")
            if not src:
                continue
            # Scarica immagine via Selenium
            img_data = driver.execute_script("""
                var xhr = new XMLHttpRequest();
                xhr.open('GET', arguments[0], false);
                xhr.responseType = 'blob';
                xhr.send();
                var reader = new FileReader();
                reader.readAsDataURL(xhr.response);
                return reader.result;
            """, src)
            img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            label = predict(img)
            labels.append(label)
        
        log(f"   Labels: {labels}")
        
        # Trova duplicato
        seen = {}
        chosen_idx = None
        for i, label in enumerate(labels):
            if label and label != "errore":
                if label in seen:
                    chosen_idx = seen[label]
                    break
                seen[label] = i
        
        if chosen_idx is None:
            log("   ❌ Nessun duplicato trovato")
            return False
        
        # Clicca sull'immagine duplicata
        images[chosen_idx].click()
        return True
        
    except Exception as e:
        log(f"   ❌ Errore riconoscimento: {e}")
        return False

def surf_loop_selenium(api_key):
    browserless_url = f"wss://chrome.browserless.io?token={api_key}"
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = None
    try:
        driver = webdriver.Remote(command_executor=browserless_url, options=options)
        wait = WebDriverWait(driver, 30)
        
        # Login
        log("🌐 Login...")
        driver.get("https://www.easyhits4u.com")
        login_btn = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Login")))
        login_btn.click()
        
        username_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
        password_field = driver.find_element(By.NAME, "password")
        username_field.send_keys(EASYHITS_EMAIL)
        password_field.send_keys(EASYHITS_PASSWORD)
        submit_btn = driver.find_element(By.XPATH, "//input[@type='submit']")
        submit_btn.click()
        
        # Attendi il redirect dopo login
        time.sleep(5)
        log("✅ Login effettuato")
        
        # Vai alla pagina di surfing
        driver.get("https://www.easyhits4u.com/surf/")
        time.sleep(3)
        
        captcha_counter = 0
        consecutive_failures = 0
        
        while True:
            try:
                # Clicca su "Start Surfing" se presente
                start_btn = driver.find_elements(By.XPATH, "//button[contains(text(), 'Start')]")
                if start_btn:
                    start_btn[0].click()
                    time.sleep(2)
                
                # Risolvi il captcha
                success = solve_captcha_with_selenium(driver, wait)
                if success:
                    captcha_counter += 1
                    log(f"✅ Captcha risolto! Totale: {captcha_counter}")
                    consecutive_failures = 0
                    time.sleep(5)
                else:
                    consecutive_failures += 1
                    log(f"❌ Fallito ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        break
                    time.sleep(10)
                
                if captcha_counter % 10 == 0:
                    gc.collect()
                    
            except Exception as e:
                log(f"❌ Errore nel loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(10)
                
    except Exception as e:
        log(f"❌ Errore fatale: {e}")
    finally:
        if driver:
            driver.quit()

# ================ MAIN =====================
def main():
    log("="*50)
    log("🚀 EasyHits4U Bot - Surfing completo con Selenium")
    log("="*50)
    
    if not load_dataset_hf():
        log("❌ Dataset non caricato")
        return
    
    for api_key in BROWSERLESS_KEYS:
        log(f"🔑 Tentativo con chiave {api_key[:10]}...")
        try:
            surf_loop_selenium(api_key)
            log("✅ Surf loop completato con questa chiave")
            return
        except Exception as e:
            log(f"❌ Errore con chiave {api_key[:10]}: {e}")
            continue
    
    log("❌ Nessuna chiave funzionante")

if __name__ == "__main__":
    main()
