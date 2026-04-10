#!/usr/bin/env python3
# app.py - Login + Autosurf con Selenium + Browserless (endpoint corretto)

import os
import time
import json
import threading
import gc
import requests
import numpy as np
import cv2
import faiss
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ================ CONFIG =====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "/tmp/errori"
HEALTH_CHECK_PORT = int(os.environ.get('PORT', 10000))
MAX_CONSECUTIVE_FAILURES = 5

# Credenziali EasyHits4U
EASYHITS_EMAIL = "sandrominori50+uiszuzoqatr@gmail.com"
EASYHITS_PASSWORD = "DDnmVV45!!"

# Lista di chiavi Browserless (usa quelle che hai generato)
BROWSERLESS_KEYS = [
    "2UJK3J6z8WVUZCnebd8f5f45581cb8e33d54c5f102ff1ca1a",
    "2UJK4Jun2RJGbpmb4744ac717d57e27d86a6f8cdea79ecb29",
    "2UJK6yKb6025jjV0ec93e78221afdd7422cba5e9c2cf215b2",
    "2UJK7NrAnPQHmLj5f59ebaeb40664e36acb5e9edb16258649",
    "2UJK9tJF6fSUI1yee47518cbceb44b754091f65ffb37385e9",
    "2UJKBhoRgHclEJteebf8c7771d9b2ac024e173d5e8c668e63",
    "2UJKIOxgYKTLcPm78093f2ec30b29d3ed2796fd80812e30e4",
    "2UJKQznXMrCRsDe7e27bb3392684dc84617e99bfebb86c6f3",
    "2UJKVmGU7EHnhBa7888f73274495565fd975f87911d955624",
    "2UJKWATu0ywwDLj6745bd019eb949bc89ee0bde7b8aefcceb",
    "2UJKZKYxDW04Fvycd08e1f4373d86ac84939fd2da94b7bb6b",
    "2UJKbrJ8mu81DDId0b0a6d5d6f09d4232e86c95d0508d2286",
    "2UJKgrHd3wpB8ER8967264f75287a7a37b6c07cd1aa385e8a",
    "2UJKj90nSPoPzbGb83f67c8d51804f74d9c294296731f16d2",
    "2UJKkfMpBrzmSiT79baea21763312b842e8d76f0294e4922b",
    "2UJKmcIjXmvO8XRd305c7cbfdc378b8c2b51b9a01431bdb05",
    "2UJKneJJJ1geO9fc62ab067a9bba69951fc680ac31f68b318",
    "2UJKpDVy5uUTm5ke53ea97a630f3e9d40890cda9536bef640",
    "2UJKqHrcH8eFPru090c449d6f3349f227d34e0b64e05d9515",
    "2UJKstbtf4Yhhihbee152f1524f4b30b4169646dacc6f57b9",
    "2UJKu8uOav7pPEid44f93cbe5506e74a6aabdb11f8c3c51bb",
    "2UJKwZYaR9WoIdNdef8ca3188b79f0f1642cf040f908a6f0d",
    "2UJL4KwP022lo430841ccf7077da9eddca8f0e600bbb78ed0",
    "2UJL5zV86C4xbIwe2fa97d07bb917acf5c449b96ab73c9241",
    "2UJLTjMfcMlNuaw76c40208b2a61ce7f8fffa3fd8b570f8be"
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

# ================ LOGIN CON SELENIUM (Browserless) =====================
def do_login_selenium(api_key):
    # ENDPOINT CORRETTO: https, non wss, con /webdriver
    browserless_url = f"https://chrome.browserless.io/webdriver?token={api_key}"
    log(f"   🌐 Connessione a {browserless_url[:50]}...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    driver = None
    try:
        driver = webdriver.Remote(command_executor=browserless_url, options=options)
        driver.get("https://www.easyhits4u.com")
        wait = WebDriverWait(driver, 30)
        login_btn = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Login")))
        login_btn.click()
        username_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
        password_field = driver.find_element(By.NAME, "password")
        username_field.send_keys(EASYHITS_EMAIL)
        password_field.send_keys(EASYHITS_PASSWORD)
        submit_btn = driver.find_element(By.XPATH, "//input[@type='submit']")
        submit_btn.click()
        # Attendi il completamento del login (presenza di cookie o reindirizzamento)
        time.sleep(8)
        # Estrai cookie
        cookies = driver.get_cookies()
        session = requests.Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'])
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': 'https://www.easyhits4u.com/surf/',
            'X-Requested-With': 'XMLHttpRequest'
        })
        # Verifica presenza sesids
        cookie_dict = session.cookies.get_dict()
        if 'sesids' in cookie_dict:
            log(f"   ✅ Login OK! user_id={cookie_dict.get('user_id')}, sesids={cookie_dict['sesids']}")
            return session
        else:
            log(f"   ❌ Cookie sesids non trovato. Cookie ricevuti: {list(cookie_dict.keys())}")
            return None
    except Exception as e:
        log(f"   ❌ Errore Selenium: {e}")
        return None
    finally:
        if driver:
            driver.quit()

# ================ DATASET HUGGING FACE =====================
def load_dataset_hf():
    global dataset, classes_fast, faiss_index
    log("📥 Caricamento dataset da Hugging Face Hub...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("zenadazurli/easyhits4u-dataset", split="train", token=None)
        log(f"✅ Dataset caricato: {len(dataset)} vettori")
        class_names = dataset.features['y'].names
        classes_fast = {i: name for i, name in enumerate(class_names)}
        
        log("🔧 Costruzione indice FAISS (FlatL2) incrementale...")
        index = faiss.IndexFlatL2(vector_dim)
        batch_size = 5000
        total = len(dataset)
        for i in range(0, total, batch_size):
            batch = dataset[i:i+batch_size]
            X_batch = np.array(batch['X'], dtype=np.float32)
            index.add(X_batch)
            if i % 50000 == 0:
                log(f"   Aggiunti {len(X_batch)} vettori, totale {index.ntotal}/{total}")
        log(f"✅ Indice FAISS creato con {index.ntotal} vettori")
        faiss_index = index
        gc.collect()
        return True
    except Exception as e:
        log(f"❌ Errore dataset: {e}")
        return False

# ================ FUNZIONI DI FEATURE EXTRACTION =====================
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

def get_features(img):
    img_centrata = centra_figura(img)
    return estrai_descrittori(img_centrata)

def predict(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None
    features = get_features(img_crop).astype(np.float32).reshape(1, -1)
    distances, indices = faiss_index.search(features, 1)
    best_idx = indices[0][0]
    true_label_idx = dataset['y'][best_idx]
    return classes_fast.get(int(true_label_idx), "errore")

def crop_safe(img, coords):
    try:
        x1, y1, x2, y2 = map(int, coords.split(","))
    except:
        return None
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def salva_errore(qpic, img, picmap, labels, chosen_idx, motivo, urlid=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(ERRORI_DIR, f"{timestamp}_{qpic}")
    os.makedirs(folder, exist_ok=True)
    full_path = os.path.join(folder, "full.jpg")
    cv2.imwrite(full_path, img)
    for i, p in enumerate(picmap):
        crop = crop_safe(img, p.get("coords", ""))
        if crop is not None and crop.size > 0:
            cv2.imwrite(os.path.join(folder, f"crop_{i+1}.jpg"), crop)
    metadata = {
        "timestamp": timestamp,
        "qpic": qpic,
        "urlid": urlid,
        "motivo": motivo,
        "labels_predette": labels,
        "chosen_idx": chosen_idx,
        "picmap_values": [p.get("value") for p in picmap] if picmap else []
    }
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"📁 Errore salvato in {folder}")

# ================ SURF LOOP =====================
def surf_loop(session):
    consecutive_failures = 0
    captcha_counter = 0

    # Inizializzazione surfing: visita /surf/
    try:
        log("🌐 Inizializzazione /surf/...")
        r = session.get("https://www.easyhits4u.com/surf/", verify=False, timeout=15)
        log(f"   /surf/ status: {r.status_code}")
        time.sleep(3)
    except Exception as e:
        log(f"   ⚠️ Errore init: {e}")

    while True:
        try:
            timestamp = int(time.time() * 1000)
            r = session.post(
                f"https://www.easyhits4u.com/surf/?ajax=1&try=1&_={timestamp}",
                verify=False, timeout=REQUEST_TIMEOUT
            )
            if r.status_code != 200:
                log(f"⚠️ Status {r.status_code}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            data = r.json()
            if data.get("redirect"):
                log(f"⚠️ Redirect a {data['redirect']}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])

            if not urlid or not qpic or not picmap or len(picmap) < 5:
                log("⚠️ Dati incompleti, riprovo tra 10 secondi")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(10)
                continue

            consecutive_failures = 0

            img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
            img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
            labels = [predict(c) for c in crops]
            log(f"Labels: {labels}")

            seen = {}
            chosen_idx = None
            for i, label in enumerate(labels):
                if label and label != "errore":
                    if label in seen:
                        chosen_idx = seen[label]
                        break
                    seen[label] = i

            if chosen_idx is None:
                log("❌ Nessun duplicato trovato")
                salva_errore(qpic, img, picmap, labels, None, "nessun_duplicato", urlid)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            time.sleep(seconds)
            word = picmap[chosen_idx]["value"]
            resp = session.get(
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                verify=False
            )
            resp_json = resp.json()
            if resp_json.get("warning") == "wrong_choice":
                log("❌ Wrong choice")
                salva_errore(qpic, img, picmap, labels, chosen_idx, "wrong_choice", urlid)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            captcha_counter += 1
            log(f"✅ OK - indice {chosen_idx} - Totale: {captcha_counter}")
            if captcha_counter % 10 == 0:
                gc.collect()
                log("🧹 Garbage collection")
            time.sleep(2)

        except Exception as e:
            log(f"❌ Eccezione: {e}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                break
            time.sleep(10)

    log("🏁 Surf loop terminato")

# ================ MAIN =====================
def main():
    log("="*50)
    log("🚀 EasyHits4U Bot - Selenium + Browserless (endpoint corretto)")
    log("="*50)

    # Carica dataset
    if not load_dataset_hf():
        log("❌ Dataset non caricato, esco")
        return

    # Prova ogni chiave finché non funziona
    for api_key in BROWSERLESS_KEYS:
        log(f"🔑 Tentativo con chiave {api_key[:10]}...")
        session = do_login_selenium(api_key)
        if session:
            log("✅ Login riuscito! Avvio surf loop...")
            surf_loop(session)
            return
        else:
            log("❌ Login fallito, passo alla prossima chiave")

    log("❌ Nessuna chiave funzionante tra quelle disponibili")

if __name__ == "__main__":
    main()
