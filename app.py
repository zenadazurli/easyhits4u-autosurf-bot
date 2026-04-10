#!/usr/bin/env python3
# app.py - Login + Autosurf per EasyHits4U con rilogin automatico

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
from supabase import create_client

# ================ CONFIG =====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "/tmp/errori"
HEALTH_CHECK_PORT = int(os.environ.get('PORT', 10000))
MAX_CONSECUTIVE_FAILURES = 3

# Credenziali EasyHits4U
EASYHITS_EMAIL = "sandrominori50+uiszuzoqatr@gmail.com"
EASYHITS_PASSWORD = "DDnmVV45!!"
REFERER_URL = "https://www.easyhits4u.com/?ref=nicolacaporale"

# Supabase (variabili d'ambiente)
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Globals per dataset
dataset = None
classes_fast = None
faiss_index = None
vector_dim = 33
server_ready = False

# ================ HEALTH CHECK SERVER =====================
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🏥 Health check server avviato sulla porta {HEALTH_CHECK_PORT}")
        server.serve_forever()
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ ERRORE health check: {e}")
        server_ready = False

health_thread = threading.Thread(target=run_health_server, daemon=True)
health_thread.start()
timeout = 10
while not server_ready and timeout > 0:
    time.sleep(0.5)
    timeout -= 0.5

# ================ LOG =====================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ================ SUPABASE FUNCTIONS =====================
def get_working_key():
    """Recupera una chiave con status 'working' o 'available' e la marca come 'in_use'"""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        log("❌ Supabase non configurato")
        return None
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        # MODIFICA: accetta sia 'working' che 'available'
        resp = supabase.table('browserless_keys')\
            .select('id', 'api_key')\
            .in_('status', ['working', 'available'])\
            .limit(1)\
            .execute()
        if resp.data:
            key_id = resp.data[0]['id']
            api_key = resp.data[0]['api_key']
            supabase.table('browserless_keys')\
                .update({'status': 'in_use'})\
                .eq('id', key_id)\
                .execute()
            log(f"📦 Chiave ottenuta: {api_key[:10]}... (status: in_use)")
            return api_key
        else:
            log("❌ Nessuna chiave 'working' o 'available' disponibile")
            return None
    except Exception as e:
        log(f"❌ Errore Supabase: {e}")
        return None

def release_key(api_key, new_status='used'):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        supabase.table('browserless_keys')\
            .update({'status': new_status})\
            .eq('api_key', api_key)\
            .execute()
        log(f"   📝 Chiave {api_key[:10]}... → '{new_status}'")
    except Exception as e:
        log(f"   ⚠️ Errore rilascio chiave: {e}")

# ================ LOGIN (Browserless BQL) =====================
def get_cf_token(api_key):
    query = """
    mutation {
      goto(url: "https://www.easyhits4u.com/logon/", waitUntil: networkIdle, timeout: 60000) {
        status
      }
      solve(type: cloudflare, timeout: 60000) {
        solved
        token
        time
      }
    }
    """
    url = f"https://production-sfo.browserless.io/chrome/bql?token={api_key}"
    try:
        start = time.time()
        response = requests.post(url, json={"query": query}, headers={"Content-Type": "application/json"}, timeout=120)
        if response.status_code != 200:
            return None
        data = response.json()
        if "errors" in data:
            return None
        solve_info = data.get("data", {}).get("solve", {})
        if solve_info.get("solved"):
            token = solve_info.get("token")
            log(f"   ✅ Token ({time.time()-start:.1f}s)")
            return token
        return None
    except Exception as e:
        log(f"   ❌ Errore token: {e}")
        return None

def do_login(api_key):
    token = get_cf_token(api_key)
    if not token:
        return None
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Firefox/148.0',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': REFERER_URL,
    }
    data = {
        'manual': '1',
        'fb_id': '',
        'fb_token': '',
        'google_code': '',
        'username': EASYHITS_EMAIL,
        'password': EASYHITS_PASSWORD,
        'cf-turnstile-response': token,
    }
    session.get(REFERER_URL)
    resp = session.post("https://www.easyhits4u.com/logon/", data=data, headers=headers, allow_redirects=True, timeout=30)
    final_cookies = session.cookies.get_dict()
    if 'user_id' in final_cookies:
        log(f"   ✅ Login OK! user_id: {final_cookies['user_id']}")
        return session
    return None

# ================ DATASET HUGGING FACE (FAISS) =====================
def load_dataset_hf():
    global dataset, classes_fast, faiss_index
    log("📥 Caricamento dataset da Hugging Face Hub...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("zenadazurli/easyhits4u-dataset", split="train", token=None)
        log(f"✅ Dataset caricato: {len(dataset)} vettori")
        class_names = dataset.features['y'].names
        classes_fast = {i: name for i, name in enumerate(class_names)}
        
        log("🔧 Costruzione indice FAISS...")
        X_list = []
        batch_size = 500
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            X_list.append(np.array(batch['X'], dtype=np.float32))
        X_all = np.vstack(X_list)
        log(f"📊 Vettori caricati: {X_all.shape}")
        
        nlist = 100
        m = 3
        d = vector_dim
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
        log("🏋️ Addestramento indice FAISS...")
        index.train(X_all)
        index.add(X_all)
        log(f"✅ Indice FAISS creato con {index.ntotal} vettori")
        faiss_index = index
        del X_list, X_all
        gc.collect()
        return True
    except Exception as e:
        log(f"❌ Errore caricamento dataset: {e}")
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

# ================ SURF LOOP CON RILOGIN AUTOMATICO =====================
def surf_loop(api_key, initial_session):
    session = initial_session
    consecutive_failures = 0
    captcha_counter = 0

    while True:
        try:
            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                verify=False, timeout=REQUEST_TIMEOUT
            )
            if r.status_code != 200:
                log(f"⚠️ Status {r.status_code} - Cookie scaduto o errore server")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log(f"❌ Troppi fallimenti consecutivi ({consecutive_failures}), esco.")
                    break
                log(f"🔄 Tentativo di rilogin (fallimenti={consecutive_failures})...")
                new_session = do_login(api_key)
                if new_session:
                    session = new_session
                    continue
                else:
                    log("❌ Rilogin fallito, esco.")
                    break

            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])

            if not urlid or not qpic or not picmap or len(picmap) < 5:
                log("⚠️ Dati incompleti (cookie scaduto?) - richiedo rilogin")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log(f"❌ Troppi fallimenti consecutivi ({consecutive_failures}), esco.")
                    break
                log(f"🔄 Tentativo di rilogin (fallimenti={consecutive_failures})...")
                new_session = do_login(api_key)
                if new_session:
                    session = new_session
                    continue
                else:
                    log("❌ Rilogin fallito, esco.")
                    break

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
                    log("Troppi errori di riconoscimento consecutivi, esco.")
                    break
                new_session = do_login(api_key)
                if new_session:
                    session = new_session
                    continue
                else:
                    break

            time.sleep(seconds)
            word = picmap[chosen_idx]["value"]
            resp = session.get(
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                verify=False
            )
            resp_json = resp.json()
            if resp_json.get("warning") == "wrong_choice":
                log("❌ Wrong choice - salvo errore e provo a riloginare")
                salva_errore(qpic, img, picmap, labels, chosen_idx, "wrong_choice", urlid)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log("Troppi wrong choice consecutivi, esco.")
                    break
                new_session = do_login(api_key)
                if new_session:
                    session = new_session
                    continue
                else:
                    break

            captcha_counter += 1
            log(f"✅ OK - indice {chosen_idx} - Totale captcha: {captcha_counter}")
            if captcha_counter % 10 == 0:
                gc.collect()
                log(f"🧹 Garbage collection eseguita (captcha {captcha_counter})")
            time.sleep(2)

        except Exception as e:
            log(f"❌ Eccezione generica: {e}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                log("Troppe eccezioni consecutive, esco.")
                break
            log("Tentativo di rilogin...")
            new_session = do_login(api_key)
            if new_session:
                session = new_session
                continue
            else:
                break

    log("🏁 Surf loop terminato")

# ================ MAIN =====================
def main():
    log("=" * 50)
    log("🚀 EasyHits4U Bot - Login + Autosurf con rilogin automatico")
    log("=" * 50)

    api_key = get_working_key()
    if not api_key:
        log("❌ Impossibile proseguire senza chiave")
        return

    session = do_login(api_key)
    if not session:
        log("❌ Login fallito, rilascio chiave come broken")
        release_key(api_key, 'broken')
        return

    if not load_dataset_hf():
        log("❌ Dataset non caricato, abort")
        release_key(api_key, 'broken')
        return

    surf_loop(api_key, session)

    release_key(api_key, 'used')
    log("🏁 Bot terminato, in attesa di riavvio")

if __name__ == "__main__":
    main()
