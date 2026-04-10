#!/usr/bin/env python3
# app.py - Login + Autosurf con inizializzazione corretta della sessione di surfing

import os
import time
import json
import threading
import gc
import re
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
MAX_CONSECUTIVE_FAILURES = 5

EASYHITS_EMAIL = "sandrominori50+uiszuzoqatr@gmail.com"
EASYHITS_PASSWORD = "DDnmVV45!!"
REFERER_URL = "https://www.easyhits4u.com/?ref=nicolacaporale"

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

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

# ================ SUPABASE =====================
def get_working_key():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        log("❌ Supabase not configured")
        return None
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
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
            log(f"📦 Chiave: {api_key[:10]}... (status: in_use)")
            return api_key
        else:
            log("❌ Nessuna chiave working/available")
            return None
    except Exception as e:
        log(f"❌ Supabase error: {e}")
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
        log(f"   ⚠️ Errore rilascio: {e}")

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
        log(f"   ❌ Token error: {e}")
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
        # Header per le richieste AJAX
        session.headers.update({
            'Referer': 'https://www.easyhits4u.com/surf/',
            'X-Requested-With': 'XMLHttpRequest'
        })
        return session
    return None

# ================ INIZIALIZZAZIONE SESSIONE SURF =====================
def init_surf_session(session):
    """Visita la pagina /surf/ per inizializzare la sessione e ottenere eventuali token"""
    log("🌐 Inizializzazione sessione surfing...")
    try:
        r = session.get("https://www.easyhits4u.com/surf/", verify=False, timeout=15)
        if r.status_code == 200:
            # Cerca eventuale token CSRF (es. name="csrf_token")
            match = re.search(r'name="csrf_token"\s+value="([^"]+)"', r.text)
            if match:
                csrf = match.group(1)
                session.headers.update({'X-CSRF-Token': csrf})
                log(f"   ✅ CSRF token trovato: {csrf[:10]}...")
            else:
                log("   ℹ️ Nessun CSRF token trovato (forse non serve)")
            # Aggiungi altri cookie che potrebbero essere settati
            time.sleep(2)
            return True
        else:
            log(f"   ⚠️ GET /surf/ risponde con {r.status_code}")
            return False
    except Exception as e:
        log(f"   ❌ Errore init: {e}")
        return False

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
        
        log("🔧 Costruzione indice FAISS (FlatL2)...")
        X_list = []
        batch_size = 500
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            X_list.append(np.array(batch['X'], dtype=np.float32))
        X_all = np.vstack(X_list)
        log(f"📊 Vettori caricati: {X_all.shape}")
        index = faiss.IndexFlatL2(vector_dim)
        index.add(X_all)
        log(f"✅ Indice FAISS creato con {index.ntotal} vettori")
        faiss_index = index
        del X_list, X_all
        gc.collect()
        return True
    except Exception as e:
        log(f"❌ Errore dataset: {e}")
        return False

# ================ FEATURE EXTRACTION (identica) =====================
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
def surf_loop(api_key, initial_session):
    session = initial_session
    consecutive_failures = 0
    captcha_counter = 0

    # Inizializza la sessione di surfing
    if not init_surf_session(session):
        log("❌ Impossibile inizializzare la sessione di surfing")
        return

    while True:
        try:
            # Ping alla homepage per mantenere sessione
            session.get("https://www.easyhits4u.com", verify=False, timeout=10)
            time.sleep(1)

            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
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
            # LOG PER DEBUG: stampiamo la risposta
            log(f"DEBUG risposta: {json.dumps(data, indent=2)[:500]}")

            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])

            if not urlid or not qpic or not picmap or len(picmap) < 5:
                log("⚠️ Dati incompleti, riprovo tra 10 secondi")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    log("❌ Troppi fallimenti consecutivi, esco.")
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
    log("🚀 EasyHits4U Bot - Login + Autosurf (con init surf)")
    log("="*50)

    max_keys_to_try = 5
    for attempt in range(max_keys_to_try):
        api_key = get_working_key()
        if not api_key:
            log("❌ Nessuna chiave disponibile")
            return

        log(f"🔑 Tentativo {attempt+1}/{max_keys_to_try} con chiave {api_key[:10]}...")
        session = do_login(api_key)
        if session:
            log("✅ Login riuscito")
            if not load_dataset_hf():
                log("❌ Dataset non caricato")
                release_key(api_key, 'broken')
                continue
            log("🚀 Avvio surf loop")
            surf_loop(api_key, session)
            release_key(api_key, 'used')
            return
        else:
            log("❌ Login fallito")
            release_key(api_key, 'broken')

    log("❌ Nessuna chiave funzionante")

if __name__ == "__main__":
    main()
