#!/usr/bin/env python3
# app.py - Login BQL + Autosurf stile Divellaeasy

import requests
import json
import time
import os
import cv2
import numpy as np
import faiss
import gc
from datetime import datetime
from datasets import load_dataset

# ==================== CONFIG ====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "/tmp/errori"
MAX_CONSECUTIVE_FAILURES = 5

EASYHITS_EMAIL = "sandrominori50+uiszuzoqatr@gmail.com"
EASYHITS_PASSWORD = "DDnmVV45!!"
REFERER_URL = "https://www.easyhits4u.com/?ref=nicolacaporale"
BROWSERLESS_URL = "https://production-sfo.browserless.io/chrome/bql"

# Chiavi valide (quelle che funzionano nel login semplice)
VALID_KEYS = [
    "2UFyHOdxsID23VMa0518a22c6b683ea3c11c1bdca148d5381",
    "2UIAf0U41Twctlr77ecbfa2545692634758496b2eb88a170c",
    "2UIAhSj6AMSpgLM5400cb96e68c36236805887d583fa1c1a8",
    "2UIAkQ4DGbDLMB06db1a95369b032405097bcfe53b9b8d444",
    "2UIAoK9f3FItlml3f95c43bb78d2b15d3e274da5c52fcb5cd",
    "2UIArIu84xpGFuV1b4e825a86352e4bec7b54db59df943bf0",
    "2UIAsvzIYtc0o6Pa719bbb072a635a0140cee8591aec0e617",
    "2UIAzLYxMfMvBTTf24fef2bee78bd26ccc8e423b6dbd9d72c",
    "2UIB0BADWlWBhpUd9b3113aae7aec11928693179b8e97adf7",
    "2UIB8rlEnDrj6Cv44d507f520ec52fa50046e7a70c30df6c6",
    "2UIB9J2tCnemabr9e97eff9685066c2072e18a52cfa283aa9",
    "2UIBB3QQ3H39YFu7d4fd1c778669ef19c8db22610905f23bb",
    "2UIBC8fgRMkg9wZ41fe0fe622994483be7093f33c02e53835",
    "2UIBGwfAlxxB6ni8919255b5bc976ec9ff72e0e7ee7f020de",
    "2UIBHpFuiMsVdXx3403174d9c61f08000e61d09260287e390",
    "2UIBJUl1ne3E92ya0949e27d64225c71a87e1d01458304c98",
    "2UIBKJ1ZL4HeXTTef781aa5c7c90ff94cc7d8e04545cf5ff9",
    "2UIBMTvCwvbW8zyeb1a2c2fc6d628643d2fc7837706f662d4",
    "2UIBOuJaRF5cBah589a83ba07a2bf4b4ae1e0bede889db139",
    "2UIBQDGaiPhyK5cc7d8d10689c2376b516809e26a4331bbe7",
    "2UIBRMkIfmmc5wU462f920ea771e4b0e8c29a96509179becd",
    "2UIBTMXwg0OXKdLdb313c233f7b40884382642b1336a75475",
    "2UIBUw762KYlNYe436d56b56b785ae327aea06af5c57b0856",
    "2UIBWGd7CenkAZP4e84a28fc45390849c04ec824c6b70c4aa",
    "2UIBX2qFOoT6UfQf0dca472d23a39ee0d2cc679711254df6e",
    "2UIBZ6iew6q5MjY587ca12d2ba6a8a7dec2887c680e0a295d",
    "2UIBalRcxjMmhLraa054e3a3fcc66019fa02e4756d40a97ca",
    "2UIBcJf0KJwjIJCc6aa92098f4b4d9677b277fa08bddaa52f",
    "2UIBdWa0VtcPa7l291b4497fca8ed7ad26b5c4d5927f54c52",
    "2UIBfg3C0DBareT4b3bc7b9de04934615085d885e0037c6a9",
    "2UIBg9igA9Adum65d15c87a1ebdbdd8462f2b769b9e6d0534",
    "2UIBiv7UFTo86PL7733f37e8662dc5ac1e44fbbfa69938c47",
    "2UIBjq41So7iISXc9b6488e29439c45ac81ec6655413598b7",
    "2UIBlZtTVvSSd9Mef4e7f74c7dadf262e366cf0d52a9278e1",
    "2UIBmotaoPEgiLGb4d8ff65588ad03856bca142e29d10f9d7",
    "2UIBoXymrMnL6rB7c0bf5d89b1d24423cf95f989c717a93da",
    "2UIBqLMCQct1MEc93871eac596a18158adf155055ea891b82",
    "2UIBsC5kqg908ss2b15a06dfd516f5477e644f4970239c2f3",
    "2UIBtryD9TY1rfLf40876aea895c6b19cfccd6d0423bb1a5a",
    "2UIBvZWEqIfKMABdb7ad2379d49b5fdb791668c5b8ae2872c",
    "2UIBwI8LlOkgnR2401030dc085c656433e9d9967c05cb8500",
    "2UIBzkNUiIo3aqf0fcbbefa77c3d721bcc90d6ea330d21b4b",
    "2UIC0txEnUKbs2e2011d4dbfcaccbf586e7cfd303ee25846c",
    "2UIC2RQTla00fnx09c8e8e078bda0be2ee065f87912fcf3ec",
    "2UIC3HmfnANB85ua2fafaa2b7d15fcddfaf43257ea8207a86",
    "2UIC5oOQStd9GOdd78704a1c13ede87f1ad076b3a3c5c014a",
    "2UIC6fQE3KZWxxF95f4c1b1514c6dd3d62ba0670368dbbdf0",
    "2UIC8HXKajhflGK4f6a4fc65b90703c46867dc5868233557d",
    "2UIC9N5NnxkvkiXc269dcbc7d2611f06b19dd6ac170a0e6a4",
    "2UICByRoMWLCFQP85171e81920c71c994e70f565ea94a5af9",
    "2UICCligGnceGaqb0567585836c440c4d21449a570494dfa6",
    "2UICEY4jAqkhpY0f3ecd736fb3d2b1df0f72a5ee544acf341",
    "2UICFz5KhinMtoGa87a2e4a5e156bb3e991297a8f794509c0",
    "2UICIQvD2zirSr161b5959fe434bec1ebe8e5ba0c62a03892",
    "2UICJ88uL7vxQXI13806d1cc2aab512c879ea4b47488aff01",
    "2UICLD7cUOCd06oe31be2d953915e565572bfc9990c96074b",
    "2UICM5P6tkSm3Qv2adc61218a5a7d6ea2f680320cd4db32ea",
    "2UICOGF3whhFISb5a4d943b2f658a0948de3321458f644f73",
    "2UICPYnut7CE37off5de03b2042b14aae1e1c8916eec85f6a",
    "2UICRMpGaWJQKP954bdcecee3ff7068055ac6c06af038c9e1",
]

dataset = None
classes_fast = None
faiss_index = None
vector_dim = 33

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==================== LOGIN BQL (solo per ottenere i cookie) ====================
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
    url = f"{BROWSERLESS_URL}?token={api_key}"
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
        log(f"   ❌ Errore: {e}")
        return None

def login_and_get_cookies(api_key):
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
    if 'user_id' in final_cookies and 'sesids' in final_cookies:
        log(f"   ✅ Login OK! user_id={final_cookies['user_id']}, sesids={final_cookies['sesids']}")
        # Restituisci la stringa cookie nel formato "sesids=...; user_id=..."
        cookie_str = f"sesids={final_cookies['sesids']}; user_id={final_cookies['user_id']}"
        return cookie_str
    else:
        log(f"   ❌ Cookie mancanti: user_id={final_cookies.get('user_id')}, sesids={final_cookies.get('sesids')}")
        return None

# ==================== DATASET ====================
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

# ==================== FUNZIONI DI RICONOSCIMENTO (stile Divellaeasy) ====================
def centra_figura(img):
    if img is None or img.size == 0:
        return cv2.resize(np.full((DIM, DIM, 3), 255, dtype=np.uint8), (DIM, DIM))
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        return cv2.resize(img, (DIM, DIM))
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return cv2.resize(img, (DIM, DIM))
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    crop = img[y:y+h, x:x+w]
    if crop is None or crop.size == 0:
        return cv2.resize(img, (DIM, DIM))
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
    img = centra_figura(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circularity = 0.0
    aspect_ratio = 0.0
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri > 0:
            circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        if h != 0:
            aspect_ratio = float(w) / float(h)
    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    radiale = []
    for r in [int(min(h, w) * v) for v in (0.2, 0.4, 0.6, 0.8)]:
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m / 255.0 for m in mean])
    spaziale = []
    quadranti = [(0,0,cx,cy),(cx,0,w,cy),(0,cy,cx,h),(cx,cy,w,h)]
    for (x1,y1,x2,y2) in quadranti:
        roi = img[y1:y2, x1:x2]
        if roi is None or roi.size == 0:
            spaziale.extend([0.0, 0.0, 0.0])
        else:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m / 255.0 for m in mean])
    vec = np.array(radiale + spaziale + [circularity, aspect_ratio] + hu, dtype=float)
    return vec

def predict(img_crop):
    if img_crop is None or img_crop.size == 0:
        return None
    features = estrai_descrittori(img_crop).astype(np.float32).reshape(1, -1)
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

def fallback_pixel_compare(crops):
    norm = []
    for c in crops:
        if c is None or c.size == 0:
            norm.append(None)
        else:
            centered = centra_figura(c)
            resized = cv2.resize(centered, (DIM, DIM)).astype(np.float32)
            norm.append(resized)
    best = None
    min_diff = float("inf")
    n = len(norm)
    for i in range(n):
        if norm[i] is None:
            continue
        for j in range(i+1, n):
            if norm[j] is None:
                continue
            diff = np.linalg.norm(norm[i].flatten() - norm[j].flatten())
            if diff < min_diff:
                min_diff = diff
                best = (i, j)
    if best and min_diff < 400.0:
        return min(best)
    return None

def riconosci_5(img, picmap):
    labels = []
    crops = []
    for p in picmap:
        cr = crop_safe(img, p.get("coords", ""))
        crops.append(cr)
        lab = predict(cr)
        labels.append(lab if lab is not None else "errore")
    # matching per label
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] != "errore" and labels[i] == labels[j]:
                return labels, i
    # fallback
    idx = fallback_pixel_compare(crops)
    return labels, idx

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

# ==================== SURF LOOP (stile Divellaeasy) ====================
def surf_loop(cookie_str):
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Cookie": cookie_str
    }
    captcha_counter = 0
    consecutive_failures = 0

    while True:
        try:
            r = session.post(
                "https://www.easyhits4u.com/surf/?ajax=1&try=1",
                headers=headers, verify=False, timeout=REQUEST_TIMEOUT
            )
            if r.status_code != 200:
                log(f"⚠️ Status {r.status_code}")
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap", [])

            if not urlid or not qpic or not picmap or len(picmap) < 5:
                log("⚠️ Dati incompleti, attendo...")
                time.sleep(seconds)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                continue

            consecutive_failures = 0

            img_url = f"https://www.easyhits4u.com/simg/{qpic}.jpg"
            rimg = session.get(img_url, verify=False)
            img = cv2.imdecode(np.frombuffer(rimg.content, np.uint8), cv2.IMREAD_COLOR)

            labels, idx = riconosci_5(img, picmap)
            log(f"Labels: {labels}")

            if idx is None:
                log("❌ Riconoscimento fallito → salvo errore")
                salva_errore(qpic, img, picmap, labels, None, "nessun_duplicato", urlid)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            # Invio risposta
            time.sleep(seconds)
            word = picmap[idx]["value"]
            url = (
                f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                f"&ajax=1&word={requests.utils.quote(word)}&screen_width=1024&screen_height=768"
            )
            resp = session.get(url, headers=headers, verify=False)
            resp_json = resp.json()
            if resp_json.get("warning") == "wrong_choice":
                log("❌ Wrong choice")
                salva_errore(qpic, img, picmap, labels, idx, "wrong_choice", urlid)
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    break
                time.sleep(5)
                continue

            captcha_counter += 1
            log(f"✅ OK - indice {idx} - Totale: {captcha_counter}")
            if captcha_counter % 10 == 0:
                gc.collect()
            time.sleep(2)

        except Exception as e:
            log(f"❌ Eccezione: {e}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                break
            time.sleep(10)

    log("🏁 Surf loop terminato")

# ==================== MAIN ====================
def main():
    log("=" * 50)
    log("🚀 LOGIN BQL + AUTOSURF (stile Divellaeasy)")
    log("=" * 50)

    if not load_dataset_hf():
        log("❌ Dataset non caricato")
        return

    for api_key in VALID_KEYS:
        log(f"🔑 Tentativo con chiave: {api_key[:10]}...")
        cookie_str = login_and_get_cookies(api_key)
        if cookie_str:
            log("✅ Login riuscito! Avvio surf loop...")
            surf_loop(cookie_str)
            log("🔄 Surf loop terminato, provo altra chiave...")
        else:
            log(f"   ❌ Login fallito")

if __name__ == "__main__":
    main()
