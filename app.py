#!/usr/bin/env python3
# app.py - Login + Autosurf con Browserless BQL (stessa sessione)

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

# ==================== CONFIGURAZIONE ====================
DIM = 64
REQUEST_TIMEOUT = 15
ERRORI_DIR = "/tmp/errori"
MAX_CONSECUTIVE_FAILURES = 5

# Credenziali EasyHits4U
EASYHITS_EMAIL = "sandrominori50+uiszuzoqatr@gmail.com"
EASYHITS_PASSWORD = "DDnmVV45!!"
REFERER_URL = "https://www.easyhits4u.com/?ref=nicolacaporale"

# Browserless endpoint
BROWSERLESS_URL = "https://production-sfo.browserless.io/chrome/bql"

# ==================== CHIAVI VALIDE (187) ====================
VALID_KEYS = [
    # Chiavi del primo test (10)
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
    # Chiavi del secondo test (177)
    "2UJK3J6z8WVUZCnebd8f5f45581cb8e33d54c5f102ff1ca1a",
    "2UJK4Jun2RJGbpmb4744ac717d57e27d86a6f8cdea79ecb29",
    "2UJK6yKb6025jjV0ec93e78221afdd7422cba5e9c2cf215b2",
    "2UJK7NrAnPQHmLj5f59ebaeb40664e36acb5e9edb16258649",
    "2UJK9tJF6fSUI1yee47518cbceb44b754091f65ffb37385e9",
    "2UJKBhoRgHclEJteebf8c7771d9b2ac024e173d5e8c668e63",
    "2UJKIOxgYKTLcPm78093f2ec30b29d3ed2796fd80812e30e4",
    "2UJKKhepmZphDJ5934eae8e34c8cc2166d53c97e18d88842f",
    "2UJKQznXMrCRsDe7e27bb3392684dc84617e99bfebb86c6f3",
    "2UJKSaJ8LbR6yPMa1de82874dc44ebb02c6538905563345db",
    "2UJKT2s7366Q95C9f93fe45d1e69c35b063479f681746371d",
    "2UJKVmGU7EHnhBa7888f73274495565fd975f87911d955624",
    "2UJKWATu0ywwDLj6745bd019eb949bc89ee0bde7b8aefcceb",
    "2UJKYEO1MHAoKLP13c6d573801f1194a2db77382e1c9ca279",
    "2UJKZKYxDW04Fvycd08e1f4373d86ac84939fd2da94b7bb6b",
    "2UJKbrJ8mu81DDId0b0a6d5d6f09d4232e86c95d0508d2286",
    "2UJKctR3YKrr0jNb8125ac5a469ffc015154b0ef2ebdfbd64",
    "2UJKgrHd3wpB8ER8967264f75287a7a37b6c07cd1aa385e8a",
    "2UJKhZ4nnu4dmrJdf0abe5a76fdcaccca3d4bde1c8e756207",
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
    "2UJLTjMfcMlNuaw76c40208b2a61ce7f8fffa3fd8b570f8be",
    "2UG2TlpDxsQJn2Wd1f204756127d4ac2136b41bd01baaa0ca",
    "2UGdbQnmFCJwS9Vd714eb85438cf63d00a8f878a898cfe865",
    "2UGdcalCbtmQNCt0c0a65e134b1833ed5d77b0c27fec4df7a",
    "2UGdeyvPnuYf2tm78f5d97e862f004feef3a8e41dfd58b3ef",
    "2UGdfrLYfztPfpy65ea1648786cdfe855a89073f49a24fa15",
    "2UGdh0XeC72wcccb12714bdae43194a6a8647ce9a836d9cf9",
    "2UGdiXdiszEa5rw5c83ff671b0f30e6b45cb159d1b7a8f221",
    "2UH1q8Mnj1ERdcZf243e8d19a8e05da8998570d64e212cc3a",
    "2UH1rvpwwnyIqKYf3d2b847c23f1bf100eb78217b4abe399e",
    "2UH1tCPjVWSuutr98a6d9529fb8c03b457496afe6466ebac0",
    "2UH1uDTJQKxWMi750e2ad5d114a378275b4f4963b81476824",
    "2UH1xtruDYkpN6qafcf735210a0d390f38b7934fee7020509",
    "2UH1yEsOSdMyVgBb79e5d9f7283da3ab24b099772a221c0c1",
    "2UH200RyjgTPJAyd69e6979481a42076d9715120add383b2f",
    "2UH21NyLelnPOXN89ef213e06c030d3a20fe91f74ed023cd6",
    "2UH23g4Tjer24qYda1b38b3bf4995babae59f6ade1b5d80d5",
    "2UH24rd152tYgA9bfd616f9e0a1eee38c91957e77f7388367",
    "2UH26buZuikxxt088fe658690e962e79f00f03bae1c9c23d3",
    "2UH27IyTT0RHycacd91e7dcd3c026b13a34334e2669771ff3",
    "2UH294cqCAfyXPYa0fb233ea57a4aa4ac1cfa9e767080324b",
    "2UH2AnTc77FXlItd61132c9805d95deacff876085a8673a9f",
    "2UH2CfWXJrCUNeVdd80c7e1b03518bbfdcf651e646f5f87d6",
    "2UH2DCjQeXY976cbc3b9a2f96b6b7c639bce3f82349f4dc3c",
    "2UH2FdGsdqj9zdBd31de95f2d5f8f661cf0cd4980112ce6d5",
    "2UH2GTfPxLjEANac954251257e3745ed64d7eeba896e59569",
    "2UH2IvxBVMIZf7pbc1f54a2696deef605bc9a8b43b5ccc8b8",
    "2UH2JmJbYEUBMQBa05981954be8f4996b345b0f8b3682cc00",
    "2UH2L4xZ5oNQ80w85bf6bc0075e1f1e91f9106ad882b73ad3",
    "2UH2N0WXkIuziiJ071449dfda09a57c174a3271491197bc93",
    "2UH2PXIv41CFGZi83f01bc2ec164655754bffb8a14e6ec8dd",
    "2UH2QtMa2NAHKqgff261a53ca86a8f8281fc78b3d18d61829",
    "2UH2bnJSP3jJh2zcfddf0eaacc03a5a36a586558c9127f6a0",
    "2UH2cb7PyfPpoxBce3f0a9868715cd7026d8e539aac36d402",
    "2UH2eKbGQKuIYUXcad0304cb6e5bee0b0c403afdbb45eb29e",
    "2UH2gUfSx5xbV8v5c1782e505ebd7c097193963887490ccf2",
    "2UH2hBN40tQzuef302dcb8aa91dbe6770856a538edbfb6673",
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
    "2UIuKcJvun4R77F313307382c902416a7b4182b41c60ab655",
    "2UIuLDWh7ajw1fw0fd8db3599f49f5b207c36196b317535d9",
    "2UIuNFMS3ilf9tL483dca9443bd696161cd6494703ad11a9a",
    "2UIuOg7KoH653yJ253de7b452f459e6c1aaa2d3877ebd7ff1",
    "2UIuQ7WsV43MTaWbf7af91b6c2d9439f19383068c67226507",
    "2UIuRxMmqCueYZT245387bdde3541387530ac4a4cdac8210d",
    "2UIuTk4NV9YEZYfec7d8ff7af421dcf1ac9be09634c3592dc",
    "2UIuUGtgLIBTdned851acb58282b709b857b533e24d3ec6bf",
    "2UIuWUpXw9Us1hkcb5496c4a18956f3f2c527b39327a99641",
    "2UIuYDMCnULuk1u51de7efb0404eb2f8347f0cecb2d87d330",
    "2UIuaQmu1XK4W5k354a019269ea140dcc45b088ec477b8c87",
    "2UIubwkm21RIi9Yf891666901a176d8776a02480b6b816550",
    "2UIudm3HYaNK5Qw4cbf85baee95be4de5a49f6e509b4c92f8",
    "2UIue6Hdr8qoK35c4dc442c59b4e92064c5a78799a08dfd5d",
    "2UIugcM1JvlSEwAef3509d5e6de44b18ba099592d1df74c1b",
    "2UIuh7yKIUaHOrs21beaed294d099e3cc3bd4efb1f87f10b8",
    "2UIujiXisPQ7Emd94bd2227dff7c4995b8e6794c5d4b6f3e5",
    "2UIukwWZppRkCOt7bd2c19791353234c49c54b1222a9a04a5",
    "2UIumgz1V68vJQ076cd051c7a0c7cffba4d9f79b770269391",
    "2UIunO9Fy65036E1d8504630b785384380d9ff0bf79288d73",
    "2UJ1tyeQPpoIq5ce393e8aedfc71bf2cd5bde8e12ce0840b8",
    "2UJ1uodfUujigTy9334dd24560921ca34118bd518d88ab3de",
    "2UJ1w5OGJhvYl9K157df6861598fc12388573b068a1a7894e",
    "2UJ1xHDfiYWz8n128afde5f8c9b0da82d7b3e9bdfaddedfb4",
    "2UJ1z8AHhwdXTtFdb03f29009a1cfede61bdfb76c90d22468",
    "2UJ2075CjdEW1XH6d65b6e14eebb068e25f42e44b4d292f7d",
    "2UJ22dGBJPN8Thf90e088d29b5553527a52c6f439a20be5f8",
    "2UJ24oJaxnpnLqd1552d3b745a167815b852ad7ba4178fd9a",
    "2UJ26RXsOPYD7rjd1da226f512e5c907bcf92d7ac515944a9",
    "2UJ278vHM8JseeE61c297cafdc30dea2417a01b7eeeeffe1a",
    "2UJG6mSb6gUajs71e0ce2c1bf27b9124c0078dd957b1039c7",
    "2UJG7rFeP5eQRdyc214fbd4ffac880268cf77ed9560d7e64b",
    "2UJG9cBi9LyX9JU86142822bb538d4c434f7493727c1fc9a2",
    "2UJGA6P4YblOW9y467d57083de65346a16bf49227ca0abbf7",
    "2UJGC0rki9aE8bI0adbae10b4841aa78e3100c5526b414406",
    "2UJGEs1JC6DV9lf786858d925c560f25eba95ac9368a835a6",
    "2UJGF0kV8xvCA6we9bf45b3333644caaa6add1d78e58b85b1",
    "2UJGHu4W4CUMImZ1df04ff2187f78fb7545c63522b4b414fe",
    "2UJGJKE1ASJPNJN659f0cd13adf550e8d80d878a4fafd85fd",
    "2UJGK3ktrDdx3FYf4f20c9a4879709937c1950a8b4d1ac0c1",
    "2UJGOIAVPXTkHlw2ccdfb765a2f4b01bcb2dbcf7dbf766958",
    "2UJGVjrIU6OVeSn5bb607957360491a59735f2353039aa3bd",
    "2UJGWPtfcfzKqUsd746fe1985f668a354cb295018fa4531b8",
    "2UJGYFHMeH8RkZi27e89167260935275ab5351a8f661a119d",
    "2UJGe5yfZXbk4kX31b22cb9bcd8bfb313789dbdc49b72752d",
    "2UJGgo8gFaFDXYrbec1cc82780dc46a0988a60a4d329e4669",
    "2UJGhWT2APrP1QU9f3d0d4d5344af01ca1fcd4c26e79693f6",
    "2UJHO74df5CZDGHd17c448fae4af05ce443e2a01851061bba",
    "2UJHPa1YEomq0aV87aad3d9c8863b45251aeb1cb0013bb95f",
    "2UJHRHsID0KjfgH58d071dd1fc655f7e17817f4cb965f248f",
    "2UJHSvWwT16KyI7bbfe514f8bba50af68e8e65f5546b7cbad",
    "2UJHU9MunyfH2w5a81f4b6ce6cc4b0211d8d2bd25eabf89c2",
    "2UJHV4xirr9OUfCfb7fd332ebf307dab58e9045757d3238f1",
    "2UJHXZ51ciLRpaR097392aee834694007dd220f550474fcdf",
    "2UJHY0rDwsFTpzx33a3fe131585d7e8d106d4c8452e62f836",
    "2UJHaQLhxmGCfK646b323e86adfbfecc39a54493366e449ab",
    "2UJHcZCyfGT4J0Y495e5607927812aaed3f19b5a88b50107d",
    "2UJHdCU6toylb1V95c783eb89b08a1a0e43b264800d715db2",
    "2UJHfJa2twbENtO138ccbf0e5d889417f5951b10edc594fbe",
    "2UJHhzqPQzt3OlU0e035f1bcf1bd57cd44c8b04fafae91faf",
    "2UJHiY0mlSxpOWA44f17e9d3c8fee2d8435a2885953c901ae",
    "2UJHkfkeXu7HPER5c39bfe1e29382e799c4cb36f4826e7865",
    "2UJHl14hTR3QOUVe891605cf6241ef662df2dea7d4814de82",
    "2UJHn2XOtL2tGr0291915b9a098bb64e9189692313d95cde0",
    "2UJHo6cvuKOzvLbe47b1f3460eb0f97f713ad92d864e6e5e4",
    "2UJHqnKVyF35it4b6fee8a27d583ae8d832c03f92e0336a92",
    "2UJHrsNyBBuXKgJ5edd21349175868d8d48f98232b7a75f1d"
]

# ==================== GLOBALS DATASET ====================
dataset = None
classes_fast = None
faiss_index = None
vector_dim = 33

# ==================== LOG ====================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ==================== DATASET HUGGING FACE ====================
def load_dataset_hf():
    global dataset, classes_fast, faiss_index
    log("📥 Caricamento dataset da Hugging Face Hub...")
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
            if i % 50000 == 0:
                log(f"   Aggiunti {len(X_batch)} vettori, totale {index.ntotal}/{total}")
        log(f"✅ Indice FAISS creato con {index.ntotal} vettori")
        faiss_index = index
        gc.collect()
        return True
    except Exception as e:
        log(f"❌ Errore dataset: {e}")
        return False

# ==================== FUNZIONI DI RICONOSCIMENTO ====================
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

# ==================== LOGIN (Browserless BQL) ====================
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
        log(f"   ❌ Errore token: {e}")
        return None

def do_login(api_key):
    """Esegue login e restituisce la sessione requests con i cookie"""
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
        session.headers.update({
            'Referer': 'https://www.easyhits4u.com/surf/',
            'X-Requested-With': 'XMLHttpRequest'
        })
        return session
    else:
        log(f"   ❌ Cookie mancanti: user_id={final_cookies.get('user_id')}, sesids={final_cookies.get('sesids')}")
        return None

# ==================== SURF LOOP ====================
def surf_loop(session):
    consecutive_failures = 0
    captcha_counter = 0

    # Inizializzazione: visita /surf/
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
                log(f"⚠️ Sessione scaduta (redirect a {data['redirect']})")
                break

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

# ==================== MAIN ====================
def main():
    log("=" * 50)
    log("🚀 LOGIN + AUTOSURF CON BROWSERLESS BQL")
    log("=" * 50)
    
    # Carica dataset
    if not load_dataset_hf():
        log("❌ Dataset non caricato, esco")
        return
    
    # Prova ogni chiave
    for api_key in VALID_KEYS:
        log(f"🔑 Tentativo con chiave: {api_key[:10]}...")
        
        session = do_login(api_key)
        if session:
            log("✅ Login riuscito! Avvio surf loop...")
            surf_loop(session)
            # Se arriviamo qui, il surf loop è terminato (sessione scaduta)
            log("🔄 Sessione scaduta, provo altra chiave...")
            continue
        else:
            log(f"❌ Login fallito, passo alla prossima chiave")
    
    log("❌ Login fallito con tutte le chiavi")

if __name__ == "__main__":
    main()
