# EasyHits4U Autosurf Bot

Bot per il login automatico e surfing su EasyHits4U, con risoluzione captcha basata su dataset Hugging Face e FAISS.

## Funzionalità

- Login con Browserless BQL e Supabase (chiavi in rotazione)
- Health check server per Render
- Riconoscimento delle figure nei captcha tramite dataset pre-addestrato
- Rilogin automatico in caso di cookie scaduti o errori
- Salvataggio degli errori per migliorare il dataset

## Variabili d'ambiente (Render)

- `SUPABASE_URL`
- `SUPABASE_SERVICE_KEY`

## Deploy su Render

Usa il piano **Standard** per garantire timeout lunghi e RAM sufficiente.