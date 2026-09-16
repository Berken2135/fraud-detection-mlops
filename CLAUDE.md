# fraud-detection-mlops

## Proje Özeti
Gerçek zamanlı kredi kartı sahtekârlık tespiti için portföy amaçlı, üretim
kalitesinde bir MLOps projesi. Model kendisi basit (XGBoost/LightGBM); asıl
odak etrafındaki altyapı: deney takibi, model registry, API servisi, drift
izleme, otomatik yeniden eğitim, CI/CD.

Veri seti: Kaggle "Credit Card Fraud Detection" (`data/creditcard.csv`,
~285K işlem, sahtekârlık oranı ~%0.17). Dosya git'e eklenmez.

## Teknoloji Yığını
Python 3.11, pandas, scikit-learn, XGBoost, MLflow, FastAPI + Pydantic,
PostgreSQL, Docker + Docker Compose, Prefect, Evidently AI, pytest, ruff,
GitHub Actions.

## Klasör Yapısı
```
data/              creditcard.csv (git'e eklenmez)
notebooks/         EDA
src/               config.py, data.py, features.py, train.py, evaluate.py
api/               main.py, schemas.py
pipelines/         retrain_flow.py
monitoring/        drift_check.py, simulate_traffic.py
tests/
.github/workflows/
```

## Dil Kuralı
Kod, yorumlar, docstring'ler, print/log/plot metinleri, notebook içeriği ve
commit/PR metinleri **İngilizce**. Bu konuşma (CLAUDE.md dahil) Türkçe devam
ediyor. (Aşama 1 notebook'u başta Türkçeydi, sonradan İngilizce'ye çevrilip
yeniden çalıştırıldı.)

## Çalışma Kuralları (proje boyunca geçerli)
- Aşamalar sırayla yapılır; her aşama sonunda durulur, Türkçe özet verilir,
  onay beklenir.
- Dengesiz veri: accuracy KULLANILMAZ. PR-AUC, recall, precision, F1 esas
  alınır.
- Data leakage yok: scaling/SMOTE sadece train verisine fit edilir.
- Sabit seed (RANDOM_STATE=42) ile tekrarlanabilirlik.
- Sabitler ve yollar `src/config.py` içinde tutulur, kod içine gömülmez.
- Tip ipuçları ve kısa docstring'ler kullanılır; gereksiz yorum yazılmaz.
- Yazılan kod çalıştırılıp doğrulanmadan aşama bitmiş sayılmaz.
- Emin olunmayan / kullanıcı kararı gereken konularda tahmin yürütülmez, sorulur.

## Aşama Durumu
- [x] Aşama 0: İskelet — klasör yapısı, requirements.txt, CLAUDE.md, venv, veri kontrolü.
- [x] Aşama 1: EDA ve baseline model (`notebooks/01_eda_baseline.ipynb`, çalıştırılıp doğrulandı).
  - Seçilen model: XGBoost + `scale_pos_weight` (class_weight yaklaşımı), SMOTE'dan daha iyi PR-AUC.
  - Test PR-AUC: 0.8364, optimize eşikte (0.922) precision=0.937, recall=0.797, F1=0.861.
  - RANDOM_STATE=42 sabit; split: %70 train / %15 val / %15 test, stratified.
- [x] Aşama 2: src/ modülerleştirme + MLflow deney takibi + Model Registry + testler.
  - `src/config.py, data.py, features.py, evaluate.py, train.py` — sabitler config'de, sızıntı yok (scaler Pipeline içinde, sadece train'e fit).
  - `python -m src.train`: iki adayı (LogisticRegression, XGBoost) MLflow'a loglar (param/metric/confusion-matrix/PR-eğrisi/model+signature), val PR-AUC'a göre en iyisini seçer, val'de F1-optimal eşiği bulur, test'te tek seferlik değerlendirir, Model Registry'ye kaydeder ve mevcut "production" alias'ından daha iyiyse (PR-AUC) otomatik promote eder (`promote_if_better`, stage 4'te retrain flow tarafından da yeniden kullanılacak).
  - `pytest tests/` (8 test, sentetik veriyle — CSV'ye bağımlı değil, CI'da çalışır) ve `ruff check` temiz.
  - Gerçek veri ile çalıştırıldı: xgboost val PR-AUC=0.833, test (threshold=0.464) PR-AUC=0.840, precision=0.906, recall=0.784, F1=0.841 → "production" alias'ına atandı (version 1).
- [x] Aşama 3: FastAPI servisi + PostgreSQL loglama + Docker Compose (kod tam, Docker Compose kısmı bu makinede Docker kurulu olmadığı için çalıştırılıp doğrulanamadı — bkz. Notlar).
  - `api/schemas.py, db.py, main.py`. `/health`, `/predict`, `/predict/batch`, `/admin/reload-model`.
  - Model, `models:/fraud-xgboost@production`'dan yüklenir; production yoksa API yine de ayağa kalkar (`model_loaded: false`, `/predict` 503 döner).
  - Her tahmin PostgreSQL'e (`predictions` tablosu, features JSONB) loglanır; DB kapalıyken loglama sessizce uyarı basar, tahmin isteğini düşürmez (init_db de aynı şekilde toleranslı — gerçek DB'siz test sırasında bulundu ve düzeltildi).
  - Gerçek registry modeliyle (Docker'sız, lokal uvicorn) uçtan uca test edildi: gerçek normal/sahte işlem örnekleriyle `/predict`, `/predict/batch`, `/admin/reload-model`, negatif `Amount` validasyonu (422) — hepsi doğru çalıştı.
  - `requirements.txt` runtime/`requirements-dev.txt` (pytest, ruff, jupyter) olarak ikiye ayrıldı — Docker imajı jupyter'ı içermiyor.
  - 15 pytest testi (7 yeni API testi mock model+DB ile) geçti, ruff temiz.
- [x] Aşama 4: Drift izleme (Evidently) + Prefect otomatik yeniden eğitim (Evidently mantığı gerçek veriyle doğrulandı; Postgres'e bağlı uçtan uca akış bu makinede test edilemedi — bkz. Notlar).
  - `monitoring/simulate_traffic.py`: gerçek API'ye `/predict` istekleri gönderir, belli bir noktadan sonra `Amount`'u kasıtlı olarak kaydırır (×8 + 500). Gerçek API'ye karşı küçük ölçekte duman testiyle doğrulandı (veri yükleme, drift enjeksiyonu, HTTP çağrıları, gerçek tahminler — hepsi doğru).
  - `monitoring/drift_check.py`: referans (eğitim verisinden örnek) ile canlı (Postgres `predictions` tablosundan) veriyi Evidently `DataDriftPreset` ile karşılaştırır, HTML rapor üretir. `compute_drift()` gerçek Evidently API'siyle sentetik kaymış/kaymamış verilerle test edildi (3 pytest testi).
  - `pipelines/retrain_flow.py`: Prefect flow — drift oranı `config.DRIFT_SHARE_THRESHOLD` (0.3) üzerindeyse `src.train.main()`'i yeni bir random split ile (yeni veri örneğini simüle etmek için) çağırır; `promote_if_better` zaten `train.main()` içinde çalıştığı için yeniden kullanılıyor, ayrı bir promotion mantığı yazılmadı. Eşik altı/üstü iki dal da mock'lu 2 pytest testiyle doğrulandı.
  - Bu aşamada iki gerçek prod-kalitesi bug bulundu ve düzeltildi: (1) Makefile `drift` hedefi script yolu ile çalıştırıyordu (`python monitoring/simulate_traffic.py`), bu da `from src import config` importunu kırıyordu — `-m` modül biçimine çevrildi. (2) DB kapalıyken her `/predict` isteği ~4 saniye gecikiyordu çünkü `localhost` hem IPv6 hem IPv4 için sırayla deneniyordu; `DATABASE_URL` varsayılanı `127.0.0.1`'e çevrildi ve `connect_timeout=1` eklendi (artık ~1sn).
  - 20 pytest testi (5 yeni: 3 drift + 2 retrain-flow) geçti, ruff temiz.
- [x] Aşama 5: CI/CD (GitHub Actions) + README dokümantasyonu.
  - `.github/workflows/ci.yml`: her push/PR'da `ruff check` + `pytest` + Docker image build (ayrı job, `docker/build-push-action`, `push: false`). YAML PyYAML ile parse edilerek sözdizimi doğrulandı (not: `on:` anahtarı PyYAML'de YAML 1.1 kuralı gereği boolean'a çevriliyor ama bu GitHub Actions'ın kendi parser'ını etkilemiyor, bilinen zararsız bir durum).
  - `README.md` (İngilizce): proje amacı, Mermaid mimari diyagramı, kurulum (yerel + Docker Compose), model sonuç tablosu, drift izleme açıklaması, tasarım kararları, bilinen sınırlamalar, gelecek geliştirmeler.
  - Mermaid diyagramı `@mermaid-js/mermaid-cli` (npx) ile gerçekten render edilip geçerli bir SVG ürettiği doğrulandı.
  - CI'daki Docker build job'ı bu makinede (Docker yok) çalıştırılıp doğrulanamadı — ilk push'ta GitHub Actions üzerinde gerçek sınavını verecek.
- [x] Aşama 5.1: README'yi görsel/profesyonel hale getirme (kullanıcı isteği üzerine ek geçiş).
  - `docs/images/`: notebook'un gömülü grafiklerinden (class distribution, amount-by-class, hourly fraud, feature correlation, PR curve comparison, threshold optimization) ve gerçek eğitim çıktısından (confusion matrix) çıkarılan 7 PNG + gerçek bir Evidently drift raporunun Playwright ile alınan ekran görüntüsü (`drift_report.png`) — hepsi gerçek/üretilmiş veriden, uydurma değil.
  - `src/evaluate.py::plot_confusion_matrix`: log-scale renklendirme + colorbar eklendi (gerçek bug/eksiklik: dengesiz veri confusion matrix'inde tek hücre görseli domine ediyordu, diğer 3 hücre ayırt edilemiyordu). `python -m src.train` yeniden çalıştırılarak gerçek veriyle doğrulandı.
  - README'ye İçindekiler, Tech Stack tablosu, Project Structure ağacı, API Reference (uç nokta tablosu + örnek istek/yanıt), "Exploratory Data Analysis" ve "Model Results" bölümlerine görsel galeriler, drift raporu ekran görüntüsü eklendi.
  - Tüm görsel referansları (`docs/images/*.png`) dosya varlığı ve `.gitignore` durumu için tek tek doğrulandı — hiçbiri yanlışlıkla ignore edilmiyor.
  - ruff + pytest (20 test) tekrar çalıştırıldı, hepsi geçti.

## Notlar
- Veri seti zaten `data/creditcard.csv` içinde mevcut (kullanıcı tarafından sağlandı).
- Sanal ortam: `.venv` (proje kökünde).
- Bu geliştirme makinesinde **Docker, `make` ve PostgreSQL kurulu değil** (gömülü bir Postgres denendi — `pgserver` paketi — ama Windows'un Türkçe sistem yereli `initdb`'yi çökertiyor, bu yüzden vazgeçildi). `Dockerfile`/`docker-compose.yml` özenle yazıldı ve mantığı gözden geçirildi, ancak `docker compose up` ve Postgres'e bağlı akışlar (API'nin DB'ye loglaması, `drift_check.py`'nin `predictions` tablosunu okuması) hiç gerçek DB ile uçtan uca çalıştırılamadı — kullanıcı kendi makinesinde ilk çalıştırmada dikkatli olmalı, sorun çıkarsa bildirsin. Makefile hedefleri ve script mantığı, Docker/DB olmadan (gerçek MLflow modeliyle, mock/gerçek-ama-kapalı DB ile) tek tek doğrulandı.
