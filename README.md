# PodaNauli — Submission AI Hackathon IT Del 2026

> **PENTING - ATURAN BLIND REVIEW:**
> Peserta **DILARANG KERAS** mencantumkan nama institusi/universitas/sekolah asal di dalam file ini maupun di seluruh _source code_. Pelanggaran terhadap aturan ini dapat berakibat pada pengurangan nilai atau diskualifikasi.

---

## 1. Deskripsi Singkat

PodaNauli menyelesaikan permasalahan data pariwisata Danau Toba yang masih tersebar, tidak seragam, dan sulit digunakan untuk menentukan prioritas peningkatan layanan. Solusi ini mengintegrasikan ulasan, rating, fasilitas, metadata tempat, dan koordinat menjadi data tempat yang konsisten serta dapat dianalisis. Pendekatan yang digunakan meliputi klasifikasi sentimen, deteksi keluhan, klasifikasi aspek _multi-label_, analisis geospasial DBSCAN-Haversine, dan _Service Gap Scoring_ dengan _Bayesian smoothing_. Hasilnya disajikan sebagai peringkat kesenjangan layanan yang dilengkapi bukti dan alasan pendukung, serta tetap memerlukan pemeriksaan manusia.

## 2. Anggota Tim

| Nama Lengkap | Peran dalam Tim | Kontak (Email / GitHub) |
| :--- | :--- | :--- |
| Reyhan Yonathan Batubara | Ketua Tim — Implementasi Sistem dan Pengembangan Model. Bertanggung jawab membangun kode, mengolah data, melatih dan mengevaluasi model, serta mempersiapkan demonstrasi sistem. | reyhanbatubara1@gmail.com |
| Jonathan David Ritonga | Anggota — Riset dan Perancangan Solusi. Bertanggung jawab mengkaji permasalahan, kebutuhan pengguna, konsep solusi, dan pendekatan analisis yang digunakan. | jonathanritonga7@gmail.com |
| Alex Sandro Dabukke | Anggota — Penyusunan Laporan dan Dokumentasi. Bertanggung jawab menyusun laporan analisis, merapikan dokumentasi, dan menyajikan hasil pengembangan secara terstruktur. | alexsidabukke123@gmail.com |

## 3. Pemanfaatan Data Pariwisata Toba

PodaNauli menggunakan **Lake Toba Smart Tourism Knowledge Dataset** yang disediakan oleh panitia.

- **Data Utama:** Data ulasan dan rating tempat wisata, restoran, dan hotel; metadata tempat; kategori; harga; fasilitas; jam operasional; alamat; transportasi; kuliner; serta koordinat lokasi.
- **Pengolahan:** Inspeksi terhadap 14 _sheet_, pemetaan skema, normalisasi nama kolom dan rating, pembersihan teks, penanganan nilai kosong dan formula Excel, deduplikasi, _entity resolution_, integrasi ulasan dengan metadata tempat, anotasi _human-gold_, serta pembagian data _train_, _validation_, dan _locked test_ berdasarkan tempat.
- **Data Tambahan:** OpenStreetMap digunakan sebagai _basemap_ untuk visualisasi sebaran koordinat. OpenStreetMap tidak digunakan sebagai data pelatihan model. Sumber: https://www.openstreetmap.org/copyright

Dataset mentah lengkap dari panitia tidak disimpan di dalam _repository_ publik.

## 4. Tautan & Aset Pendukung

- **Link Video Demonstrasi (Wajib):** https://youtu.be/so1iLgP9ClM
- **Link Pitch Deck / Proposal (Wajib):** https://drive.google.com/drive/folders/1osUZ2CXkWHAoyC5AAkfhmRksNAtA91ZC?usp=sharing
- **Link Deployment / Live App (Opsional):** Belum tersedia. Purwarupa saat ini dijalankan melalui _notebook_ secara lokal dan _offline_.

## 5. Struktur Repository

```text
.
├── configs/                  # Konfigurasi model, taksonomi aspek, dan Service Gap Score
├── data/                     # Data human-gold dan data hasil pemrosesan
├── demo/                     # Notebook, requirements, dan aset demonstrasi
├── models/                   # Model champion dan metadata model
├── notebooks/                # Notebook eksplorasi, pemodelan, dan evaluasi
├── outputs/                  # Metrik, ranking, figur, peta, dan hasil analisis
├── scripts/                  # Script persiapan, validasi, dan pelaksanaan demo
├── src/                      # Implementasi pipeline machine learning
├── tests/                    # Automated tests
├── .env.example              # Contoh konfigurasi environment
├── pytest.ini                # Konfigurasi pengujian
└── README.md                 # Dokumentasi utama project
```

## 6. Prasyarat (Requirements)

Sebelum menjalankan PodaNauli, pastikan perangkat telah memiliki:

- **Environment:** Python 3.11
- **Package manager:** `pip`
- **Virtual environment:** `venv`
- **Software tambahan:** Visual Studio Code dengan ekstensi Python dan Jupyter
- **Database:** Tidak diperlukan
- **Docker:** Tidak diperlukan
- **Layanan cloud:** Tidak diperlukan
- **API key eksternal:** Tidak diperlukan
- **Koneksi internet:** Hanya diperlukan saat mengunduh _repository_ dan memasang _package_

## 7. Environment Variables

PodaNauli tidak membutuhkan _environment variable_, kredensial, atau API key untuk menjalankan evaluasi dan demonstrasi secara lokal.

File `.env.example` tetap disediakan dengan isi berikut:

```env
# PodaNauli tidak membutuhkan API key atau kredensial eksternal.
# File ini disediakan untuk menegaskan bahwa tidak ada konfigurasi rahasia.
```

Juri tidak perlu mengubah atau mengisi variabel apa pun pada file tersebut.

## 8. Langkah Instalasi & Menjalankan Project Secara Lokal

```bash
# 1. Clone repository
git clone <URL_REPOSITORY_DARI_HALAMAN_SUBMISSION>
cd PodaNauli

# 2. Buat virtual environment
python -m venv .venv
```

### Windows PowerShell

```powershell
# 3. Aktifkan virtual environment
.\.venv\Scripts\Activate.ps1
```

Apabila aktivasi ditolak oleh PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
# 3. Aktifkan virtual environment
source .venv/bin/activate
```

### Instalasi dan Pengujian

```bash
# 4. Perbarui pip
python -m pip install --upgrade pip

# 5. Install dependencies demo
pip install -r demo/requirements-demo.txt

# 6. Jalankan automated tests
python -m pytest -q

# 7. Validasi kesiapan demo
python scripts/validate_video_demo.py
```

Apabila validator menampilkan:

```text
Status demo: READY
Pemeriksaan: 12/12 lulus
```

buka _notebook_ berikut melalui Visual Studio Code:

```text
demo/01_podanauli_video_demo.ipynb
```

Pilih kernel Python dari `.venv`, kemudian jalankan **Restart Kernel and Run All**.

## 9. Cara Menggunakan / Testing (Evaluasi Model)

PodaNauli belum menyediakan alamat web lokal karena purwarupa dijalankan melalui _notebook_ evaluasi.

Urutan penggunaan:

1. Buka `demo/01_podanauli_video_demo.ipynb`.
2. Pilih kernel `.venv`.
3. Jalankan seluruh sel menggunakan **Restart Kernel and Run All**.
4. Periksa ringkasan karakteristik dan pengolahan dataset.
5. Pastikan model sentimen, model keluhan, dan model aspek berhasil dimuat.
6. Jalankan contoh inferensi ulasan.
7. Periksa hasil evaluasi _locked test_.
8. Periksa _Service Gap Ranking_ dan hasil validasi manusia.
9. Periksa bagian _error analysis_ dan keterbatasan model.

Contoh input yang dapat digunakan:

```text
Pemandangannya sangat indah, tetapi toilet kurang bersih dan area parkir sempit.

Pelayanannya ramah, tempatnya nyaman, dan makanan disajikan dengan cepat.

Akses jalan rusak, petunjuk arah kurang jelas, dan kendaraan sulit mencapai lokasi.
```

Keluaran utama meliputi:

- prediksi sentimen positif, netral, atau negatif;
- deteksi ada atau tidaknya keluhan;
- satu atau beberapa aspek layanan;
- skor atau tingkat keyakinan model; serta
- informasi pendukung untuk pembentukan _Service Gap Ranking_.

## 10. Known Issues / Batasan

- _Human-gold_ masih berasal dari satu anotator sehingga _inter-annotator agreement_ belum tersedia.
- Saran model terlihat selama proses anotasi sehingga potensi _confirmation bias_ masih mungkin terjadi.
- Kelas sentimen netral masih menjadi kelas yang paling sulit diprediksi.
- Label aspek `lainnya` memiliki jumlah contoh yang sangat sedikit pada _locked test_.
- Validasi _Service Gap Ranking_ baru dilakukan terhadap 20 hasil teratas dengan _overall validity_ sebesar 0,80.
- _Evidence validity_ sebesar 0,80 berada tepat pada batas minimum penerimaan yang ditetapkan.
- Tidak seluruh tempat memiliki metadata koordinat, fasilitas, harga, dan jam operasional yang lengkap.
- Metadata yang kosong tidak dapat langsung diartikan bahwa suatu fasilitas tidak tersedia.
- Beberapa variasi nama tempat masih memerlukan pemeriksaan manual dalam proses _entity resolution_.
- Purwarupa belum menyediakan API, dashboard, pemantauan _data drift_, dan mekanisme _rollback_ model.
- Hasil evaluasi berlaku pada karakteristik dan distribusi dataset yang digunakan dalam proyek ini.
