# Kode — Memahami K-Nearest Neighbors

Repositori pendamping buku **_Memahami K-Nearest Neighbors: Belajar dari
Tetangga, Jarak demi Jarak_** (Edisi Pertama, 2026) oleh Mohammad
Jamhuri, seri *Memahami*.

Berisi seluruh kode Python yang dipakai buku, per bab, beserta
pembangkit gambarnya. **Setiap angka keluaran yang tercetak di buku
dihasilkan oleh kode di sini**, dan `periksa.py` membuktikannya: skrip
itu menjalankan ulang kode setiap bab dan mencocokkan hasilnya dengan
blok keluaran yang tercetak di buku. Setiap hitungan tangan di kotak
*Contoh Soal* juga diperiksa, oleh `kode/babNN_contoh.py`.

Seluruh kode boleh dipakai, disalin, diubah, dan disebarluaskan secara
bebas untuk keperluan apa pun, termasuk komersial, tanpa kewajiban
mencantumkan sumber (lisensi [0BSD](LICENSE)).

Repositori ini tidak menyimpan data pihak lain. Data angka tulisan
tangan, anggur, dan diabetes ikut terpasang bersama scikit-learn; data
harga rumah California diunduh scikit-learn ketika kode Bab 12 dan 19
pertama kali dijalankan. Setiap data tetap mengikuti lisensi aslinya;
asal dan lisensinya dicatat di [`data/SUMBER.md`](data/SUMBER.md).

## Menjalankan

```bash
git clone https://github.com/jamhuri-tech/buku-memahami-knn.git
cd buku-memahami-knn
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python kode/bab04_tangan.py
```

Setiap skrip dijalankan dari akar repositori. Pembangkit bilangan acak
selalu memakai benih tetap (20260929), sehingga keluarannya sama setiap
kali dijalankan.

## Memeriksa angka di buku

```bash
.venv/bin/python periksa.py        # semua bab
.venv/bin/python periksa.py 08     # Bab 8 saja
```

Keluaran `SEMUA COCOK` berarti setiap blok keluaran di buku dihasilkan
ulang oleh kode ini. `kode/bab14_waktu.py` mengukur waktu dan hasilnya
bergantung pada komputer; skrip itu dilewati.

## Struktur

```
kode/
  bab04_*.py         kode Bab 4, dan seterusnya per bab
  babNN_contoh.py    pemeriksa hitungan tangan Contoh Soal Bab NN
data/SUMBER.md       asal dan lisensi setiap data
keluaran/babNN.txt   blok keluaran yang tercetak di Bab NN
gen_gambar.py        membangkitkan semua gambar buku ke gbr/
periksa.py           mencocokkan kode dengan keluaran/
requirements.txt     versi pustaka yang dipakai buku
```

## Versi

Tag `edisi-1` menandai kode yang tepat dipakai untuk mencetak Edisi
Pertama. `requirements.txt` mematok versi yang dipakai buku (Python
3.13.5, NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, Matplotlib
3.10.0). Versi lain biasanya berjalan, tetapi digit terakhir sebagian
angka dapat berbeda.

## Salah ketik, galat, dan saran

Silakan buka [issue](../../issues) di repositori ini: sebutkan bab,
halaman, dan apa yang keliru. Saran juga dapat dikirim ke
m.jamhuri@live.com.
