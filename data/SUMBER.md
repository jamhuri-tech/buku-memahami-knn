# Sumber data

Setiap data yang dipakai buku dicatat di sini: nama, asal, lisensi, dan
bab. Data rekaan (delapan titik rekaan, data gelombang, model satu
dimensi Bab 8, data bangkitan lain) dibangkitkan oleh kode bab dengan
benih tetap 20260929 dan tidak disimpan di sini. Tidak ada data yang
disimpan di repositori ini: semua data nyata dimuat atau diunduh oleh
scikit-learn.

| Data | Asal | Lisensi | Bab |
|---|---|---|---|
| Angka tulisan tangan, `load_digits` (1.797 citra 8 x 8) | Ikut terpasang bersama scikit-learn; salinan data uji *Optical Recognition of Handwritten Digits* UCI (Alpaydin & Kaynak 1998), DOI 10.24432/C50P49 | CC BY 4.0 (UCI) | 1, 4, 5, 10, 14-18 |
| Anggur, `load_wine` (178 x 13) | Ikut terpasang bersama scikit-learn; *Wine* UCI (Aeberhard & Forina 1991), DOI 10.24432/C5PC7J | CC BY 4.0 (UCI) | 6, 17, 18 |
| Diabetes, `load_diabetes` (442 x 10) | Ikut terpasang bersama scikit-learn; Efron, Hastie, Johnstone & Tibshirani (2004) | disebarkan bersama scikit-learn (BSD-3); **periksa** | 12 |
| Harga rumah California, `fetch_california_housing` (20.640 x 8) | Diunduh scikit-learn dari figshare ke `~/scikit_learn_data`; asal StatLib, Pace & Barry (1997) | **periksa** (StatLib, tanpa lisensi eksplisit) | 12, 19 |

## Catatan lisensi

- Kode buku berlisensi 0BSD. Data di atas **bukan** bagian dari lisensi
  itu.
- Lisensi diabetes dan California perlu dipastikan sebelum terbit
  (lihat README buku, "Yang perlu diputuskan").
