# Tugas Praktikum K-Means dan Hierarchical Clustering

## Tugas - Clustering dengan K-Means dan Hierarchical Clustering

**Dataset:** `Live.csv` ([Facebook Live Sellers in Thailand - UCI](https://archive.ics.uci.edu/dataset/488/facebook+live+sellers+in+thailand))

**File notebook:** `NRP_KMeans_HC.ipynb`

### Deskripsi Dataset

Dataset berisi 7.050 posting Facebook dari 10 penjual fashion dan kosmetik di Thailand, dikumpulkan melalui Facebook Graph API pada periode 2012 sampai 2018. Setiap baris adalah satu posting beserta jumlah interaksinya (like, komentar, share, dan reaksi emoji). Data bersifat mentah dan tidak dilengkapi label.

Catatan: reaksi emoji baru tersedia di Facebook sejak tahun 2016, sehingga posting sebelum tahun tersebut bernilai nol pada kolom emoji. Tipe kolom pada halaman UCI ada yang keliru; kolom `num_...` seluruhnya adalah hasil hitungan (bukan biner).

### Fitur

| Kolom | Tipe | Keterangan |
|-------|------|------------|
| `status_id` | Numerical | ID data |
| `status_type` | Categorical | Tipe posting (photo, video, status, link) |
| `status_published` | Datetime | Waktu posting (format `m/d/yyyy h:mm`) |
| `num_reactions` | Numerical | Total reaksi pada posting |
| `num_comments` | Numerical | Jumlah komentar |
| `num_shares` | Numerical | Jumlah share |
| `num_likes` | Numerical | Jumlah like |
| `num_loves` | Numerical | Jumlah reaksi love |
| `num_wows` | Numerical | Jumlah reaksi wow |
| `num_hahas` | Numerical | Jumlah reaksi haha |
| `num_sads` | Numerical | Jumlah reaksi sad |
| `num_angrys` | Numerical | Jumlah reaksi angry |
| `Column1` - `Column4` | - | Kolom kosong |

Pada tugas ini kalian akan membangun model clustering (unsupervised learning) untuk mengelompokkan posting Facebook penjual berdasarkan pola engagement-nya. Fokus utama tugas ini adalah pemahaman cara kerja K-Means dan Hierarchical Clustering, khususnya:

- Penentuan jumlah cluster yang optimal (Elbow Method dan Silhouette Score) serta pengaruh transformasi log dan feature scaling
- Perbandingan metode linkage pada Hierarchical Clustering (ward, complete, average, single) dan pembacaan dendrogram
- Perbandingan hasil antara K-Means dan Hierarchical Clustering

Pada tugas ini kalian akan membangun dan membandingkan:
- K-Means Clustering dengan beberapa nilai k
- Hierarchical Clustering (Agglomerative) dengan beberapa jenis linkage
- Hasil kedua metode menggunakan metrik evaluasi internal (Silhouette, Davies-Bouldin, Calinski-Harabasz) serta visualisasi PCA 2D

Catatan: kolom `status_type` **tidak boleh dipakai sebagai fitur clustering**. Kolom ini hanya dipakai untuk membantu menafsirkan cluster pada tahap evaluasi.

Ikuti instruksi pada setiap cell notebook. Baca baik-baik keterangan pada setiap section sebelum menulis kode.

## Ketentuan Pengerjaan

- Kerjakan secara mandiri.
- Isi bagian analisis dan kesimpulan berdasarkan hasil eksperimen kalian sendiri, bukan hasil orang lain.
- Ganti nama dan NRP pada tabel di bagian atas notebook.
- Kumpulkan file notebook dengan format penamaan: `NRP_KMeans_HC.ipynb`
