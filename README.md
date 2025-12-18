# Implementasi Transformer dari Nol (From Scratch)

Proyek ini adalah implementasi arsitektur *decoder-only Transformer* (bergaya GPT) dari nol menggunakan NumPy. Implementasi ini dibuat sebagai pemenuhan tugas individu yang berfokus pada pembangunan alur *forward pass* dari model, mulai dari lapisan embedding hingga menghasilkan distribusi probabilitas untuk prediksi token berikutnya.

Seluruh komputasi matematis (operasi matriks, fungsi aktivasi, dll.) dilakukan secara eksklusif dengan NumPy, tanpa bantuan dari library deep learning seperti PyTorch atau TensorFlow.

## Komponen yang Diimplementasikan

Berikut adalah daftar komponen arsitektur Transformer yang telah berhasil diimplementasikan sesuai dengan persyaratan tugas:

- **Token Embedding**: Mengubah token ID menjadi vektor.
- **Positional Encoding**: Menggunakan metode sinusoidal untuk menyisipkan informasi urutan.
- **Causal Masking**: Mencegah model untuk mengakses informasi dari token masa depan.
- **Scaled Dot-Product Attention**: Mekanisme atensi inti dengan normalisasi dan fungsi softmax.
- **Multi-Head Attention**: Menjalankan beberapa mekanisme atensi secara paralel untuk menangkap relasi yang lebih kaya.
- **Feed-Forward Network (FFN)**: Jaringan saraf dua lapis yang diaplikasikan pada setiap posisi token.
- **Residual Connection + Layer Normalization**: Menggunakan arsitektur *pre-norm* untuk stabilitas training.
- **Output Layer**: Melakukan proyeksi akhir ke ukuran kosakata dan menerapkan softmax untuk mendapatkan probabilitas.

## Dependensi

Proyek ini memiliki dependensi minimal dan hanya memerlukan library berikut:
- **NumPy**: Untuk semua operasi komputasi.

Anda dapat menginstalnya menggunakan pip:
```bash
pip install numpy
```

## Cara Menjalankan

Kode program disajikan dalam format Jupyter Notebook (`.ipynb`) agar mudah dijalankan dan diinspeksi secara modular.

1.  **Clone Repositori**:
    ```bash
    git clone https://github.com/aDJi2003/numpy_transformer
    ```
2.  **Pastikan Dependensi Terpasang**: Jalankan perintah `pip install` di atas.
3.  **Buka Notebook**: Buka file `main.ipynb` menggunakan Jupyter Notebook atau Jupyter Lab.
4.  **Jalankan Sel**: Jalankan semua sel kode secara berurutan dari atas ke bawah. Output dari setiap tahap, termasuk pengecekan dimensi tensor dan hasil probabilitas akhir, akan ditampilkan langsung di bawah setiap sel.
