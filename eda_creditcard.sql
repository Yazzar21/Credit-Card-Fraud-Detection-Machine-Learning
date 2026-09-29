-- 1. Mengambil sampel data transaksi untuk verifikasi awal
SELECT * 
FROM creditcard 
LIMIT 5;

-- 2. Analisis Distribusi Ketimpangan Kelas (Extreme Class Imbalance)
-- Untuk mengetahui perbandingan transaksi normal vs penipuan secara langsung dari database
SELECT 
    Class, 
    COUNT(*) AS jumlah_transaksi
FROM creditcard
GROUP BY Class;