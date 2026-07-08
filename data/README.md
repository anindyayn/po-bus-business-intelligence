# Manual Analytics Python - Proyek BI PO Bus

Pipeline ini membaca dataset PO Bus, menjalankan analytics manual untuk kebutuhan informasi ADDIE `I1.1` sampai `I3.3`, lalu menghasilkan file siap import ke Power BI.

Kode analytics dipisahkan per stakeholder:

- `src/s1_operasional.py`: Manager Operasional (`I1.1` sampai `I1.3`).
- `src/s2_perawatan_armada.py`: Manager Perawatan Armada (`I2.1` sampai `I2.3`).
- `src/s3_keuangan.py`: Manager Keuangan (`I3.1` sampai `I3.3`).
- `src/manual_analytics.py`: orchestrator untuk membaca data dan menulis Excel/CSV.
- `src/common.py`: helper, metric, fallback dependency, dan formatter workbook.

## Cara Menjalankan

```bash
python3 -m pip install -r requirements.txt
python3 src/manual_analytics.py
```

Jika `scikit-learn` atau `statsmodels` belum terpasang, script tetap berjalan dengan fallback berbasis rule-based dan rolling average. Setelah dependency terpasang, jalankan ulang script untuk mendapatkan model Decision Tree, K-Means, Isolation Forest, dan Holt-Winters.

## Output

- `outputs/analytics_results.xlsx`: workbook ringkasan semua hasil analytics.
- `outputs/csv/*.csv`: sembilan CSV per kebutuhan informasi untuk visualisasi Power BI.
- `data/raw/po_bus_dummy_datasets_rich.xlsx`: salinan dataset sumber agar pipeline reproducible.
