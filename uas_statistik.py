import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Membaca file CSV langsung dari folder yang sama
nama_file = 'tgm 2020-2023_eng.csv'

try:
    df = pd.read_csv(nama_file, delimiter=';')
    print("====================================================")
    print(f"Data '{nama_file}' berhasil dimuat!")
    print(f"Total Jumlah Data: {len(df)} sampel")
    print("====================================================\n")
    
    # Menentukan nama kolom X dan Y
    X_column = 'Daily Reading Duration (in minutes)'
    Y_column = 'Tingkat Kegemaran Membaca (Reading Interest)'
    
    # Membersihkan format angka desimal (koma menjadi titik)
    df[X_column] = df[X_column].astype(str).str.replace(',', '.').astype(float)
    df[Y_column] = df[Y_column].astype(str).str.replace(',', '.').astype(float)
    
    df = df.dropna(subset=[X_column, Y_column])
    X = df[X_column]
    Y = df[Y_column]

    # 1. ANALISIS DESKRIPTIF
    print("--- 1. ANALISIS DESKRIPTIF DATA ---")
    print(f"Rata-rata {X_column}: {X.mean():.2f} (Std Dev: {X.std():.2f})")
    print(f"Rata-rata {Y_column}: {Y.mean():.2f} (Std Dev: {Y.std():.2f})\n")

    # 2. ANALISIS KORELASI & REGRESI
    print("--- 2. HASIL ANALISIS REGRESI & KORELASI ---")
    korelasi = X.corr(Y)
    print(f"Nilai Korelasi (R)          : {korelasi:.3f}")
    
    X_dengan_konstanta = sm.add_constant(X)
    model = sm.OLS(Y, X_dengan_konstanta).fit()
    
    print(f"Nilai R Square              : {model.rsquared:.3f}")
    print(f"Persamaan Regresi Linear   : Y = {model.params['const']:.4f} + {model.params[X_column]:.4f} * X")
    print(f"Nilai Signifikansi (Sig.)  : {model.pvalues[X_column]:.6f}")
    print("----------------------------------------------------")
    
    print("\n--- TABEL RINGKASAN OLS REGRESSION ---")
    print(model.summary())

    # 3. VISUALISASI DATA
    # Grafik Scatter Plot & Garis Regresi
    plt.figure(figsize=(8, 5))
    sns.regplot(x=X, y=Y, data=df, scatter_kws={'alpha':0.6, 'color':'blue'}, line_kws={'color':'red'})
    plt.title('Analisis Regresi: Durasi Membaca vs Tingkat Kegemaran Membaca')
    plt.xlabel('Durasi Membaca Harian dalam Menit (X)')
    plt.ylabel('Tingkat Kegemaran Membaca (Y)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Grafik Histogram X
    plt.figure(figsize=(5, 3.5))
    sns.histplot(X, kde=True, color='skyblue')
    plt.title('Histogram Durasi Membaca (X)')
    plt.tight_layout()

    # Grafik Histogram Y
    plt.figure(figsize=(5, 3.5))
    sns.histplot(Y, kde=True, color='lightgreen')
    plt.title('Histogram Tingkat Kegemaran (Y)')
    plt.tight_layout()
    
    plt.show()
    

except FileNotFoundError:
    print("Eror: File CSV masih tidak ditemukan. Pastikan VS Code sudah membuka folder tempat CSV berada.")
    # ========================================================
# KODE TAMBAHAN: GRAFIK BATANG UNTUK POIN D
# ========================================================

# Hitung rata-rata tingkat kegemaran membaca per tahun
df_tahun = df.groupby('Year')[Y_column].mean().reset_index()

# Membuat Grafik Batang
plt.figure(figsize=(7, 5))
sns.barplot(x='Year', y=Y_column, data=df_tahun, palette='Blues_d')

# Menambahkan judul dan label teks
plt.title('Grafik Batang: Rata-Rata Tingkat Kegemaran Membaca Per Tahun\n', fontsize=12)
plt.xlabel('Tahun', fontsize=10)
plt.ylabel('Rata-Rata Tingkat Kegemaran Membaca', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Menampilkan angka di atas setiap batang grafik
for index, row in df_tahun.iterrows():
    plt.text(index, row[Y_column] + 1, f'{row[Y_column]:.2f}', color='black', ha="center")

plt.tight_layout()
plt.show()