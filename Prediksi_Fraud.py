import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ==========================================
# --- FASE 1: MEMUAT & MENGECEK DATA ---
# ==========================================
print("--- FASE 1: MEMUAT DATASET FRAUD DETECTION ---")
# Path disesuaikan dengan folder proyek baru Anda
file_path = r'D:\Data Analytics\Credit Card Fraud Project Portofolio\creditcard.csv'
df = pd.read_csv(file_path)

print(f"Dimensi dataset: {df.shape[0]:,} baris dan {df.shape[1]} kolom.")

# ==========================================
# --- FASE 2: EKSPLORASI & PREPROCESSING ---
# ==========================================
print("\n--- FASE 2: VISUALISASI & PENANGANAN DATA TIMPANG ---")

# 1. Visualisasi Ketimpangan Data 
print("Menyiapkan grafik Distribusi Kelas... (Silakan cek popup)")
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x='Class', palette='Set1')
plt.title('Ketimpangan Ekstrem: Normal (0) vs Fraud (1)', fontsize=14, fontweight='bold')
plt.xlabel('Kelas (0: Normal, 1: Penipuan)', fontsize=12)
plt.ylabel('Jumlah Transaksi (Skala Log)', fontsize=12)
plt.yscale('log') # Kita ubah sumbu Y jadi logaritmik karena bedanya ratusan ribu vs ratusan

for p in ax.patches:
    ax.annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
plt.tight_layout()
plt.show()

# 2. Standarisasi Fitur Time dan Amount (Menggunakan RobustScaler)
print("Menerapkan RobustScaler pada kolom Time dan Amount...")
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

# Membuang Time dan Amount asli, dan memasukkan yang sudah di-scale
df.drop(['Time','Amount'], axis=1, inplace=True)
df.insert(0, 'scaled_amount', df.pop('scaled_amount'))
df.insert(1, 'scaled_time', df.pop('scaled_time'))

# 3. Memisahkan Fitur (X) dan Target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# 4. Train-Test Split (WAJIB DILAKUKAN SEBELUM SMOTE!)
# stratify=y memastikan proporsi fraud yang sangat kecil tetap terbagi rata ke data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nData Latih (Train) SEBELUM SMOTE: Normal = {sum(y_train==0):,}, Fraud = {sum(y_train==1):,}")

# 5. Menerapkan SMOTE (Hanya pada Data Latih)
print("Menerapkan SMOTE untuk menyintesis data Fraud tiruan...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Data Latih (Train) SETELAH SMOTE: Normal = {sum(y_train_smote==0):,}, Fraud = {sum(y_train_smote==1):,}")
print("\nFase 2 Selesai! Data latih sudah 100% seimbang dan siap untuk diajarkan ke Machine Learning.")

# ==========================================
# --- FASE 3: PELATIHAN & BENCHMARKING MODEL ---
# ==========================================
print("\n--- FASE 3: MENGADU 6 MODEL DENGAN METRIK FRAUD ---")
print("Harap bersabar, proses ini memakan waktu beberapa menit karena ukuran data yang masif...")

# 1. Mendefinisikan 6 Algoritma Machine Learning
# Parameter disesuaikan agar tidak memakan waktu terlalu lama namun tetap akurat
daftar_model = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42, thread_count=-1)
}

hasil_evaluasi = []

# 2. Looping Pelatihan Model
for nama_model, model in daftar_model.items():
    print(f"-> Sedang melatih & menguji: {nama_model}...")
    
    # PERHATIAN: Mesin BELAJAR dari data yang sudah di-SMOTE (seimbang)
    model.fit(X_train_smote, y_train_smote)
    
    # PERHATIAN: Mesin DIUJI menggunakan data TEST asli (yang tetap timpang/imbalanced)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    # 3. Menghitung Nilai Ujian (Fokus ke PR-AUC dan F1-Score)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_test, y_prob) # Standar industri untuk Fraud Detection
    
    hasil_evaluasi.append({
        "Model": nama_model,
        "Akurasi": round(acc * 100, 2),
        "Presisi": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1-Score": round(f1 * 100, 2),
        "PR-AUC": round(pr_auc * 100, 2)
    })

# 4. Menampilkan Hasil Papan Klasemen
df_hasil = pd.DataFrame(hasil_evaluasi)
# Mengurutkan berdasarkan F1-Score untuk mencari keseimbangan terbaik
df_hasil = df_hasil.sort_values(by="F1-Score", ascending=False).reset_index(drop=True)

print("\n==================================================================")
print("TABEL BENCHMARKING DETEKSI PENIPUAN (DIURUTKAN BERDASARKAN F1-SCORE)")
print("==================================================================")
print(df_hasil.to_string(index=False))
print("==================================================================")

# 5. Visualisasi Hasil Benchmarking
print("\nMenyiapkan grafik perbandingan performa... (Silakan cek popup)")
plt.figure(figsize=(12, 6))

df_melted = df_hasil.melt(id_vars="Model", value_vars=["PR-AUC", "F1-Score", "Recall", "Presisi"], 
                          var_name="Metrik", value_name="Skor (%)")

sns.barplot(data=df_melted, x="Model", y="Skor (%)", hue="Metrik", palette="rocket")

plt.title("Perbandingan Performa 6 Model ML (Credit Card Fraud Detection)", fontsize=14, fontweight='bold')
plt.xlabel("Algoritma Model", fontsize=12)
plt.ylabel("Skor Persentase (%)", fontsize=12)
plt.ylim(0, 110)
plt.xticks(rotation=15, fontsize=10)
plt.legend(title="Metrik Evaluasi", loc='lower right')

plt.tight_layout()
plt.show()

# ==========================================
# --- FASE 4: EVALUASI CONFUSION MATRIX ---
# ==========================================
print("\n--- FASE 4: MEMBEDAH CONFUSION MATRIX ---")
print("Menyiapkan 6 Confusion Matrix... (Silakan cek popup)")

fig_cm, axes_cm = plt.subplots(2, 3, figsize=(15, 10))
axes_cm = axes_cm.flatten()

for i, (nama_model, model) in enumerate(daftar_model.items()):
    # Menggunakan model yang sudah dilatih di Fase 3 untuk menebak X_test
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualisasi Heatmap Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes_cm[i], cbar=False,
                xticklabels=['Normal (0)', 'Fraud (1)'], 
                yticklabels=['Normal (0)', 'Fraud (1)'])
    axes_cm[i].set_title(f'{nama_model}', fontweight='bold')
    axes_cm[i].set_xlabel('') 
    axes_cm[i].set_ylabel('Kenyataan (Aktual)')

plt.suptitle("Confusion Matrix: False Positive vs False Negative", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()