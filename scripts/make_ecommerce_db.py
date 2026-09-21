"""Membuat database studi kasus e-commerce Indonesia (app/ecommerce.db).

Semua data sintetis (Faker locale id_ID), bukan data pribadi nyata.
Jalankan: python scripts/make_ecommerce_db.py
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

SEED = 42
OUT = Path(__file__).resolve().parents[1] / "app" / "ecommerce.db"

KOTA = ["Jakarta", "Bandung", "Surabaya", "Medan", "Semarang", "Yogyakarta", "Makassar", "Denpasar"]
BARANG = {
    "Elektronik": [("Earphone Bluetooth", 250000), ("Power Bank 10000mAh", 180000),
                   ("Smartwatch", 1250000), ("Speaker Portable", 450000),
                   ("Mouse Wireless", 120000), ("Keyboard Mekanik", 750000),
                   ("Monitor 24 Inci", 1850000), ("Webcam HD", 350000), ("Tablet 10 Inci", 3200000)],
    "Fashion": [("Kemeja Batik", 275000), ("Kaos Polos", 65000), ("Celana Jeans", 320000),
                ("Sepatu Sneakers", 550000), ("Tas Ransel", 290000), ("Jaket Hoodie", 210000),
                ("Hijab Voal", 85000), ("Sandal Kulit", 175000), ("Topi Baseball", 70000)],
    "Makanan": [("Kopi Arabika 250g", 95000), ("Keripik Singkong", 25000), ("Sambal Botol", 35000),
                ("Madu Hutan 500ml", 120000), ("Teh Hijau Premium", 60000), ("Rendang Kemasan", 85000),
                ("Kacang Mete 250g", 75000), ("Cokelat Batang", 30000), ("Beras Organik 5kg", 110000)],
    "Kecantikan": [("Serum Wajah", 150000), ("Sunscreen SPF50", 95000), ("Lipstik Matte", 80000),
                   ("Masker Wajah", 45000), ("Parfum 50ml", 350000), ("Toner Wajah", 90000),
                   ("Sabun Muka", 40000), ("Body Lotion", 65000), ("Sampo Herbal", 55000)],
    "Rumah Tangga": [("Panci Set", 425000), ("Rice Cooker", 380000), ("Blender", 460000),
                     ("Setrika Listrik", 210000), ("Sprei Katun", 240000), ("Lampu LED 12W", 45000),
                     ("Rak Piring", 160000), ("Kipas Angin", 330000), ("Dispenser Air", 890000)],
}


def main():
    rnd = random.Random(SEED)
    fake = Faker("id_ID")
    Faker.seed(SEED)
    OUT.unlink(missing_ok=True)
    con = sqlite3.connect(OUT)
    con.executescript("""
    CREATE TABLE pelanggan (id_pelanggan INTEGER PRIMARY KEY, nama TEXT, kota TEXT,
                            no_hp TEXT, email TEXT, tanggal_daftar TEXT);
    CREATE TABLE barang (id_barang INTEGER PRIMARY KEY, nama_barang TEXT, kategori TEXT,
                         harga INTEGER);
    CREATE TABLE transaksi (id_transaksi INTEGER PRIMARY KEY, id_pelanggan INTEGER,
                            id_barang INTEGER, jumlah INTEGER, total INTEGER, tanggal TEXT,
                            FOREIGN KEY (id_pelanggan) REFERENCES pelanggan(id_pelanggan),
                            FOREIGN KEY (id_barang) REFERENCES barang(id_barang));
    """)

    pelanggan = []
    for i in range(1, 151):
        daftar = date(2024, 1, 1) + timedelta(days=rnd.randint(0, 540))
        pelanggan.append((i, fake.name(), rnd.choice(KOTA),
                          "08" + "".join(str(rnd.randint(0, 9)) for _ in range(10)),
                          fake.user_name() + "@" + rnd.choice(["gmail.com", "yahoo.co.id"]),
                          daftar.isoformat()))
    con.executemany("INSERT INTO pelanggan VALUES (?,?,?,?,?,?)", pelanggan)

    barang, bid = [], 1
    for kat, items in BARANG.items():
        for nama, harga in items:
            barang.append((bid, nama, kat, harga))
            bid += 1
    con.executemany("INSERT INTO barang VALUES (?,?,?,?)", barang)

    # 5 barang terakhir sengaja tidak pernah terjual (untuk pertanyaan "belum pernah terjual")
    laku = barang[:-5]
    transaksi = []
    for t in range(1, 601):
        b = rnd.choice(laku)
        jumlah = rnd.randint(1, 5)
        tgl = date(2025, 1, 1) + timedelta(days=rnd.randint(0, 364))
        transaksi.append((t, rnd.randint(1, 150), b[0], jumlah, jumlah * b[3], tgl.isoformat()))
    con.executemany("INSERT INTO transaksi VALUES (?,?,?,?,?,?)", transaksi)
    con.commit()
    con.close()
    print(f"Database dibuat: {OUT}")


if __name__ == "__main__":
    main()
