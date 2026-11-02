"""Các loại cửa hàng cho dữ liệu tự sinh: cố ý **khác** 5 loại của train (tạp hoá, cà phê, vật liệu xây dựng, nhà thuốc,
quán ăn) để giả lập cửa hàng chưa gặp ở private. Mỗi mặt hàng: (tên, nhóm, đơn vị cơ bản, giá bán một đơn vị cơ bản,
{đơn vị lớn: hệ số}, độ phổ biến tương đối)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Archetype:
    key: str
    label: str  # câu mô tả đầu README
    items: list[tuple[str, str, str, int, dict, float]]
    orders_per_day: float
    channels: list[str] = field(default_factory=lambda: ["Tại quầy"])
    payments: list[str] = field(default_factory=lambda: ["Tiền mặt", "Chuyển khoản"])
    known_customer_rate: float = 0.5  # tỉ lệ đơn có ghi khách
    debt: bool = False  # có bán ghi nợ
    imports: bool = True  # có sổ nhập hàng
    open_hours: tuple[int, int] = (7, 21)


ARCHETYPES = [
    Archetype("nha_sach", "Nhà sách - văn phòng phẩm.", [
        ("Bút bi Thiên Long TL-027", "Bút viết", "cây", 5_000, {"hộp": 20}, 8),
        ("Bút chì 2B Staedtler", "Bút viết", "cây", 7_000, {"hộp": 12}, 4),
        ("Vở Campus 200 trang", "Vở - giấy", "quyển", 18_000, {"lốc": 10}, 7),
        ("Giấy A4 Double A 70gsm", "Vở - giấy", "ram", 82_000, {"thùng": 5}, 3),
        ("Băng keo trong 5cm", "Dụng cụ", "cuộn", 12_000, {"cây": 6}, 3),
        ("Thước kẻ nhựa 30cm", "Dụng cụ", "cái", 6_000, {}, 3),
        ("Bìa còng 7cm Kingstar", "Dụng cụ", "cái", 45_000, {"thùng": 20}, 2),
        ("Sách Đắc nhân tâm", "Sách", "cuốn", 86_000, {}, 2),
        ("Sách Nhà giả kim", "Sách", "cuốn", 79_000, {}, 2),
        ("Máy tính Casio FX-580VN X", "Dụng cụ", "cái", 690_000, {}, 1),
    ], 22, channels=["Tại quầy", "Shopee"], known_customer_rate=0.35),
    Archetype("my_pham", "Cửa hàng mỹ phẩm.", [
        ("Sữa rửa mặt Cetaphil 125ml", "Làm sạch", "chai", 215_000, {}, 6),
        ("Nước tẩy trang Bioderma 500ml", "Làm sạch", "chai", 389_000, {}, 4),
        ("Kem chống nắng Anessa 60ml", "Chống nắng", "tuýp", 465_000, {}, 5),
        ("Son Black Rouge A12", "Trang điểm", "cây", 175_000, {}, 6),
        ("Phấn phủ Innisfree No Sebum", "Trang điểm", "hộp", 155_000, {}, 3),
        ("Mặt nạ giấy Innisfree trà xanh", "Dưỡng da", "miếng", 25_000, {"hộp": 10}, 7),
        ("Serum The Ordinary Niacinamide 30ml", "Dưỡng da", "chai", 290_000, {}, 4),
        ("Dầu gội Tsubaki 490ml", "Tóc", "chai", 198_000, {"thùng": 12}, 3),
    ], 14, channels=["Tại quầy", "Shopee", "TikTok Shop"], payments=["Tiền mặt", "Chuyển khoản", "COD"],
        known_customer_rate=0.7),
    Archetype("thu_cung", "Cửa hàng thú cưng.", [
        ("Hạt Royal Canin Mini Adult 2kg", "Thức ăn", "túi", 420_000, {"thùng": 6}, 5),
        ("Pate Whiskas cá ngừ 85g", "Thức ăn", "gói", 14_000, {"hộp": 12}, 8),
        ("Xương gặm Pedigree Dentastix", "Thức ăn", "gói", 52_000, {}, 4),
        ("Cát vệ sinh mèo Nhật 10L", "Vệ sinh", "bao", 115_000, {}, 5),
        ("Sữa tắm chó mèo SOS 530ml", "Vệ sinh", "chai", 135_000, {}, 3),
        ("Vòng cổ chó size M", "Phụ kiện", "cái", 65_000, {}, 2),
        ("Nhà cây cho mèo 3 tầng", "Phụ kiện", "cái", 890_000, {}, 1),
    ], 12, known_customer_rate=0.75, debt=True),
    Archetype("tiem_banh", "Tiệm bánh ngọt.", [
        ("Bánh bông lan trứng muối", "Bánh ngọt", "hộp", 65_000, {}, 7),
        ("Su kem", "Bánh ngọt", "cái", 12_000, {"hộp": 6}, 8),
        ("Bánh tart trứng", "Bánh ngọt", "cái", 18_000, {"hộp": 4}, 6),
        ("Bánh mì hoa cúc", "Bánh mì", "ổ", 95_000, {}, 4),
        ("Bánh kem bắp 16cm", "Bánh kem", "cái", 230_000, {}, 2),
        ("Bánh kem sinh nhật 20cm", "Bánh kem", "cái", 380_000, {}, 2),
        ("Cookie bơ 250g", "Bánh khô", "hũ", 85_000, {}, 3),
    ], 35, channels=["Tại quầy", "GrabFood"], known_customer_rate=0.2, imports=False, open_hours=(6, 22)),
    Archetype("dien_thoai", "Cửa hàng phụ kiện điện thoại.", [
        ("Ốp lưng iPhone 15 silicon", "Ốp - kính", "cái", 90_000, {}, 6),
        ("Kính cường lực 9H", "Ốp - kính", "miếng", 50_000, {"hộp": 10}, 7),
        ("Cáp sạc Type-C Anker 1m", "Sạc - cáp", "sợi", 150_000, {}, 5),
        ("Củ sạc nhanh Baseus 20W", "Sạc - cáp", "cái", 220_000, {}, 4),
        ("Tai nghe Bluetooth QCY T13", "Âm thanh", "cái", 390_000, {}, 3),
        ("Sạc dự phòng Xiaomi 10000mAh", "Sạc - cáp", "cái", 450_000, {}, 2),
        ("Giá đỡ điện thoại để bàn", "Phụ kiện khác", "cái", 60_000, {}, 2),
    ], 16, channels=["Tại quầy", "Shopee", "Lazada"], payments=["Tiền mặt", "Chuyển khoản", "COD"],
        known_customer_rate=0.4),
    Archetype("vat_tu_nong_nghiep", "Cửa hàng vật tư nông nghiệp.", [
        ("Phân NPK 16-16-8 Đầu Trâu", "Phân bón", "kg", 19_000, {"bao": 50}, 7),
        ("Phân Urê Phú Mỹ", "Phân bón", "kg", 13_500, {"bao": 50}, 7),
        ("Vôi bột nông nghiệp", "Cải tạo đất", "kg", 3_500, {"bao": 25}, 3),
        ("Thuốc trừ sâu Regent 800WG", "Thuốc BVTV", "gói", 18_000, {"hộp": 20}, 5),
        ("Thuốc diệt cỏ Gramoxone 1L", "Thuốc BVTV", "chai", 145_000, {"thùng": 12}, 3),
        ("Hạt giống rau muống", "Hạt giống", "gói", 10_000, {}, 4),
        ("Lưới che nắng 70%", "Dụng cụ", "mét", 9_000, {"cuộn": 100}, 2),
    ], 15, known_customer_rate=0.85, debt=True),
    Archetype("thoi_trang", "Shop quần áo.", [
        ("Áo thun cotton basic", "Áo", "cái", 150_000, {}, 8),
        ("Áo sơ mi nam trắng", "Áo", "cái", 290_000, {}, 4),
        ("Quần jean nữ ống suông", "Quần", "cái", 350_000, {}, 5),
        ("Quần short kaki", "Quần", "cái", 220_000, {}, 4),
        ("Váy hoa nhí", "Váy - đầm", "cái", 320_000, {}, 4),
        ("Tất cổ ngắn", "Phụ kiện", "đôi", 15_000, {"lốc": 5}, 5),
    ], 18, channels=["Tại quầy", "Shopee", "TikTok Shop", "Facebook"], payments=["Tiền mặt", "Chuyển khoản", "COD"],
        known_customer_rate=0.6),
    Archetype("phu_tung_xe", "Cửa hàng phụ tùng xe máy.", [
        ("Nhớt Castrol Power1 0.8L", "Dầu nhớt", "chai", 115_000, {"thùng": 24}, 8),
        ("Nhớt Motul 3000 1L", "Dầu nhớt", "chai", 105_000, {"thùng": 20}, 5),
        ("Bugi NGK C7HSA", "Phụ tùng", "cái", 35_000, {"hộp": 10}, 5),
        ("Má phanh Honda Vision", "Phụ tùng", "bộ", 85_000, {}, 4),
        ("Lốp Michelin 80/90-14", "Lốp - săm", "cái", 520_000, {}, 2),
        ("Săm xe máy IRC 2.50-17", "Lốp - săm", "cái", 75_000, {}, 3),
        ("Bóng đèn LED H4", "Điện", "cái", 180_000, {}, 2),
    ], 20, known_customer_rate=0.45, debt=True),
    Archetype("quan_bun", "Quán bún phở.", [
        ("Bún bò Huế", "Món chính", "tô", 55_000, {}, 9),
        ("Phở bò tái", "Món chính", "tô", 60_000, {}, 8),
        ("Hủ tiếu Nam Vang", "Món chính", "tô", 55_000, {}, 5),
        ("Quẩy", "Ăn kèm", "cái", 5_000, {}, 6),
        ("Trà đá", "Đồ uống", "ly", 5_000, {}, 8),
        ("Nước sâm", "Đồ uống", "ly", 15_000, {}, 4),
    ], 60, channels=["Tại quán", "GrabFood", "ShopeeFood"], known_customer_rate=0.05, imports=False,
        open_hours=(6, 21)),
]
