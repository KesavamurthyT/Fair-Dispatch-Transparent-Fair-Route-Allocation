import json
import qrcode
import os
import zlib
import base64

# ---- Specify your JSON file path here ----
json_file_path = "allocation.json"   # <-- Change this to your file path

# ---- Check if file exists ----
if not os.path.exists(json_file_path):
    print(f"Error: File '{json_file_path}' not found.")
    exit()

# ---- Load JSON from file ----
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Convert JSON to compact string, then compress ----
json_string = json.dumps(data, separators=(',', ':'))
compressed = zlib.compress(json_string.encode('utf-8'), level=9)
encoded = base64.b64encode(compressed).decode('ascii')

print(f"Original size : {len(json_string)} bytes")
print(f"Compressed+B64: {len(encoded)} bytes")

# ---- Create QR Code ----
qr = qrcode.QRCode(
    version=None,  # Auto size
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # Use L for max capacity
    box_size=10,
    border=4,
)

qr.add_data(encoded)
qr.make(fit=True)

img = qr.make_image(fill_color="black", back_color="white")

# ---- Save output QR ----
output_file = "allocation_qr.png"
img.save(output_file)

print(f"✅ QR Code generated successfully: {output_file}")
print(f"To decode: base64-decode → zlib-decompress → JSON")