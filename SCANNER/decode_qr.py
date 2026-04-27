import json
import zlib
import base64
from pyzbar.pyzbar import decode
from PIL import Image

# ---- Scan QR code ----
qr_file = "allocation_qr.png"
img = Image.open(qr_file)
results = decode(img)

if not results:
    print("Error: No QR code found in the image.")
    exit()

# ---- Extract encoded string ----
encoded = results[0].data.decode("utf-8")

# ---- Base64 decode → zlib decompress → JSON parse ----
compressed = base64.b64decode(encoded)
json_string = zlib.decompress(compressed).decode("utf-8")
data = json.loads(json_string)

# ---- Save to file ----
output_file = "decoded_allocation.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"✅ JSON extracted successfully → {output_file}")
print(json.dumps(data, indent=2))
