from cryptography.fernet import Fernet
from base64 import urlsafe_b64encode
from hashlib import sha256
keyText = "Test123456"
key=urlsafe_b64encode(sha256(keyText).digest())
f = Fernet(key)
token = f.encrypt("yCR3sd)]]S=n6b++")
print (token)