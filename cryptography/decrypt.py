from cryptography.fernet import Fernet
from base64 import urlsafe_b64encode
from hashlib import sha256
keyText = "Test123456"
key=urlsafe_b64encode(sha256(keyText).digest())
f = Fernet(key)
cipher = "gAAAAABcY9bzp1sTBAuIjjoR3kaI3eDSIGGwcMptR_45JwhADCpSxZVGsHYnfnU7YTrvN0LEthoU4b8GuQx91iDdRfetxxGTb56uNyBWbyAA1gNSewYFhqY="
token=f.decrypt(cipher)
print(token)