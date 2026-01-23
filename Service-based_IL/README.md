# Illustration


# How to use this archirtecture as a service
- Chạy với các lệnh phía bên dưới:
> sudo chmod +x install.sh
> sudo ./install.sh


# CÁC CÔNG PHÁP FIX LỖI/BẾ TẮC
### PIP REQUIREMENTS
Dùng luôn pigar
> pigar generate

### SERVICE
```
while read p; do
  echo "Installing $p"
  pip install "$p" || echo "❌ Failed: $p"
done < requirements.txt
```

### Công pháp pip
```
pip list --format=freeze > requirements.txt
```

### Nếu chạy service lỗi tại ký tự ^M$ của windows 

```
sed -i 's/\r$//' install_service.sh
- Sợ thì tạo backup
sed -i.bak 's/\r$//' install_service.sh
```