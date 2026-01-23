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
while read p; do
  echo "Installing $p"
  pip install "$p" || echo "❌ Failed: $p"
done < requirements.txt


# Công pháp pip
pip list --format=freeze > requirements.txt


# Nếu chạy service lỗi tại ký tự ^M$ của windows 
sed -i 's/\r$//' install_service.sh
> Sợ thì tạo backup
sed -i.bak 's/\r$//' install_service.sh




# KỊCH BẢN
# Tạo cặp veth0 và veth1 (tự động kết nối với nhau)
sudo ip link add veth0 type veth peer name veth1

# Đưa cả 2 interface lên
sudo ip link set veth0 up
sudo ip link set veth1 up

# Gán IP để test ping
sudo ip addr add 192.168.0.1/24 dev veth0
sudo ip addr add 192.168.0.2/24 dev veth1

# Kiểm tra
ip addr show veth0
ip addr show veth1

sudo hping3 --rand-source -c 100000 -i u300 -q 10.0.0.2