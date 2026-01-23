#!/bin/bash
set -e

# ===== CONFIG =====
PROJECT_DIR="/opt/incremental_ids"
SYSTEMD_DIR="/etc/systemd/system"
SERVICE_SRC_DIR="$PROJECT_DIR/systemd_service_file"
JAVA_BIN=$(readlink -f $(which java))

IL_SERVICES=(
  "ids_dashboard.service"
  "ids_il.timer"
  "ids_il.service"
)

# ===== CHECK ROOT =====
echo "Cấp quyền đọc ghi cho User hiện tại: $SUDO_USER" 

if [[ "$EUID" -ne 0 ]]; then
  echo "\033[0;32m[!]\033[0m Please run as root"
  exit 1
fi

# ===== CHECK REQUIRE PACKAGE - PYTHONVENV =====
echo -e "\033[0;32m[i]\033[0m Installing required apt packages..."

packages=("python3" "python3-pip" "python3-venv" "jq")
for package in "${packages[@]}"; do
    
    if ! dpkg-query -l "$package" | grep -E "^ii"; then
    # sudo apt update
    sudo apt install -y "$package"
    fi
done

# ===== Tạo thư mục chạy =====
echo -e "==========================================="
echo -e "\033[0;32m[2/5]\033[0m Creating project structure..."

# không nên xóa rồi tạo lại vì có logs
if [ ! -d "/opt/incremental_ids" ]; then
    sudo mkdir -p /opt/incremental_ids
    sudo chown -R $SUDO_USER:$SUDO_USER /opt/incremental_ids
fi

if [ ! -d "/opt/incremental_ids/app_logs" ]; then
    sudo mkdir -p /opt/incremental_ids/app_logs
fi

if [ ! -d "/opt/incremental_ids/flows_parquet" ]; then
    sudo mkdir -p /opt/incremental_ids/flows_parquet
fi

sudo chown -R $SUDO_USER:$SUDO_USER /opt/incremental_ids/flows_parquet /opt/incremental_ids/app_logs
# sudo chmod 600 /opt/incremental_ids/flows_parquet /opt/incremental_ids/app_logs #755

# ===================
# cp src qua
# ===================
sudo cp -r ./systemd_service_file /opt/incremental_ids/
sudo cp ./requirements.txt /opt/incremental_ids/
sudo cp -r ./src /opt/incremental_ids/
sudo cp -r ./train_data /opt/incremental_ids/

sudo chown -R $SUDO_USER:$SUDO_USER /opt/incremental_ids/train_data

# ===================
# Tạo venv
# ===================
echo -e "==========================================="
echo -e "\033[0;32m[i]\033[0m Creating venv ..."

python3 -m venv /opt/incremental_ids/venv
source /opt/incremental_ids/venv/bin/activate
pip install -r requirements.txt

# enable java running without root
if [[ "$JAVA_BIN" == */jre/bin/java ]]; then
    JRE_JAVA_DIR=$(dirname "$JAVA_BIN")
    sudo setcap cap_net_raw,cap_net_admin+eip "$JRE_JAVA_DIR"/* 2>/dev/null || true
fi

echo "[i] Kiểm tra capability"
getcap "$JAVA_BIN"
echo "[i] Đã gán capability cho: $JAVA_BIN"

# ====== =================
# INSTALL SERVICES
# =====================
echo "\033[0;32m[+]\033[0m Installing IDS services..."

# ===== CHECK PROJECT DIR =====
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "[!] Project directory not found: $PROJECT_DIR"
  exit 1
fi

# ===== COPY SERVICE FILES =====
for svc in "${IL_SERVICES[@]}"; do
  echo "  → Installing $svc"
  cp "$SERVICE_SRC_DIR/$svc" "$SYSTEMD_DIR/$svc"
done

# ===== SYSTEMD RELOAD =====
systemctl daemon-reexec
systemctl daemon-reload

# ===== ENABLE & START =====
systemctl enable ids_dashboard.service
systemctl start ids_dashboard.service

systemctl enable ids_il.timer
systemctl start ids_il.timer

# ===== DONE ==========
echo ""
echo "\033[0;32m[✓]\033[0m IDS services installed & running"
echo ""
echo "Check status:"
for svc in "${SERVICES[@]}"; do
  echo "  systemctl status ${svc%.service}"
done
