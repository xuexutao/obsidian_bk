cd "$(dirname "$0")"

# 读写后端（/api/read、/api/save，监听 60051），已运行则跳过
if ! lsof -nP -iTCP:60051 -sTCP:LISTEN >/dev/null 2>&1; then
  nohup python3 server.py > /tmp/obsidian_vault_server.log 2>&1 &
fi

open http://127.0.0.1:60051/index.html

# 静态页面服务（监听 60051）
python3 -m http.server 60051
