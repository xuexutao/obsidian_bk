cd "$(dirname "$0")"

# 确保运行最新版的 server.py（同时承担静态页面服务与 /api/* 读写接口，监听 60051）
if lsof -nP -iTCP:60051 -sTCP:LISTEN >/dev/null 2>&1; then
  kill $(lsof -t -iTCP:60051 -sTCP:LISTEN) 2>/dev/null || true
  sleep 0.3
fi

(sleep 0.6 && open http://127.0.0.1:60051/index.html) &
python3 server.py


