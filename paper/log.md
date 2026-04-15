```python
import sys
import os
from pathlib import Path
from loguru import logger
# -----------------------------------------------------------------------------
# 配置路径
# -----------------------------------------------------------------------------
# 获取项目根目录 (假设 logger.py 在 src/utils/ 下)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]
LOG_DIR = PROJECT_ROOT / "logs"
# 自动创建 logs 目录
if not LOG_DIR.exists():
	LOG_DIR.mkdir(parents=True)
# -----------------------------------------------------------------------------
# 自定义日志格式
# -----------------------------------------------------------------------------

# <green>时间</green> | <level>级别</level> | <cyan>模块:行号</cyan> - <level>内容</level>

CONSOLE_FORMAT = (
	"<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
	"<level>{level: <8}</level> | "
	"<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
	"<level>{message}</level>"
	)

FILE_FORMAT = (
	"{time:YYYY-MM-DD HH:mm:ss} | "
	"{level: <8} | "
	"{name}:{line} - "
	"{message}"
)
def setup_logger(log_file_name="autoscene.log"):
	"""
	配置并返回全局 logger
	Args:
	log_file_name: 日志文件名
	"""
	# 1. 移除默认的 handler (避免重复打印)
	logger.remove()
	# 2. 添加控制台输出 (带颜色)
	logger.add(
	sys.stderr,
	format=CONSOLE_FORMAT,
	level="INFO", # 开发阶段可以用 DEBUG
	colorize=True
	)
	
	# 3. 添加文件输出 (滚动保存，每天一个文件或每10MB一个文件)
	
	log_file_path = LOG_DIR / log_file_name
	
	logger.add(
		log_file_path,
		format=FILE_FORMAT,
		level="DEBUG", # 文件里记录更详细的信息
		rotation="10 MB", # 文件超过10MB重命名
		retention="1 week", # 保留一周的日志
		encoding="utf-8"
	)
	return logger

# 初始化一个默认 logger 供直接导入使用

LOGGER = setup_logger()
```