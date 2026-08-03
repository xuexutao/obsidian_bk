# Python Code Style

Apply the following rules only when creating or modifying Python code:
- 要满足 python3.9 编译的需要
- Add type annotations to public APIs.
- Prefer pathlib.Path over os.path.
- Catch specific exceptions.

## 导入顺序

标准库 → 第三方库 → 本地模块，三段式，组间空行，组内按字母序。

禁止：`import *`、`from xxx import *`、循环导入、无理由的函数内 import。

## 命名

| 类型       | 风格                         | 示例                         |
| -------- | -------------------------- | -------------------------- |
| 模块/包     | snake_case                 | `data_loader.py`           |
| 类        | PascalCase                 | `DataPipeline`             |
| 函数/变量/方法 | snake_case                 | `fetch_records`, `user_id` |
| 常量       | UPPER_SNAKE                | `MAX_RETRY`                |
| 私有       | `_` 前缀                     | `_parse_header`            |
| 布尔函数     | `is_/has_/should_/can_` 前缀 | `is_valid`                 |

避免单字符名(`l/O/I`)、拼音、非通用缩写。

## 脚本入口

所有可执行脚本必须：

```python
def main() -> int:
    args = parse_args()
    try:
        result = run(args)
        logger.info("done: %s", result)
        return 0
    except Exception as exc:
        logger.exception("failed: %s", exc)
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

- `main()` 返回 int 退出码，`parse_args()` 单独抽出，业务逻辑入 `run()` 或子函数
- 顶层只放 import / 常量 / `main()` / 入口守卫，禁止裸执行代码

## Docstring 与注释

- 模块顶部 docstring 简述用途
- 公共函数/类必须有 docstring（Google 或 reST 风格），说明参数、返回值、异常
- 行内注释解释 *为什么*，不是 *是什么*
- 复杂逻辑（正则、状态机、算法）必须有 1-3 行注释说明意图

## 进度条 (tqdm)

任何遍历超过 1 秒的操作必须用 tqdm 包裹：

```python
from tqdm import tqdm

for item in tqdm(items, desc="processing", unit="item"):
    ...
```

嵌套 tqdm 设 `position=1, leave=False`。禁止用 print 模拟进度条。

## 并发

| 场景 | 选型 |
|------|------|
| IO 密集（HTTP/文件/DB） | `ThreadPoolExecutor` |
| CPU 密集（计算/加密/图像） | `ProcessPoolExecutor` |
| 大量异步 IO（数千连接） | `asyncio` + `aiohttp` |

- 默认 `max_workers = min(32, os.cpu_count() + 4)`
- 共享可变状态必须用 `Lock`；进程池禁止传 lambda/闭包/不可 pickle 对象
- 始终处理每个 future 的异常

## 路径处理

统一用 `pathlib.Path`，不用 `os.path`：

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent
output = ROOT / "data" / f"result_{date:%Y%m%d}.json"
output.parent.mkdir(parents=True, exist_ok=True)
```

## 日志

禁止用 `print` 输出业务日志：

```python
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
```

- 脚本入口用 `logger.info`，调试用 `logger.debug`，异常用 `logger.exception`
- 字符串用 `%s` 占位符，不 f-string 拼接

## 错误处理

- 禁止裸 `except:` / `except Exception:` 不做处理
- 仅顶层 `main()` 允许宽 `except`（统一退出码）
- 具体异常细粒度捕获：`except requests.Timeout`
- 抛异常用 `raise X from Y` 保留原因链
- 上下文用 `with`：文件、锁、连接、DB session
- 对外接口 docstring 中声明 `Raises`

## 类型注解

- 所有公共函数必须有类型注解（参数 + 返回值）
- 私有函数鼓励加
- Python 3.9+ 用小写 `list[int]`、`dict[str, Any]`

## 其他

- 行长 ≤ 100，字符串拼接用括号而非 `\` 续行
- 字符串默认双引号，docstring 内层用单引号
- 字典/列表末尾保留逗号
- magic number 抽常量并命名
- 随机数：`random.seed(42)` 复现；密码相关用 `secrets`
- 时间：存 UTC、渲染再转本地
