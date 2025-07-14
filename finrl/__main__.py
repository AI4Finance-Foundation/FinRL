"""
FinRL 包入口文件 (Package Entry Point)

这个文件允许用户通过 'python -m finrl' 命令来运行 FinRL 框架
而不需要直接调用具体的脚本文件。

使用方法:
    python -m finrl --mode=train   # 训练模式
    python -m finrl --mode=test    # 测试模式  
    python -m finrl --mode=trade   # 交易模式

这种设计符合 Python 包的标准实践，使得框架更易于使用和部署。
"""
from __future__ import annotations

from finrl.main import main

if __name__ == "__main__":
    # 调用主函数并使用其返回值作为程序退出状态码
    # 0 表示成功，非0表示有错误发生
    raise SystemExit(main())
