"""启动入口：python run.py 即可启动后端并提供前端页面。"""

import uvicorn


def main():
    """启动 FastAPI 服务（本机访问）。"""
    # TODO：把原命令 python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
    # 的三个部分填进来：
    #   第 1 个参数是应用位置的字符串，格式 "模块路径:变量名"，原命令里已经有了
    #   host 和 port 是关键字参数，值照抄原命令
    uvicorn.run("app.backend.main:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()