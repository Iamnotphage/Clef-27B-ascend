"""Clef-27B 昇腾推理适配入口骨架。"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="Clef-27B 昇腾推理入口，推理适配待实现。"
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR", "/workspace/yangchen/models/Clef-27B"),
        help="完整模型目录，可通过 MODEL_DIR 设置默认值。",
    )
    parser.parse_args()
    parser.error("推理适配待实现。当前入口用于仓库初始化。")


if __name__ == "__main__":
    main()
