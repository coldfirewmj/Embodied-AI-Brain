#!/bin/bash

# ===== 安全清理代理设置，防止干扰本地通信 =====
unset HTTP_PROXY
unset http_proxy
unset HTTPS_PROXY
unset https_proxy
unset ALL_PROXY
unset all_proxy
export PYTHONPATH=/usr/local/lib:$(pwd):${PYTHONPATH}

python chat_to_llm.py
