"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保ったまま続けて質問する（宿題・実装）

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(model=MODEL, contents=question)

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）

    return response.text


def chat() -> None:
    """会話履歴を保ったまま、続けて質問できる対話モード。

    client.chats.create() で作った Chat は、送受信したメッセージを履歴として保持し、
    send_message() のたびに履歴ごとモデルに送る。空行・exit・Ctrl+D で終了する。
    """
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL)

    print("会話モード（空行・exit・Ctrl+D で終了）")
    while True:
        try:
            question = input("\n[質問] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question in ("", "exit", "quit"):
            break

        try:
            response = session.send_message(question)
        except errors.APIError as e:
            # 失敗したやり取りは履歴に残らないので、そのまま次の質問に進める
            print(api_error_message(e.code, e.message))
            continue

        print(f"\n{response.text}")

    turns = len(session.get_history()) // 2
    print(f"[終了] {turns} 往復のやり取りをしました。")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--chat", action="store_true", help="会話履歴を保ったまま続けて質問する"
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    if args.chat:
        chat()
        return
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
