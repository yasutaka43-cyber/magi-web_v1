# MAGI風 合議システム（Streamlit Cloud）
https://magi-webv1-hivsnsjeb7v5rudddyuvtz.streamlit.app/

## 使い方（URLを開くだけ）
Streamlit CloudでDeploy後、発行されたURLをクリックすると使えます。

## Deploy手順（クリック中心）
1. このリポジトリをStreamlit CloudでNew app → Deploy
2. Main file path は `app.py`

## OpenAI連携（任意）
AIで人格生成したい場合のみ設定します。

Streamlit Cloud → App → Settings → Secrets に下を追加:

OPENAI_API_KEY="sk-xxxx"
OPENAI_MODEL="gpt-5-mini"

※APIキーは必ずSecretsへ。コードやHTMLに直書きしない。
