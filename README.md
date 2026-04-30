# site-orcamento-ia

Aplicativo Streamlit simples para preenchimento inteligente de planilhas de orcamento.

## Execucao

```bash
streamlit run app.py
```

## Motor de matching

O app usa um motor textual TF-IDF com `char_wb` e `ngram_range=(3, 5)`, combinado com:

- peso semantico: `0.55`
- peso fuzzy: `0.20`
- peso de regras tecnicas: `0.25`

## LLM para casos ambiguos

O app pode usar uma LLM apenas como reranqueadora em casos ambiguos, com fallback automatico para o ranking tradicional.

Configuracao padrao atual:

- backend: `huggingface`
- modelo logico: `qwen2.5:3b-instruct`
- modelo remoto: `Qwen/Qwen2.5-3B-Instruct`
- uso: apenas em casos ambiguos

### Streamlit Community Cloud

Para habilitar a LLM no Streamlit Cloud, configure um secret:

- `HF_API_TOKEN = seu_token_do_hugging_face`

Opcionalmente, voce tambem pode definir:

- `HF_PROVIDER = nome_do_provider`

O app le `st.secrets` automaticamente e usa esse token no backend remoto do Hugging Face.

## Fallback seguro

Se o backend remoto falhar, atrasar, retornar algo invalido ou o token nao estiver configurado, o app continua funcionando com fallback automatico sem LLM.

## Backend local opcional

Se quiser rodar a LLM localmente fora do Streamlit Cloud, ainda e possivel usar `llama-cpp-python` com GGUF:

- `LLM_BACKEND_MODE=llama_cpp`
- `LLM_MODEL_REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- `LLM_MODEL_FILE=qwen2.5-1.5b-instruct-q2_k.gguf`

## Observacao importante

No Streamlit Community Cloud, o caminho mais estavel para LLM e o backend remoto do Hugging Face. O backend `llama-cpp-python` continua integrado, mas pode nao estar disponivel dependendo da versao de Python do ambiente.
