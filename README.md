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

Configuracao padrao:

- modelo logico: `qwen2.5:1.5b-instruct`
- repositorio Hugging Face: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- arquivo GGUF: `qwen2.5-1.5b-instruct-q2_k.gguf`
- backend: `llama-cpp-python`

## Observacao importante

Se `llama-cpp-python` nao estiver corretamente instalado no ambiente, o app continua funcionando com fallback automatico sem LLM. A melhor qualidade de decisao ambigua depende da instalacao correta de `llama-cpp-python` e do download do modelo GGUF.

No Streamlit Community Cloud, a versao de Python do ambiente pode ser superior a `3.12`. Como `llama-cpp-python` pode nao ter wheel compativel nessas versoes, o `requirements.txt` instala essa dependencia apenas quando `python_version < "3.13"`, preservando o deploy com fallback automatico sem LLM.
