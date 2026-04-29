# site-orcamento-ia

Aplicativo Streamlit simples para preenchimento inteligente de planilhas de orçamento.

## Execução

```bash
streamlit run app.py
```

## Motor de matching

O app usa um motor textual TF-IDF com `char_wb` e `ngram_range=(3, 5)`, combinado com:

- peso semântico: `0.55`
- peso fuzzy: `0.20`
- peso de regras técnicas: `0.25`

## LLM para casos ambíguos

O app pode usar uma LLM apenas como reranqueadora em casos ambíguos, com fallback automático para o ranking tradicional.

Configuração padrão:

- modelo lógico: `qwen2.5:1.5b-instruct`
- repositório Hugging Face: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`
- arquivo GGUF: `qwen2.5-1.5b-instruct-q2_k.gguf`
- backend: `llama-cpp-python`

## Observação importante

Se `llama-cpp-python` não estiver corretamente instalado no ambiente, o app continua funcionando com fallback automático sem LLM. A melhor qualidade de decisão ambígua depende da instalação correta de `llama-cpp-python` e do download do modelo GGUF.
