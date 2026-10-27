# Declaração de uso de ferramentas de IA

## Ferramenta usada

**Claude Code** (agente de programação da Anthropic, modelo Claude da família Opus), num
ambiente de nuvem com terminal. Não usei Copilot nem ChatGPT.

## Onde e como a IA ajudou

| Etapa | O que a IA fez | O que eu (candidato) fiz/validei |
|---|---|---|
| Levantamento de fontes | Testou a acessibilidade de cada fonte sugerida no ambiente (QUALAR, Open-Meteo, Planetary Computer, INPE etc. estavam bloqueados) e encontrou alternativas públicas e anônimas: espelho CETESB na OpenAQ (S3), GOES-16 (AOD e fogo), ERA5 do NCAR, Sentinel-5P em COG, WorldCover, Copernicus DEM | Decidir as trocas e conferir que as fontes são oficiais/reconhecidas |
| Descoberta das estações | Varreu o bucket da OpenAQ por location IDs dentro do estado de SP e encontrou a rede CETESB (IDs 5012-5305 e 233408-233410) | Conferir nomes/coordenadas das estações |
| Código da Parte 1 | Escreveu os coletores (leitura parcial por HTTP range de NetCDF/HDF5 e COG), a engenharia de variáveis, o benchmark, o CQR, a previsão D+1, o SHAP, os mapas, a CLI e a API | Revisar a física das variáveis (AOD/BLH, correção higroscópica, fogo a favor do vento), o desenho da validação e a ausência de vazamento no lag espacial |
| Código da Parte 2 | Escreveu o parser hierárquico (art./§/inciso/alínea/anexo), o chunking, a recuperação híbrida com reranker, o agente (plano → ferramentas → resposta → validação), o log e os scripts de avaliação | Escrever/revisar as perguntas e respostas esperadas do conjunto de avaliação contra o texto das normas |
| Escolha de modelos | Montou e rodou os benchmarks (modelos de ML, embeddings, LLMs locais) | Interpretar os resultados e escolher com base nos números |
| Documentação | Redigiu README, RELATORIO e o roteiro do vídeo | Revisar e ajustar |

## Cuidados tomados

- Nenhum número do relatório foi escrito "de cabeça": todos vêm dos CSVs em
  `reports/metrics/`, gerados pelo código do repositório.
- Os textos das normas vêm só das fontes oficiais baixadas pelo script. A IA não
  transcreveu norma de memória.
- O LLM do agente (local, Qwen3/Qwen2.5 em GGUF) é um componente do sistema, separado
  da IA usada para programar.
- Consigo explicar e modificar qualquer trecho. Os pontos centrais estão comentados
  no próprio código (docstrings de cada módulo).
