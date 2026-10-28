# Declaração de uso de ferramentas de IA

## Ferramenta usada

**Claude Code** (agente de programação da Anthropic, modelo Claude da família Opus). Usei em
duas etapas: primeiro num ambiente de nuvem com terminal (Parte 1 e o código da Parte 2) e
depois na minha máquina (Windows, extensão do VS Code) para rodar a Parte 2 com os textos
reais das normas, que o ambiente de nuvem não conseguia baixar. Não usei Copilot nem ChatGPT.

## Onde e como a IA ajudou

| Etapa | O que a IA fez | O que eu (candidato) fiz/validei |
|---|---|---|
| Levantamento de fontes | Testou a acessibilidade de cada fonte sugerida no ambiente (QUALAR, Open-Meteo, Planetary Computer, INPE etc. estavam bloqueados) e encontrou alternativas públicas e anônimas: espelho CETESB na OpenAQ (S3), GOES-16 (AOD e fogo), ERA5 do NCAR, Sentinel-5P em COG, WorldCover, Copernicus DEM | Decidir as trocas e conferir que as fontes são oficiais/reconhecidas |
| Descoberta das estações | Varreu o bucket da OpenAQ por location IDs dentro do estado de SP e encontrou a rede CETESB (IDs 5012-5305 e 233408-233410) | Conferir nomes/coordenadas das estações |
| Código da Parte 1 | Escreveu os coletores (leitura parcial por HTTP range de NetCDF/HDF5 e COG), a engenharia de variáveis, o benchmark, o CQR, a previsão D+1, o SHAP, os mapas, a CLI e a API | Revisar a física das variáveis (AOD/BLH, correção higroscópica, fogo a favor do vento), o desenho da validação e a ausência de vazamento no lag espacial |
| Código da Parte 2 | Escreveu o parser hierárquico (art./§/inciso/alínea/anexo), o chunking, a recuperação híbrida com reranker, o agente (plano → ferramentas → resposta → validação), o log e os scripts de avaliação | Escrever/revisar as perguntas e respostas esperadas do conjunto de avaliação contra o texto das normas |
| Parte 2 com os textos reais | Localizou cópias públicas do DOU para a 491/2018 (FAOLEX/FAO) e a 506/2024 (Prefeitura do Rio), porque o in.gov.br bloqueia acesso automatizado. Ajustou o parser ao formato real (cabeçalho de impressão do DOU, tabelas dos anexos, HTML malformado do Planalto) e modelou a revogação parcial da 491 pela 506 | Conferir no texto das normas os valores, as datas de vigência e os dispositivos revogados, e os gabaritos do conjunto de avaliação |
| Depuração | Achou um defeito no reranker (o llama-cpp-python 0.3 deixou de preencher os logits que o código lia) e reorganizou a recuperação para caber em CPU | Validar a correção com exemplos conhecidos |
| Escolha de modelos | Montou e rodou os benchmarks (modelos de ML, embeddings, LLMs locais) | Interpretar os resultados e escolher com base nos números |
| Avaliação final e correções | Rodou a avaliação completa do agente, leu os erros no log e propôs as correções: a etapa vigente do padrão calculada fora do LLM (`padrao_mp25_vigente`), recusa quando sobra número sem fonte, citações do texto como fonte da lista. Também rodou `make all-sample` num clone limpo | Conferir cada erro no log e decidir quais perguntas rodar de novo, sem mexer no gabarito |
| Documentação | Redigiu README, RELATORIO e o roteiro do vídeo | Revisar e ajustar |

## Cuidados tomados

- Nenhum número do relatório foi escrito "de cabeça": todos vêm dos CSVs em
  `reports/metrics/`, gerados pelo código do repositório.
- Os textos das normas vêm de fontes oficiais (Planalto, CONAMA) ou de cópias públicas da
  página do DOU (FAOLEX, Prefeitura do Rio), com a URL registrada ao lado de cada arquivo.
  A IA não transcreveu norma de memória.
- O LLM do agente (local, Qwen3/Qwen2.5 em GGUF) é um componente do sistema, separado
  da IA usada para programar.
- Consigo explicar e modificar qualquer trecho. Os pontos centrais estão comentados
  no próprio código (docstrings de cada módulo).
