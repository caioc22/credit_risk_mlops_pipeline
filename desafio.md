Desafio Técnico: MLOps Engineer Pleno - Agibank
1. Contexto e Objetivo
No Agibank, nosso time de Data Science é movido por gerar valor rápido e escalável. Atualmente, temos uma forte cultura de modelagem em R, e nosso desafio como Engenharia de Machine Learning é sustentar esse ecossistema na AWS, garantindo que o ciclo de vida dos modelos seja automatizado, robusto e simples de manter.
Sua missão é implementar um pipeline básico de MLOps que operacionalize um modelo de crédito, focando em boas práticas de engenharia e automação.

2. O Desafio de Engenharia e Negócio
O dataset de referência é o Home Credit Default Risk, cujo objetivo é prever a capacidade de pagamento de clientes.
Neste cenário, você não será avaliado pela performance estatística do modelo, mas sim pela sua capacidade de estruturar um código experimental em um pipeline organizado e reproduzível.
Sua responsabilidade: O time de Data Science entregou um script funcional em R. Cabe a você:
Organizar o treinamento: Garantir que o script rode de forma isolada e reproduzível.
Criar uma API: Disponibilizar o modelo através de uma API REST.
Implementar automação básica: Configurar um pipeline CI/CD simples.

3. Escopo do Desafio
O desafio está dividido nos próximos quatro pilares técnicos principais.
Pilar 1: Conteinerização
Crie um ambiente reproduzível para executar o modelo.
Tasks:
Construir um Dockerfile que prepare um ambiente capaz de executar o script R fornecido.
Garantir que o treinamento rode dentro do container e gere o artefato model_v1.rds.
Usar variáveis de ambiente para configurações (paths, parâmetros, etc.).
Pilar 2: API de Inferência
Disponibilize o modelo treinado através de uma API.
Tasks:
Criar uma API REST simples (pode usar Python com Flask/FastAPI ou R com Plumber).
Implementar os seguintes endpoints:
/predict - Recebe dados e retorna predição com probabilidade
/health - Health check da aplicação
/model-info - Retorna informações sobre o modelo (versão, features esperadas)
Implementar validação básica de entrada e tratativa de erro.
Diferenciais: Documentação automática da API (Swagger/OpenAPI) e logging estruturado das requisições.
Pilar 3: CI/CD Básico
Configure um pipeline automatizado simples.
Task: Configurar um workflow (GitHub Actions, GitLab CI ou similar) que execute as seguintes etapas:
Execute validação básica do código (linting).
Execute o build do Docker image e verificar se o modelo foi gerado corretamente após o treino.
Execute um teste de integração simples (ex: executar um teste chamando o endpoint de /health).
Gere tags para as imagens Docker (pode usar hash do commit ou data)
Diferenciais: Cache de dependências para acelerar o pipeline e relatório de cobertura de testes.
Nota: Não é necessário fazer push real para registry. Simular o processo é suficiente.
Pilar 4: Gestão de Features
Demonstre organização no tratamento de dados.
Tasks:
Criar uma estrutura para armazenar features processadas (pode ser Parquet, CSV ou JSON).
Criar um arquivo de metadados (features_metadata.json) que descreva:
Nome das features utilizadas
Tipo de cada feature
Versão do schema
Exemplo de metadata:

```json
{
  "version": "1.0",
  "features": [
    {
      "name": "AMT_INCOME_TOTAL",
      "type": "numeric",
      "required": true
    }
  ],
  "created_at": "2026-01-01"
}
```


4. O que será avaliado 
Será julgado seu poder de síntese, de fazer mais com menos, de arquitetar a solução da forma mais simples e robusta possível.
Critérios principais:
Funcionalidade: A solução funciona? Todos os pilares obrigatórios foram implementados?
Qualidade de Código: Organização, legibilidade, modularidade.
Automação: Pipeline CI/CD funcional e está bem estruturada?
Documentação: Instruções estão claras? As decisões estão documentadas?
Boas Práticas: Qualidade do código, tratamento de erros, validações, logs.
5. Entrega
Código: Repositório público no GitHub, GitLab ou arquivo .zip com histórico de commits.
Documentação Obrigatória: README.md com visão geral da solução, pré-requisitos, intruções de passo a passo, exemplos de uso (curl ou requests Python) e decisões técnicas relevantes.
Prazo: 7 dias corridos. Caso precise de extensão, comunique com antecedência.
6. Diferenciais
Não são obrigatórios, mas demonstram atenção a detalhes.

Boa sorte, Agilizado! Estamos ansiosos para ver sua solução.
