import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. PARAMETRIZAÇÃO DO CENÁRIO (Dados extraídos do Edital)
total_candidatos = 7781
vagas_segunda_fase = 300
pontuacao_candidato = 59
pontuacao_maxima = 80

# 2. MODELAGEM ESTATÍSTICA (Distribuição Normal)
# Baseado na complexidade da banca (FGV), estimamos os parâmetros da curva.
media_estimada = 36  # Média projetada de acertos
desvio_padrao = 9.6  # Dispersão da amostra

# 3. CÁLCULOS DE PERFORMANCE E PROBABILIDADE
# Cálculo da posição relativa (Z-Score)
z_score = (pontuacao_candidato - media_estimada) / desvio_padrao
percentil = stats.norm.cdf(z_score) * 100

# Estimativa da Nota de Corte para o Top 300
proporcao_vagas = vagas_segunda_fase / total_candidatos
corte_z = stats.norm.ppf(1 - proporcao_vagas)
nota_corte_estimada = media_estimada + corte_z * desvio_padrao

# 4. VISUALIZAÇÃO DE DADOS (Data Visualization)
x = np.linspace(0, pontuacao_maxima, 1000)
y = stats.norm.pdf(x, media_estimada, desvio_padrao)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Distribuição de Notas (Projeção)', color='#1f77b4', lw=2)

# Linhas de Referência
plt.axvline(pontuacao_candidato, color='green', linestyle='--', label=f'Pontuação Candidato ({pontuacao_candidato} pts)')
plt.axvline(nota_corte_estimada, color='red', linestyle=':', label=f'Corte Estimado ({nota_corte_estimada:.1f} pts)')

# Área de Classificação
plt.fill_between(x, y, where=(x >= nota_corte_estimada), color='red', alpha=0.2, label='Zona de Classificação (Top 3.8%)')

# Estética Profissional (UX do Gráfico)
plt.title('Simulação Estatística: Concurso Delegado PC-PI', fontsize=14, pad=15)
plt.xlabel('Pontuação (Questões Corretas)')
plt.ylabel('Densidade de Candidatos')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

# Output de Análise
print(f"Resultado da Análise: O candidato superou {percentil:.2f}% da amostra estimada.")
