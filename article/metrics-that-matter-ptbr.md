# Nem Todo Erro Custa o Mesmo

**Fundamentos Bayesianos das Métricas de Avaliação para Detecção de Fraude: de F1 a Precision@Recall**

*Bruno Ramos Martins — 2026*

---

## Resumo

Detecção de fraude é um problema de classificação desbalanceado no qual dois tipos de erro carregam consequências fundamentalmente diferentes: deixar passar uma fraude custa muito mais do que bloquear uma transação legítima. Apesar dessa assimetria, o F1 score — que trata ambos os tipos de erro como igualmente custosos — permanece como a métrica de avaliação padrão na maioria dos workflows de praticantes. Este artigo argumenta que o F1 não está errado, mas que suas premissas raramente são explicitadas; e que quando essas premissas são violadas, ele produz rankings de modelos enganosos. Partindo de uma leitura probabilística da matriz de confusão, derivamos precisão e recall como probabilidades condicionais e as conectamos ao teorema de Bayes. Mostramos que o desbalanceamento de classes não é um problema de dados, mas um problema de taxa base, e que o limiar de decisão ótimo decorre diretamente da razão entre os custos dos erros. Em seguida, mapeamos o panorama completo de métricas — curvas ROC, curvas Precision-Recall e Precision@Recall — mostrando que cada métrica codifica uma premissa diferente sobre o contexto operacional. Um conjunto de experimentos reproduzíveis em um dataset sintético de fraude (100.000 transações, 0,1% de taxa de fraude) confirma que a escolha da métrica altera rankings de modelos, que o limiar padrão de 0,5 raramente é ótimo, e que Precision@Recall é a métrica operacionalmente mais honesta quando um alvo de recall de negócio está definido. O artigo encerra com um framework de decisão: um conjunto de perguntas práticas cujas respostas levam inequivocamente à métrica correta para um dado contexto de implantação.

---

## 1. Introdução

A motivação para este artigo surgiu de uma entrevista técnica centrada em um caso de detecção de fraude. O formato era direto: em cada etapa da construção de uma solução, era necessário explicar o que fazer, como fazer e por quê.

Esse tipo de problema expõe uma lacuna importante. Ao se preparar para vagas em machine learning, é comum focar intensamente em algoritmos, ajuste de hiperparâmetros e técnicas de otimização. Há uma quantidade substancial de teoria a cobrir, e grande parte do processo de aprendizado gira em torno de entender o comportamento do modelo e melhorar o desempenho. No entanto, quando o problema é apresentado em um cenário mais realista e contextualizado, essa abordagem "checklist" frequentemente se desfaz.

Em determinado momento da entrevista, uma pergunta aparentemente simples foi feita: entre diversas métricas de avaliação, qual você escolheria e por quê. A resposta foi baseada no meu conhecimento da época, mas carecia de profundidade. Em retrospecto, a questão não era a escolha em si, mas a ausência de um processo de raciocínio explícito. A seleção da métrica era amplamente automática, sem uma articulação clara das premissas sendo assumidas.

Essa percepção motivou este artigo.

A questão não é se o F1 é uma métrica ruim. É se o F1 é uma métrica *honesta* em um contexto específico: classificação binária sob desbalanceamento severo de classes, onde os dois tipos de erro carregam custos assimétricos. Em detecção de fraude, um falso negativo — aprovar uma transação fraudulenta — tipicamente custa uma ordem de magnitude a mais que um falso positivo — bloquear uma transação legítima. O F1 pondera esses dois tipos de erro igualmente por construção. Essa premissa implícita raramente é explicitada e, na prática, raramente é válida.

Este artigo torna essas premissas explícitas, deriva como deveria ser a métrica apropriada sob custos assimétricos, e mostra empiricamente que a escolha da métrica altera rankings de modelos e decisões de deploy de formas que afetam materialmente os resultados.

### O que move este artigo

Toda métrica de avaliação é uma resposta comprimida à pergunta: *"Para o que o modelo deve otimizar?"* Diferentes métricas codificam diferentes respostas. Accuracy codifica: *"Toda predição correta conta igualmente."* F1 codifica: *"Precision e Recall importam igualmente, e me preocupo mais com a classe minoritária do que a accuracy sozinha."* Precision@Recall codifica: *"Tenho um requisito de recall não negociável, e quero a maior precision alcançável dentro dele."*

A tese central é esta: em detecção de fraude, a última formulação é a honesta. A derivação matemática que a justifica se conecta ao teorema de Bayes, à teoria de decisão sensível a custos e à geometria das curvas Precision-Recall. Toda essa maquinaria está aqui, construída de baixo para cima.

### Para quem é este artigo

Um cientista de dados ou engenheiro de machine learning que usa precision, recall, F1 e AUC-ROC no trabalho diário, e que quer entender não apenas *como* essas métricas são computadas, mas *o que* elas medem e *quando* cada uma é apropriada. Nenhum conhecimento prévio de inferência bayesiana é assumido. A probabilidade condicional necessária é desenvolvida do zero.

### Roteiro

O artigo está estruturado da seguinte forma. As Seções 2–4 constroem a base probabilística: a matriz de confusão como probabilidade condicional, o papel da taxa base via teorema de Bayes e a teoria de decisão sensível a custos com seu limiar ótimo. As Seções 5–8 mapeiam o panorama de métricas: F1, a curva ROC, a curva Precision-Recall e Precision@Recall. A Seção 9 apresenta cinco experimentos que fornecem evidência empírica para cada afirmação feita nas seções teóricas. A Seção 10 sintetiza o framework de seleção de métricas. A Seção 11 encerra com conclusões práticas.

---

## 2. A Matriz de Confusão — Uma Leitura Probabilística

A maioria dos praticantes encontra a matriz de confusão cedo em sua formação, geralmente apresentada como uma tabela 2×2 de contagens. Esta seção a reformula como uma tabela de *probabilidades condicionais*. A reformulação é pequena, mas muda tudo.

### 2.1 As quatro células como eventos

Seja $Y \in \lbrace 0, 1 \rbrace$ o rótulo verdadeiro — 1 para fraude, 0 para legítimo — e $\hat{Y} \in \lbrace 0, 1 \rbrace$ a predição do modelo. As quatro células da matriz de confusão correspondem às quatro interseções do espaço de eventos $\lbrace Y = 0, Y = 1 \rbrace \times \lbrace \hat{Y} = 0, \hat{Y} = 1 \rbrace$:

| | Predito: Fraude ($\hat{Y}=1$) | Predito: Legítimo ($\hat{Y}=0$) |
|---|---|---|
| **Real: Fraude** ($Y=1$) | Verdadeiro Positivo (TP) | Falso Negativo (FN) |
| **Real: Legítimo** ($Y=0$) | Falso Positivo (FP) | Verdadeiro Negativo (TN) |

Cada contagem é proporcional a uma probabilidade conjunta. $TP \propto P(Y=1 \cap \hat{Y}=1)$, e assim por diante para cada célula.

O que torna esse enquadramento útil é que toda métrica padrão é uma probabilidade condicional derivada dessas probabilidades conjuntas.

### 2.2 Precision como probabilidade a posteriori

Precision responde à pergunta: *"Dado que o modelo sinalizou uma transação como fraude, qual a probabilidade de ser realmente fraude?"*

$$P = \frac{TP}{TP + FP} = \frac{P(Y=1 \cap \hat{Y}=1)}{P(\hat{Y}=1)} = P(Y=1 \mid \hat{Y}=1)$$

Na linguagem bayesiana, Precision é a **probabilidade a posteriori** de fraude dada a predição positiva do modelo. Ela responde à pergunta da perspectiva do operador: quando o alarme soa, com que frequência é real?

### 2.3 Recall como verossimilhança

Recall responde à pergunta: *"Dado que uma transação é de fato fraudulenta, qual a probabilidade de ser sinalizada?"*

$$R = \frac{TP}{TP + FN} = \frac{P(Y=1 \cap \hat{Y}=1)}{P(Y=1)} = P(\hat{Y}=1 \mid Y=1)$$

Na linguagem bayesiana, Recall é a **verossimilhança** de receber uma predição positiva dado que a classe verdadeira é positiva. Ela responde à pergunta da perspectiva da equipe de fraude: de todas as fraudes no dataset, quantas estamos capturando?

### 2.4 Por que esse enquadramento importa

Precision e Recall condicionam em coisas diferentes. Precision condiciona na saída do modelo (o que aconteceu após a predição). Recall condiciona na verdade fundamental (o que estava nos dados). Essas não são quantidades simétricas, e não podem ser combinadas em uma única métrica sem uma decisão de ponderação explícita — que o F1 faz silenciosamente.

Um modelo com Recall perfeito e Precision zero captura toda fraude, mas sinaliza toda transação legítima também. Um modelo com Precision perfeita e Recall zero é inteiramente seletivo, mas perde a maioria das fraudes. O equilíbrio correto entre esses extremos depende do custo de cada tipo de erro — que varia por aplicação. Em detecção de fraude, perder uma fraude é tipicamente muito mais custoso que um falso alarme. O enquadramento da matriz de confusão torna isso concreto: FN e FP medem modos de falha condicionais diferentes e justificam custos diferentes.

---

## 3. Teorema de Bayes e o Problema da Taxa Base

Saber que Precision é $P(Y=1 \mid \hat{Y}=1)$ e Recall é $P(\hat{Y}=1 \mid Y=1)$ convida à pergunta natural: como elas se relacionam? A resposta é o teorema de Bayes. E a ponte entre elas é a **taxa base** — a prevalência de fraude na população.

### 3.1 A derivação bayesiana

Pelo teorema de Bayes:

$$P(Y=1 \mid \hat{Y}=1) = \frac{P(\hat{Y}=1 \mid Y=1) \cdot P(Y=1)}{P(\hat{Y}=1)}$$

Traduzindo para o vocabulário de métricas, onde $\pi = P(Y=1)$ é a taxa base:

$$\text{Precision} = \frac{\text{Recall} \cdot \pi}{\text{Recall} \cdot \pi + (1 - \text{Specificity}) \cdot (1 - \pi)}$$

onde Specificity $= P(\hat{Y}=0 \mid Y=0) = TN / (TN + FP)$.

Esta equação tem uma consequência que surpreende muitos praticantes: **mesmo um modelo com alto recall pode ter precision muito baixa se a taxa base $\pi$ for muito pequena**.

### 3.2 Exemplo numérico: a armadilha da taxa base

Suponha que um modelo de detecção de fraude tenha Recall = 0,90 e Taxa de Falso Positivo = 0,01. À primeira vista, parece um modelo excelente: captura 90% das fraudes e sinaliza incorretamente apenas 1% das transações legítimas.

Agora aplique a fórmula bayesiana com uma taxa base de fraude típica de $\pi = 0{,}001$ (0,1%):

$$\text{Precision} = \frac{0{,}90 \times 0{,}001}{0{,}90 \times 0{,}001 + 0{,}01 \times 0{,}999} \approx \frac{0{,}0009}{0{,}0009 + 0{,}00999} \approx 0{,}082$$

Com apenas 8,2% de precision, aproximadamente 9 de cada 10 transações sinalizadas pelo modelo são legítimas. Apesar de excelente Recall e uma pequena Taxa de Falso Positivo, quase 92% dos alertas do modelo são falsos alarmes. Isso não é uma falha do modelo em nenhum sentido tradicional — o modelo é genuinamente bom. É uma consequência da taxa base.

### 3.3 Desbalanceamento de classes é um problema de taxa base

Este exemplo numérico reformula o "problema de desbalanceamento de classes" de forma importante. Desbalanceamento de classes é tipicamente apresentado como um problema de dados que requer técnicas como oversampling, undersampling ou geração de dados sintéticos. Essas técnicas podem ajudar durante o treinamento, mas não resolvem a questão fundamental.

A questão fundamental é que quando $\pi$ é muito pequeno, Recall e Precision têm uma relação estruturalmente adversarial. Um modelo que tenta alcançar alto Recall em uma classe rara inevitavelmente produzirá muitos Falsos Positivos relativos aos Verdadeiros Positivos, porque há muito mais negativos para serem erroneamente classificados. Isso não é um bug no pipeline de dados. É o teorema de Bayes operando sobre a distribuição real de classes da população.

A resposta correta ao desbalanceamento de classes na avaliação não é fingir que o desbalanceamento não existe (o que accuracy faz) ou calcular a média sobre ambos os tipos de erro sem reconhecer seus custos (o que F1 faz). A resposta correta é incorporar explicitamente a taxa base e os custos de erro assimétricos na própria métrica.

---

## 4. A Assimetria dos Erros

A seção anterior estabeleceu que o desbalanceamento de classes, interpretado como fenômeno de taxa base, cria uma tensão estrutural entre Precision e Recall. Esta seção formaliza o que essa tensão custa e deriva o limiar de decisão ótimo sob custos de erro assimétricos.

### 4.1 A matriz de custos

Todo sistema de predição binária implícita ou explicitamente codifica custos para cada célula da matriz de confusão. Em detecção de fraude:

- **Verdadeiro Positivo ($C_{TP}$)**: Uma fraude é sinalizada. A investigação identifica e bloqueia. Custo: overhead de investigação, geralmente pequeno ou compensado pela prevenção de perdas.
- **Verdadeiro Negativo ($C_{TN}$)**: Uma transação legítima é aprovada. Resultado operacional padrão. Custo: zero.
- **Falso Negativo ($C_{FN}$)**: Uma fraude é aprovada. A perda recai sobre a instituição (ou o portador do cartão, dependendo do regime de responsabilidade). Custos típicos incluem o valor da transação, tratamento de disputas, taxas de chargeback e dano reputacional.
- **Falso Positivo ($C_{FP}$)**: Uma transação legítima é bloqueada. O custo imediato é a receita perdida da transação. O custo indireto é fricção com o cliente, potencial abandono de conta e perda de valor de relacionamento.

Concretamente: se o valor médio de transação em risco em uma fraude é R\$2.500 e o custo de um falso alarme (investigação + atendimento ao cliente + valor de transação perdido) é R\$75, então $C_{FN} \approx 2.500$ e $C_{FP} \approx 75$. A razão é aproximadamente 33:1.

Esses números são ilustrativos. Os valores exatos variam por instituição, regras da bandeira e acordos de responsabilidade. O que importa para essa derivação é a **razão** $C_{FN} / C_{FP}$, não os valores absolutos.

### 4.2 O limiar ótimo

Um classificador probabilístico padrão produz um score $\hat{p}(x) = P(Y=1 \mid x)$ para cada transação. Na implantação, um limiar $\tau$ converte esse score em uma predição binária: sinalizar como fraude se $\hat{p}(x) \geq \tau$.

A escolha convencional é $\tau = 0{,}5$. Essa escolha minimiza o custo esperado apenas quando os dois tipos de erro custam o mesmo — o que quase nunca é verdade em detecção de fraude.

Para derivar o limiar ótimo, considere o custo esperado de predizer fraude versus predizer legítimo para uma transação com score $\hat{p}(x)$:

- **Custo esperado de predizer fraude** ($\hat{Y}=1$): $C_{FP} \cdot P(Y=0 \mid x) = C_{FP} \cdot (1 - \hat{p}(x))$
- **Custo esperado de predizer legítimo** ($\hat{Y}=0$): $C_{FN} \cdot P(Y=1 \mid x) = C_{FN} \cdot \hat{p}(x)$

Predizer fraude quando o primeiro é menor que o segundo:

$$C_{FP} \cdot (1 - \hat{p}(x)) < C_{FN} \cdot \hat{p}(x)$$

Resolvendo para $\hat{p}(x)$:

$$\hat{p}(x) > \frac{C_{FP}}{C_{FP} + C_{FN}} \equiv \tau^*$$

O limiar ótimo é a razão entre o custo do falso positivo e o custo total de erro. Com $C_{FN} = 500$ e $C_{FP} = 15$:

$$\tau^* = \frac{15}{15 + 500} \approx 0{,}029$$

Isso significa: sinalizar qualquer transação para a qual o modelo atribui pelo menos 2,9% de probabilidade de fraude. O limiar padrão de 0,5 exigiria 50% de probabilidade a posteriori — um padrão astronomicamente conservador dado que a probabilidade a priori é de apenas 0,1%.

### 4.3 O limiar padrão codifica uma premissa implícita

O limiar $\tau = 0{,}5$ não é neutro. Ele é o limiar ótimo apenas quando $C_{FP} = C_{FN}$. Usá-lo em um contexto onde essa premissa é violada equivale a declarar implicitamente que bloquear uma transação legítima e deixar passar uma fraude custam o mesmo — uma declaração que nenhuma equipe de fraude endossaria explicitamente, mas que muitos pipelines de avaliação fazem por padrão.

Essa observação motiva o restante do artigo. Uma vez que o limiar ótimo é derivado, a métrica de avaliação deve ser desenhada para refletir o desempenho em $\tau^*$ e em torno dele, não no arbitrário $\tau = 0{,}5$.

---

## 5. F1 e a Família F-beta

F1 é a métrica composta mais amplamente utilizada para classificação desbalanceada. Esta seção a deriva de primeiros princípios, explica o que ela realmente mede e declara seus limites honestamente.

### 5.1 A média harmônica

A média aritmética de dois números pondera cada valor linearmente. A média harmônica penaliza desbalanceamento extremo: se um valor é muito pequeno, a média harmônica é puxada fortemente para baixo.

$$F_1 = \frac{2 \cdot P \cdot R}{P + R} = \frac{2}{\frac{1}{P} + \frac{1}{R}}$$

Esta é precisamente a média harmônica de Precision e Recall. Sua propriedade-chave é que um modelo não pode alcançar um F1 alto se destacando em uma métrica enquanto ignora a outra. Um modelo com $P = 0{,}99, R = 0{,}01$ tem $F_1 \approx 0{,}02$, não 0,50.

F1 também é expressável em termos de contagens da matriz de confusão:

$$F_1 = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

Essa formulação torna a ponderação implícita explícita: FP e FN são tratados simetricamente. Cada predição falsa de qualquer tipo contribui igualmente ao denominador.

### 5.2 O que "peso igual a Precision e Recall" significa

A frase "peso igual" soa equilibrada. Em termos de custo, não é neutra. Ponderar Precision e Recall igualmente significa penalizar Falsos Positivos e Falsos Negativos identicamente — o que é outra forma de dizer $C_{FP} = C_{FN}$. Esta é exatamente a mesma premissa embutida no limiar padrão $\tau = 0{,}5$.

F1 é consistente com esse limiar e essa premissa de custo. Quando essas premissas são violadas — como são em detecção de fraude — F1 fornece uma resposta coerente para a *pergunta errada*.

### 5.3 A generalização F-beta

A família F-beta generaliza o F1 introduzindo um parâmetro $\beta$ que controla a importância relativa do Recall sobre Precision:

$$F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$

Quando $\beta = 1$, reduz-se ao F1. Quando $\beta > 1$, Recall é ponderado mais fortemente — apropriado quando perder um positivo é mais custoso. Quando $\beta < 1$, Precision é ponderada mais fortemente — apropriado quando falsos alarmes são mais custosos.

Para detecção de fraude com $C_{FN} \gg C_{FP}$, valores de $\beta$ na faixa 2–5 são comuns, refletindo que perder uma fraude importa mais do que gerar um falso alarme. No entanto, o mapeamento de razões de custo para valores de $\beta$ não é direto — $\beta$ controla a ponderação da métrica, não a ponderação de custos — então Precision@Recall (Seção 8) fornece uma conexão mais principiada com a restrição operacional real.

### 5.4 Uma declaração honesta sobre F1

F1 é uma estatística resumo útil. Ela evita a armadilha da accuracy em dados desbalanceados, penaliza modelos unilaterais e fornece um único número para comparação de modelos. Essas são vantagens genuínas.

Sua limitação é que sua premissa de custo implícita — custos iguais para FP e FN — raramente é examinada. A maioria dos praticantes que usam F1 não estão conscientemente endossando custos iguais; estão seguindo uma convenção. A convenção é aceitável quando os custos são similares. Em detecção de fraude e domínios similares de alta assimetria, não é aceitável — e os experimentos na Seção 9 demonstram que isso altera rankings de modelos.

---

## 6. A Curva ROC

A curva Receiver Operating Characteristic (ROC) fornece um resumo visual do desempenho do classificador ao longo de todos os limiares possíveis. Entender sua geometria e suas limitações é essencial para o argumento que segue.

### 6.1 Definição e AUC-ROC

Conforme o limiar de decisão $\tau$ varia de 1 (não predizer nada) a 0 (predizer tudo), tanto a Taxa de Verdadeiro Positivo (Recall) quanto a Taxa de Falso Positivo mudam. A curva ROC plota $TPR(\tau)$ contra $FPR(\tau)$ para todos os valores de $\tau$:

$$TPR = \frac{TP}{TP + FN} = R \qquad FPR = \frac{FP}{FP + TN}$$

Um classificador aleatório traça a diagonal de $(0,0)$ a $(1,1)$. Um classificador perfeito alcança $(0,1)$ — zero falsos positivos, todos os verdadeiros positivos capturados — antes que qualquer falso positivo ocorra.

A Área Sob a Curva ROC (AUC-ROC) é a integral dessa curva:

$$\text{AUC-ROC} = \int_0^1 TPR \, d(FPR)$$

AUC-ROC tem uma interpretação probabilística limpa: é a probabilidade de que o modelo atribua um score mais alto a uma instância positiva (fraude) escolhida aleatoriamente do que a uma instância negativa (legítima) escolhida aleatoriamente. Um valor de 0,5 indica ranking aleatório; 1,0 indica discriminação perfeita.

### 6.2 Origem histórica: Teoria de Detecção de Sinal

A análise ROC não foi inventada para machine learning. Ela teve origem na **Teoria de Detecção de Sinal**, desenvolvida durante a Segunda Guerra Mundial para analisar o desempenho de operadores de radar. Dado um sinal ruidoso, com que frequência o operador detecta corretamente uma aeronave inimiga (TPR) versus reporta incorretamente um falso alarme (FPR)? A mesma pergunta surge em radiologia (detecção de tumores), sismologia (identificação de terremotos) e detecção de fraude.

O contexto original da ROC é instrutivo: o operador era um tomador de decisão humano enfrentando o mesmo problema de custo assimétrico que estamos abordando. Uma aeronave inimiga não detectada (Falso Negativo) tinha consequências muito diferentes de um falso alarme (Falso Positivo) no radar. A teoria de detecção de sinal desenvolveu a noção de um ponto de operação ótimo na curva ROC que dependia desses custos e da taxa base — as mesmas quantidades que derivamos nas Seções 3 e 4.

### 6.3 Uma ressalva crítica: ROC e desbalanceamento de classes

Apesar de sua elegância probabilística, AUC-ROC tem uma limitação bem documentada no contexto de datasets desbalanceados. A Taxa de Falso Positivo $FPR = FP / (FP + TN)$ é normalizada pelo número de negativos. Quando a classe negativa é esmagadoramente grande (como em detecção de fraude), mesmo um grande número absoluto de falsos positivos produz um FPR pequeno.

Isso significa que classificadores com perfis de precision substancialmente diferentes — e portanto custos reais diferentes — podem parecer indistinguíveis em uma curva ROC. A próxima seção demonstra isso com a curva Precision-Recall, que não possui essa propriedade de mascaramento.

---

## 7. A Curva Precision-Recall

A curva Precision-Recall (PR) plota Precision contra Recall conforme o limiar $\tau$ varia. Diferente da curva ROC, ela não tem termo de FPR e é portanto totalmente sensível ao desempenho de falsos positivos mesmo quando a classe negativa domina.

### 7.1 Definição e AUC-PR

A curva PR plota $P(\tau)$ contra $R(\tau)$:

$$\text{AUC-PR} = \int_0^1 P \, dR$$

Um classificador aleatório (não informativo) em um dataset com taxa base $\pi$ produz uma curva PR plana na altura $\pi$ — porque predições aleatórias rendem uma fração de Verdadeiros Positivos proporcional à taxa base, independente do limiar. Isso torna a linha de base para AUC-PR aproximadamente igual a $\pi$.

Para detecção de fraude com $\pi = 0{,}001$, um classificador aleatório alcança AUC-PR $\approx 0{,}001$. Um modelo útil precisa alcançar AUC-PR substancialmente acima disso. Para ROC, por contraste, a linha de base aleatória é sempre 0,5, independente da taxa base — o que torna mais difícil avaliar visualmente se um modelo está fazendo algo útil.

### 7.2 Por que curvas PR são mais informativas para detecção de fraude

Davis e Goadrich (2006) provaram uma relação formal entre curvas ROC e PR: um modelo que domina outro no espaço ROC também domina no espaço PR, mas a recíproca não é verdadeira. Modelos podem ser separados no espaço PR que parecem idênticos no espaço ROC.

Isso acontece precisamente no regime desbalanceado. Quando há muito mais negativos que positivos, o eixo FPR da curva ROC comprime a distinção entre classificadores que diferem significativamente em seus perfis de precision. A curva PR torna essa distinção visível.

### 7.3 O formato do trade-off e seu significado de negócio

O formato da curva PR codifica um tipo específico de informação de negócio. Conforme Recall aumenta (o modelo é pressionado a capturar mais fraudes ao reduzir $\tau$), Precision tipicamente diminui (mais falsos alarmes são gerados). A curva PR é a fronteira de trade-off.

Uma curva PR côncava indica que o modelo pode ser operado em múltiplos pontos na fronteira ajustando $\tau$. A escolha de negócio — quanta precision pode ser sacrificada para alcançar um dado alvo de recall — é uma decisão que pertence ao stakeholder, não ao modelo. A curva PR fornece a informação necessária para tomar essa decisão explicitamente.

Essa observação prepara o terreno para a métrica final.

---

## 8. Precision@Recall — Fixando o Ponto de Operação

Precision@Recall é o clímax teórico e narrativo deste artigo. É onde a maquinaria probabilística das Seções 2–4, o panorama de métricas das Seções 5–7 e a realidade de negócio da detecção de fraude convergem.

### 8.1 Definição formal

Dado um alvo de recall $r \in (0, 1)$, Precision@Recall é definido como:

$$P\text{@}r = P(\tau_r) \quad \text{onde} \quad \tau_r = \arg\min_\tau \lvert R(\tau) - r \rvert$$

Em palavras: encontre o limiar $\tau_r$ que alcança o nível de recall $r$; então reporte a precision naquele limiar. Na curva PR, este é a coordenada de Precision do ponto onde a curva cruza a linha horizontal $R = r$.

### 8.2 O parâmetro de negócio: r não é fixo, é escolhido

O valor $r$ não é um parâmetro técnico — é uma **decisão de negócio**. Ele codifica a resposta à pergunta: *"Que fração de fraudes estamos dispostos a deixar passar?"*

Em um banco de varejo com prevenção agressiva de fraude, $r = 0{,}90$ pode ser o recall mínimo aceitável: pelo menos 90% das fraudes devem ser capturadas. Em um ambiente de maior fricção e menor volume, $r = 0{,}95$ ou até $r = 0{,}99$ pode ser apropriado. Por outro lado, em um contexto onde falsos positivos são muito custosos (transações de alto valor com clientes sensíveis), $r = 0{,}70$ pode ser escolhido para preservar precision.

O ponto crucial é que essa decisão é feita pelo negócio, não pelo algoritmo. O papel do cientista de dados é fornecer a curva PR (a fronteira de trade-off) e computar Precision@$r$ para o $r$ definido pelo stakeholder. Essa separação de papéis — o modelo fornece a fronteira; o negócio escolhe o ponto de operação — é a interface mais honesta possível entre machine learning e implantação.

### 8.3 Conexão com a teoria de decisão sensível a custos

Precision@Recall se conecta diretamente à derivação do limiar ótimo na Seção 4. Lembre que $\tau^* = C_{FP} / (C_{FP} + C_{FN})$. O recall alcançado em $\tau^*$ define o recall de operação ótimo em custo:

$$r^* = R(\tau^*)$$

Se o negócio fez uma especificação explícita de custos (Seção 4.1), então $r^*$ é o alvo de recall economicamente justificado, e $P\text{@}r^*$ é a precision que o modelo alcança no ponto de operação ótimo em custo. Esta é uma descrição mais completa do desempenho do modelo do que qualquer estatística resumo única sobre todos os limiares.

### 8.4 F1 versus Precision@Recall: uma distinção fundamental

F1 é uma otimização irrestrita: identifica o limiar que maximiza a média harmônica de Precision e Recall. Esse limiar pode ou não coincidir com o ponto de operação requerido pelo negócio.

Precision@$r$ é uma formulação restrita: fixa o requisito de recall e maximiza precision sujeito a essa restrição. Esta é a formulação correta quando o requisito de recall é não negociável (ex.: mínimos regulatórios ou contratuais sobre taxas de detecção de fraude).

A distinção importa para o ranking de modelos. A Seção 9 (Experimento E) demonstra que rankear modelos por F1 e rankeá-los por Precision@$r$ pode produzir ordenações diferentes. O modelo que maximiza F1 não é necessariamente o modelo que entrega a maior precision no alvo de recall requerido. Quando um piso de recall é fixado pelo negócio, Precision@$r$ é a métrica honesta; F1 não está errado, mas responde a uma pergunta diferente.

---

## 9. Experimentos

Os experimentos a seguir foram conduzidos em um dataset sintético desbalanceado (n = 100.000; 20 features, 5 informativas; taxa de fraude = 0,1%). Todo código, configuração e figuras são reproduzíveis executando `python scripts/run_all.py` a partir da raiz do repositório com um ambiente corretamente configurado. Seeds, hiperparâmetros, parâmetros de custo e configurações do dataset são especificados em `config.yaml`.

Todos os experimentos usam a mesma divisão de dataset: 80% treino, 20% teste, estratificado por classe. Seeds aleatórias são fixas ao longo de todo o processo. Os modelos são treinados no conjunto de treino; todas as métricas são reportadas no conjunto de teste reservado.

---

### Experimento A — A Armadilha da Accuracy

**Afirmação teórica**: Em um dataset severamente desbalanceado, um classificador degenerado (que sempre prediz negativo) alcança accuracy muito alta mas recall zero. Accuracy é portanto uma métrica de avaliação inválida para detecção de fraude.

**Setup**: Um `DummyClassifier` (sempre prediz a classe majoritária) e um modelo de `LogisticRegression` são avaliados no dataset sintético de fraude (n = 100.000; taxa de fraude = 0,1%). Seis métricas são reportadas: Accuracy, Precision, Recall, F1, AUC-ROC e AUC-PR.

**Figura**:

![Experimento A: A Armadilha da Accuracy — comparando DummyClassifier vs LogisticRegression em seis métricas de avaliação.](../figures/exp_a_accuracy_trap.png)

*Figura 1. Gráfico de barras comparando um DummyClassifier (sempre prediz a classe majoritária) com um modelo de LogisticRegression em seis métricas. O DummyClassifier alcança Accuracy = 0,999 mas Precision, Recall e F1 são todos zero (destacados com anotações). Seu AUC-ROC é 0,500 (chance aleatória) e seu AUC-PR corresponde à taxa base (0,001). LogisticRegression sacrifica accuracy (0,926) em troca de Recall significativo (0,850) e AUC-PR (0,142).*

**Observação**: O DummyClassifier alcança accuracy próxima a $1 - \pi \approx 0{,}999$ enquanto alcança Precision = 0, Recall = 0 e F1 = 0. Seu AUC-ROC de 0,500 o identifica corretamente como um ranqueador aleatório, e seu AUC-PR de 0,001 corresponde à taxa base — o desempenho esperado de um classificador sem capacidade de discriminação. O modelo LogisticRegression aceita uma redução em accuracy (de 0,999 para 0,926) em troca de Recall significativo (0,850) e um AUC-PR duas ordens de magnitude acima da linha de base.

**Conexão teórica**: O DummyClassifier explora o problema da taxa base (Seção 3.3). Sua accuracy é igual a $1 - \pi$, que é muito alta quando fraude é rara. O F1 de zero o identifica corretamente como inútil — não porque F1 é a métrica ótima, mas porque ele condiciona na classe positiva (através de Recall e Precision), o que accuracy não faz. Notavelmente, AUC-PR fornece o resumo de número único mais honesto: o DummyClassifier pontua 0,001 (a taxa base) enquanto LogisticRegression pontua 0,142.

---

### Experimento B — ROC vs. Curvas Precision-Recall

**Afirmação teórica**: No regime desbalanceado, modelos que parecem similares em uma curva ROC são revelados como substancialmente diferentes em uma curva Precision-Recall. Além disso, o próprio ranking de modelos pode diferir entre os dois espaços.

**Setup**: Três modelos são avaliados — `LogisticRegression`, `RandomForestClassifier` e um `DecisionTreeClassifier` fraco (profundidade = 2) — no dataset sintético de fraude (n = 100.000; taxa de fraude = 0,1%; 5 features informativas de 20). Ambas as curvas ROC e PR são plotadas com proporções quadradas idênticas para comparação visual direta. As curvas incluem marcadores para distinção em impressão em escala de cinza.

**Figura**:

![Experimento B — Parte 1: Curvas ROC para três modelos.](../figures/exp_b_roc_curves.png)

*Figura 2. Curvas ROC para três classificadores. Os valores de AUC-ROC são comprimidos: Logistic Regression (0,951), Random Forest (0,917), Classificador Fraco (0,876). Todos os modelos parecem competitivos no espaço ROC. O ranking sugere que Logistic Regression é o melhor modelo.*

**Figura**:

![Experimento B — Parte 2: Curvas Precision-Recall para três modelos.](../figures/exp_b_pr_curves.png)

*Figura 3. Curvas Precision-Recall para os mesmos três classificadores. A linha de base horizontal marca a taxa base (anotada com seta). Os valores de AUC-PR são muito mais dispersos: Random Forest (0,426), Classificador Fraco (0,409), Logistic Regression (0,122). O ranking é invertido: o modelo que parecia melhor no espaço ROC é o pior no espaço PR. Linhas verticais marcam os alvos de recall usados nos Experimentos D e E.*

**Observação**: As curvas ROC e PR produzem **rankings de modelos opostos**. No espaço ROC, Logistic Regression lidera (AUC-ROC = 0,951); no espaço PR, é o modelo mais fraco (AUC-PR = 0,122). Essa inversão ocorre porque o modelo linear tem dificuldade com a fronteira de decisão não linear criada pelo conjunto reduzido de features (5 de 20 features são informativas), mas sua discriminação ao longo de toda a distribuição de scores — que ROC mede — permanece forte. A curva PR, que é sensível à precision na faixa de operação relevante, revela essa fraqueza.

**Conexão teórica**: Este resultado é uma confirmação mais forte do que o esperado da Seção 7.2. Davis e Goadrich (2006) provaram que modelos podem ser separados no espaço PR enquanto parecem idênticos no espaço ROC. Aqui observamos o caso mais dramático: modelos são **inversamente ranqueados** entre os dois espaços. O eixo FPR da curva ROC normaliza falsos positivos pelo número de negativos, que são esmagadoramente numerosos (99.900 transações legítimas). Um grande aumento absoluto em falsos positivos — que afeta diretamente precision — produz apenas um aumento marginal em FPR.

---

### Experimento C — Seleção de Limiar

**Afirmação teórica**: O limiar padrão de $\tau = 0{,}5$ raramente é ótimo sob custos assimétricos. O limiar ótimo em custo $\tau^*$ desloca o ponto de operação para maior Recall ao custo de menor Precision, reduzindo o custo esperado.

**Setup**: Um modelo de `LogisticRegression` é avaliado em três pontos de operação: o limiar padrão ($\tau = 0{,}5$), o limiar ótimo-F1 (encontrado por grid search) e o limiar ótimo em custo $\tau^* = C_{FP} / (C_{FP} + C_{FN})$ usando os valores de custo em `config.yaml` ($C_{FP}$ = \$5, $C_{FN}$ = \$200). Um gráfico de varredura de limiar e matrizes de confusão para os três limiares são mostrados.

**Figura**:

![Experimento C — Parte 1: Varredura de limiar mostrando Precision, Recall e F1 como função do limiar.](../figures/exp_c_threshold_sweep.png)

*Figura 4. Varredura de limiar para um modelo LogisticRegression. Precision, Recall e F1 são plotados como funções do limiar de decisão. Linhas verticais marcam três pontos de operação: o limiar padrão ($\tau = 0{,}5$), o limiar ótimo-F1 ($\tau = 0{,}990$) e o limiar ótimo em custo ($\tau^* = 0{,}024$). Precision e F1 permanecem próximos de zero para a maior parte da faixa de limiar e só sobem abruptamente acima de $\tau = 0{,}9$, indicando que neste problema altamente desbalanceado, a distribuição de scores do modelo está concentrada próximo a 1,0 para casos de fraude.*

**Figura**:

![Experimento C — Parte 2: Matrizes de confusão em três limiares diferentes.](../figures/exp_c_confusion_matrices.png)

*Figura 5. Matrizes de confusão (normalizadas por linha) nos três pontos de operação. Em $\tau = 0{,}5$ (esquerda): Recall = 0,850 mas Precision = 0,011 (1.470 falsos positivos). Em $\tau = 0{,}990$ / ótimo-F1 (centro): Recall cai para 0,400 e Precision sobe para 0,078 (94 falsos positivos). Em $\tau^* = 0{,}024$ / ótimo em custo (direita): Recall aumenta para 0,900 com Precision = 0,003 (5.824 falsos positivos). Cada matriz mostra tanto contagens brutas quanto taxas normalizadas por linha.*

**Observação**: Os três limiares produzem matrizes de confusão qualitativamente diferentes. O limiar padrão ($\tau = 0{,}5$) alcança 85% de recall mas ao custo de 1.470 falsos positivos — uma Precision de apenas 1,1%. O limiar ótimo-F1 ($\tau = 0{,}990$) reduz falsos positivos drasticamente (para 94) mas sacrifica recall para 40%, perdendo a maioria das fraudes. O limiar ótimo em custo ($\tau^* = 0{,}024$) maximiza recall (90%) ao custo de 5.824 falsos positivos — o trade-off correto dado que cada fraude perdida custa 40 vezes mais que um falso alarme.

Um resultado notável é que o limiar ótimo-F1 é 0,990 — muito acima do padrão de 0,5. Isso não é um erro; reflete a distribuição de scores do modelo sob desbalanceamento extremo de classes. A vasta maioria dos scores está próxima de zero (para transações legítimas), e a média harmônica F1 atinge seu pico apenas quando o limiar é alto o suficiente para reduzir substancialmente os falsos positivos.

**Conexão teórica**: Esta é uma confirmação empírica direta da Seção 4.3: o limiar padrão codifica a premissa implícita $C_{FP} = C_{FN}$, que é falsa aqui. O limiar ótimo em custo segue da derivação na Seção 4.2 e produz uma matriz de confusão qualitativamente diferente. A distância entre os limiares ótimo-F1 e ótimo em custo (0,990 vs. 0,024) ilustra quão longe o critério F1 diverge do ponto de operação economicamente justificado.

---

### Experimento D — Precision@Recall em Alvos Definidos pelo Negócio

**Afirmação teórica**: Diferentes contextos de negócio implicam diferentes requisitos de recall, e a precision alcançável em cada requisito é uma propriedade do modelo que deve ser avaliada explicitamente. O trade-off tem consequências financeiras diretas.

**Setup**: Um `RandomForestClassifier` (o melhor modelo do Experimento B por AUC-PR) é avaliado em cinco alvos de recall: $r \in \lbrace 0{,}75;\ 0{,}80;\ 0{,}85;\ 0{,}90;\ 0{,}95 \rbrace$. Precision, contagem de falsos positivos e contagem de falsos negativos são computados em cada ponto de operação. O custo de negócio esperado ($C_{FP}$ = \$5, $C_{FN}$ = \$200) é anotado na figura.

**Figura**:

![Experimento D: Precision@Recall em cinco alvos de recall definidos pelo negócio com anotações de custo.](../figures/exp_d_precision_at_recall.png)

*Figura 6. Painel esquerdo: Precision em cada alvo de recall. Precision declina de 0,072 em r = 0,75 para 0,011 em r = 0,95. Painel direito: Contagens de falsos positivos e falsos negativos, com anotações de custo esperado acima de cada grupo. Em r = 0,75, o modelo gera 193 falsos positivos e 5 falsos negativos (custo total: \$1.965). Em r = 0,90, falsos positivos saltam para 1.480 e o custo quadruplica para \$8.000. A transição entre r = 0,85 e r = 0,90 marca um ponto de inflexão acentuado na curva de custo.*

**Observação**: Precision declina de forma constante de r = 0,75 a r = 0,85 (7,2% a 5,7%), mas cai abruptamente em r = 0,90 (para 0,8%) conforme o modelo esgota suas predições positivas de alta confiança e começa a sinalizar transações de baixa confiança. A análise de custo revela um ponto de inflexão: abaixo de r = 0,85, o custo esperado total é aproximadamente \$2.000; acima de r = 0,90, salta para \$8.000. Essa não linearidade significa que a decisão de negócio entre 85% e 90% de recall tem um impacto de custo de 4x — informação que é invisível em uma métrica resumo como AUC-PR.

**Conexão teórica**: Este experimento operacionaliza a Seção 8.2: $r$ é uma decisão de negócio, e Precision@$r$ é a métrica que reflete a qualidade do modelo naquele ponto de decisão. As anotações de custo se conectam diretamente ao framework de matriz de custos na Seção 4.1, traduzindo contagens abstratas da matriz de confusão em termos monetários. A curva PR completa (Figura 3) é a fronteira de trade-off completa; Precision@$r$ é o ponto de operação específico nessa fronteira.

---

### Experimento E — A Escolha da Métrica Altera Rankings de Modelos

**Afirmação teórica**: Rankear modelos por F1 e rankeá-los por Precision@$r$ pode produzir ordenações diferentes. Quando um piso de recall é fixado pelo negócio, o ranking baseado em F1 pode recomendar o modelo errado.

**Setup**: Cinco variantes de modelo são treinadas — três configurações de Logistic Regression (balanceada, C = 0,01, C = 10) e duas configurações de Random Forest (profundidade = 5, profundidade = 15) — todas com pesos de classe balanceados. Cada modelo é ranqueado por sete métricas: F1 (melhor alcançável ao longo dos limiares), AUC-PR, e Precision@$r$ para $r \in \lbrace 0{,}75;\ 0{,}80;\ 0{,}85;\ 0{,}90;\ 0{,}95 \rbrace$. Os resultados são exibidos como um heatmap anotado com rankings por célula e destaque de divergência.

**Figura**:

![Experimento E: Comparação de ranking de modelos em sete métricas de avaliação.](../figures/exp_e_ranking_comparison.png)

*Figura 7. Heatmap de scores de modelos em sete métricas. Cada célula mostra o valor da métrica e o ranking ordinal (ex.: #1, #2). Células com borda laranja destacam onde o modelo top-1 difere do ranking F1. RF (profundidade = 15) é #1 por F1 (0,519); RF (profundidade = 5) é #1 por P@75R (0,146). A divergência está concentrada na coluna P@75R — os modelos que alcançam o melhor trade-off F1 não são os mesmos modelos que entregam a maior precision em um piso de recall fixo.*

**Observação**: O ranking diverge na coluna P@75R. RF (profundidade = 15) é o melhor modelo por F1 (0,519) e AUC-PR (0,449), mas RF (profundidade = 5) entrega quase o dobro de precision em 75% de recall (0,146 vs. 0,074). Isso significa que se o negócio requer que pelo menos 75% das fraudes sejam capturadas, o modelo recomendado por F1 produziria aproximadamente o dobro de falsos alarmes comparado ao modelo recomendado por P@75R. As três variantes de Logistic Regression são consistentemente mais fracas em todas as métricas, confirmando que a divergência no ranking não é um artefato de modelos fracos — ocorre entre os dois modelos de melhor desempenho no pool.

**Conexão teórica**: Este experimento é o núcleo empírico da tese do artigo. Ele conecta as Seções 5 a 8: F1 (Seção 5) e Precision@Recall (Seção 8) respondem a perguntas diferentes, e quando essas perguntas têm respostas diferentes, a escolha da métrica determina qual modelo é implantado. O heatmap torna isso concreto: o mesmo conjunto de modelos, o mesmo dataset, a mesma divisão treino-teste — mas uma métrica diferente seleciona um modelo diferente. Isso não é ruído; é informação sobre o desalinhamento entre a premissa implícita de custo igual do F1 e as restrições operacionais reais do negócio.

---

## 10. Um Framework para Escolher a Métrica Correta

As seções teóricas estabeleceram os fundamentos matemáticos; os experimentos forneceram evidência empírica. Esta seção sintetiza um framework de decisão prático.

### 10.1 A tabela de decisão

A escolha da métrica de avaliação decorre de responder a um pequeno conjunto de perguntas sobre o contexto de implantação:

| Pergunta | Resposta | Métrica Recomendada |
|---|---|---|
| As distribuições de classes são balanceadas? | Sim | Accuracy, F1 |
| As distribuições de classes são balanceadas? | Não | Continue abaixo |
| Um piso de recall de negócio está definido? | Sim | Precision@r, AUC-PR |
| Um piso de recall de negócio está definido? | Não | AUC-PR, F1 |
| Custos de erro são explicitamente estimados? | Sim | Limiar ótimo em custo + Precision@r* |
| Custos de erro são explicitamente estimados? | Não | AUC-PR (ranking); F1 no limiar ótimo-F1 |
| O objetivo é ranking de modelos (sem limiar fixo)? | Sim | AUC-PR |
| O objetivo é seleção de limiar? | Sim | Curva PR + limiar ótimo em custo |

### 10.2 Recomendações específicas para detecção de fraude

Dadas as propriedades estruturais da detecção de fraude — desbalanceamento severo de classes, custos de erro assimétricos e um requisito mínimo de recall regulatório ou comercial — a abordagem recomendada é:

1. **Durante o desenvolvimento**: ranquear modelos por AUC-PR. Isso fornece uma comparação livre de limiar que é sensível ao desempenho na classe positiva (fraude).
2. **Durante a seleção de limiar**: computar o limiar ótimo em custo $\tau^* = C_{FP} / (C_{FP} + C_{FN})$ a partir das estimativas de custo da instituição.
3. **Para avaliação de implantação**: reportar Precision@$r$ onde $r$ é o piso de recall definido pelo negócio. Este é o número que responde diretamente à pergunta operacional: na taxa de detecção requerida, que fração dos nossos alertas são fraudes reais?
4. **Para comunicação com stakeholders**: expressar resultados em termos de contagens de FP e FN e seus custos associados, não apenas percentuais. "Com 90% de recall, geramos aproximadamente 47 falsos alarmes por dia a um custo combinado de investigação de R\$X" é mais acionável do que "P@90R = 0,15."

### 10.3 Erros comuns e como evitá-los

**Erro 1: Usar accuracy em um dataset desbalanceado.** O Experimento A demonstra que accuracy recompensa modelos que ignoram a classe minoritária. Sempre verifique se um DummyClassifier alcançaria accuracy competitiva antes de usar accuracy como métrica.

**Erro 2: Usar AUC-ROC como métrica primária para datasets desbalanceados.** AUC-ROC é um complemento útil mas pode mascarar grandes diferenças nos perfis de precision (Experimento B). Use AUC-PR como métrica primária de ranking.

**Erro 3: Avaliar no limiar padrão de 0,5.** O limiar padrão é ótimo apenas quando custos são simétricos e a taxa base é 0,5. Em detecção de fraude, $\tau = 0{,}5$ tipicamente produz recall próximo de zero (Experimento C). Sempre reporte qual limiar foi usado e por quê.

**Erro 4: Escolher uma métrica sem um requisito de negócio declarado.** F1 é um default razoável *quando nenhuma outra informação está disponível*. Se um piso de recall é conhecido, Precision@$r$ é mais informativo. Se custos são conhecidos, análise de limiar ótimo em custo é mais informativa. A métrica deve ser compatível com a informação disponível.

### 10.4 O princípio de encerramento

A métrica correta é uma decisão de negócio codificada em matemática. Toda métrica de avaliação codifica premissas sobre custos, distribuições de classes e requisitos operacionais. Tornar essas premissas explícitas — em vez de herdá-las de convenções — é a prática mais importante que um praticante de ML pode adotar.

As ferramentas matemáticas para isso não são exóticas. São o teorema de Bayes, probabilidade condicional e uma matriz de custos. Este artigo as montou de baixo para cima. A montagem era o ponto.

---

## 11. Conclusão

### O que foi demonstrado

Este artigo partiu de uma pergunta concreta — por que F1 em detecção de fraude? — e traçou sua resposta através de quatro corpos de conhecimento interligados:

1. **Probabilidade condicional** revelou que Precision e Recall medem coisas qualitativamente diferentes: a probabilidade a posteriori de fraude dada uma predição positiva, e a verossimilhança de uma predição positiva dada a fraude. Combiná-las requer tornar explícita uma decisão de ponderação.

2. **Teorema de Bayes** conectou a taxa base à tensão entre Precision e Recall, mostrando que desbalanceamento de classes é um problema de taxa base e que qualquer métrica que ignore a taxa base não pode caracterizar completamente o desempenho do modelo em cenários desbalanceados.

3. **Teoria de decisão sensível a custos** derivou o limiar ótimo de primeiros princípios, mostrando que o limiar convencional de 0,5 é um caso especial que vale apenas quando os dois tipos de erro custam o mesmo.

4. **O panorama de métricas** — ROC, PR e Precision@Recall — mapeou o espaço de escolhas de avaliação e demonstrou empiricamente que a escolha da métrica altera rankings de modelos e decisões de implantação.

### Conclusões práticas

Essas conclusões são condicionais. Aplique-as quando as condições forem atendidas.

- Quando o desbalanceamento de classes é severo, nunca use accuracy como métrica de avaliação primária.
- Ao comparar modelos sem um limiar fixo, prefira AUC-PR sobre AUC-ROC.
- Quando um piso de recall de negócio é definido, avalie com Precision@$r$ naquele piso.
- Quando custos de erro são estimados, derive o limiar ótimo em custo e avalie nele.
- Ao reportar resultados para stakeholders, traduza valores de métricas em contagens de FP/FN e seus custos de negócio associados.
- Ao escolher entre dois modelos, pergunte qual métrica melhor representa o contexto de implantação antes de consultar o leaderboard.

### O que vem a seguir

Para o leitor: a extensão natural deste trabalho é para cenários dinâmicos e sequenciais — fluxos de fraude com distribuições não estacionárias, onde a taxa base muda ao longo do tempo e o limiar ótimo precisa ser recalibrado. O framework bayesiano desenvolvido aqui se estende naturalmente para aprendizado online e tomada de decisão sequencial.

Para o modelo e dataset usados aqui: os experimentos foram executados em um snapshot estático. Detecção de fraude em produção opera sobre dados de transação em evolução com concept drift, atraso de rótulos (chargebacks podem chegar semanas após a transação) e loops de feedback adversariais. O framework de métricas está correto para o caso estático; estendê-lo para o caso dinâmico é um problema mais rico.

Para o autor: este artigo foi escrito em paralelo com o aprendizado. A próxima iteração incorporará calibração — o grau em que as estimativas de probabilidade do modelo $\hat{p}(x)$ são confiáveis — porque a derivação do limiar ótimo em custo na Seção 4 depende inteiramente de $\hat{p}(x)$ ser uma probabilidade a posteriori bem calibrada.

---

## Referências

[1] Bayes, T. (1763). *An Essay towards Solving a Problem in the Doctrine of Chances*. Philosophical Transactions of the Royal Society of London, 53, 370–418.

[2] Davis, J. & Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. *Proceedings of the 23rd International Conference on Machine Learning (ICML)*, 233–240. https://doi.org/10.1145/1143844.1143874

[3] Fawcett, T. (2006). An Introduction to ROC Analysis. *Pattern Recognition Letters*, 27(8), 861–874. https://doi.org/10.1016/j.patrec.2005.10.010

[4] Saito, T. & Rehmsmeier, M. (2015). The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432

[5] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

[6] Green, D.M. & Swets, J.A. (1966). *Signal Detection Theory and Psychophysics*. Wiley. *(Formulação original da análise ROC a partir da Teoria de Detecção de Sinal.)*

[7] Dal Pozzolo, A., Caelen, O., Johnson, R.A., & Bontempi, G. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. *2015 IEEE Symposium Series on Computational Intelligence*, 159–166. https://doi.org/10.1109/SSCI.2015.33

[8] ULB Machine Learning Group (2018). *Credit Card Fraud Detection* [Dataset]. Kaggle. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

*Código-fonte e dados: [github.com/brunoramosmartins/precision-recall-fraud](https://github.com/brunoramosmartins/precision-recall-fraud)*

*Glossário de notação: [article/notation.md](notation.md)*

*Referências BibTeX: [article/references.bib](references.bib)*
